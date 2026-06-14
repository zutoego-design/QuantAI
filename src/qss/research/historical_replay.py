from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from qss.backtest.engine import (
    BacktestRunSpec,
    _attach_reference_benchmarks,
    _prepare_ledger_market_data,
    _simulate_ledger,
    load_backtest_data,
    run_backtest,
)
from qss.backtest.metrics import (
    compute_backtest_metrics,
    factor_diagnostics,
    forward_returns_for_factors,
)
from qss.config.schema import AppConfig
from qss.data.storage import resolve_path, write_csv, write_parquet
from qss.research.portfolio_evaluation import (
    PortfolioEvaluation,
    simulate_score_portfolio,
    targets_from_scores,
)
from qss.research.snapshot import build_data_snapshot, write_data_snapshot
from qss.runs.manifest import config_hash, create_run_context

REPLAY_SCOPE = "exploratory_historical_replay"
CONTROL_ID = "v1_control"


class ReplayFold(BaseModel):
    development_start: str
    development_end: str
    test_start: str
    test_end: str

    @field_validator(
        "development_start",
        "development_end",
        "test_start",
        "test_end",
        mode="before",
    )
    @classmethod
    def normalize_date(cls, value: Any) -> str:
        return str(value)

    @model_validator(mode="after")
    def validate_order(self) -> "ReplayFold":
        development_start = pd.Timestamp(self.development_start)
        development_end = pd.Timestamp(self.development_end)
        test_start = pd.Timestamp(self.test_start)
        test_end = pd.Timestamp(self.test_end)
        if not development_start < development_end < test_start <= test_end:
            raise ValueError("Replay fold dates must be ordered and non-overlapping.")
        return self

    @property
    def fold_id(self) -> str:
        return str(pd.Timestamp(self.test_start).year)


class SelectionRules(BaseModel):
    minimum_positive_years: int = 5
    minimum_spy_outperformance_years: int = 4
    minimum_sharpe_improvement: float = 0.10
    maximum_drawdown_deterioration: float = 0.02
    robustness_maximum_sharpe_decline: float = 0.30
    required_cost_bps: float = 25.0


class ReplaySuite(BaseModel):
    study_id: str
    evaluation_scope: str = REPLAY_SCOPE
    candidates: list[str]
    folds: list[ReplayFold]
    selection: SelectionRules = Field(default_factory=SelectionRules)

    @model_validator(mode="after")
    def validate_suite(self) -> "ReplaySuite":
        if self.evaluation_scope != REPLAY_SCOPE:
            raise ValueError(f"Replay evaluation_scope must be {REPLAY_SCOPE}.")
        if len(self.folds) != 7:
            raise ValueError("Historical replay requires exactly seven annual folds.")
        test_ranges = [
            (pd.Timestamp(fold.test_start), pd.Timestamp(fold.test_end))
            for fold in self.folds
        ]
        for index, (start, end) in enumerate(test_ranges):
            for other_start, other_end in test_ranges[index + 1 :]:
                if start <= other_end and other_start <= end:
                    raise ValueError("Historical replay test windows must not overlap.")
        return self


@dataclass(frozen=True)
class CandidateDefinition:
    candidate_id: str
    description: str
    path: Path
    file_sha256: str
    config: AppConfig


@dataclass
class CandidateReplay:
    definition: CandidateDefinition
    source_run_id: str
    source_run_path: Path
    source_daily: pd.DataFrame
    source_rebalances: pd.DataFrame
    scores: pd.DataFrame
    factors: pd.DataFrame
    fold_metrics: pd.DataFrame
    combined_daily: pd.DataFrame
    combined_rebalances: pd.DataFrame
    summary: dict[str, Any]


@dataclass
class HistoricalReplayResult:
    run_id: str
    run_path: Path
    selected_strategy_id: str | None
    challenger_strategy_id: str | None
    decision: str


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return payload


def _deep_merge(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in incoming.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_replay_suite(path: str | Path) -> ReplaySuite:
    return ReplaySuite.model_validate(_read_yaml(resolve_path(path)))


def load_candidate_definition(
    base_config: AppConfig,
    path: str | Path,
) -> CandidateDefinition:
    candidate_path = resolve_path(path)
    payload = _read_yaml(candidate_path)
    candidate_id = str(payload.get("candidate_id", "")).strip()
    if not candidate_id:
        raise ValueError(f"Candidate {candidate_path} is missing candidate_id.")
    mode = payload.get("factor_groups_mode", "merge")
    if mode not in {"merge", "replace"}:
        raise ValueError(f"Candidate {candidate_id} has invalid factor_groups_mode.")
    overrides = payload.get("overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError(f"Candidate {candidate_id} overrides must be a mapping.")
    base_payload = base_config.model_dump(mode="json")
    if mode == "replace" and "factor_groups" in overrides:
        base_payload["factor_groups"] = {}
    resolved = AppConfig.model_validate(_deep_merge(base_payload, overrides))
    return CandidateDefinition(
        candidate_id=candidate_id,
        description=str(payload.get("description", "")),
        path=candidate_path,
        file_sha256=_sha256(candidate_path),
        config=resolved,
    )


def _metric_map(metrics: pd.DataFrame) -> dict[str, float]:
    if metrics.empty:
        return {}
    return {
        str(row.metric): float(row.value)
        for row in metrics.itertuples(index=False)
    }


def _spy_metrics(
    daily: pd.DataFrame,
    rebalances: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    if "secondary_benchmark_return" not in frame:
        raise ValueError("Historical replay requires SPY secondary benchmark returns.")
    frame["benchmark_return"] = pd.to_numeric(
        frame["secondary_benchmark_return"],
        errors="coerce",
    )
    if frame["benchmark_return"].isna().any():
        raise ValueError("SPY benchmark returns are incomplete in a replay fold.")
    metrics = compute_backtest_metrics(frame, rebalances)
    return frame, _metric_map(metrics)


def _fold_rebalances(
    rebalances: pd.DataFrame,
    start: str,
    end: str,
) -> pd.DataFrame:
    if rebalances.empty:
        return rebalances.copy()
    frame = rebalances.copy()
    date_column = "execution_date" if "execution_date" in frame else "signal_date"
    dates = pd.to_datetime(frame[date_column]).dt.normalize()
    return frame.loc[dates.between(pd.Timestamp(start), pd.Timestamp(end))].copy()


def _evaluate_fold(
    replay_root: Path,
    definition: CandidateDefinition,
    fold: ReplayFold,
    scores: pd.DataFrame,
    prices: pd.DataFrame,
    market_data,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    fold_scores = scores.loc[
        pd.to_datetime(scores["date"]).between(
            pd.Timestamp(fold.development_start),
            pd.Timestamp(fold.test_end),
        )
    ].copy()
    output = replay_root / "candidates" / definition.candidate_id / "folds" / fold.fold_id
    evaluation = simulate_score_portfolio(
        fold_scores,
        prices,
        definition.config,
        start_date=fold.development_start,
        end_date=fold.test_end,
        exact_target_count=False,
        market_data=market_data,
    )
    test_daily = evaluation.daily_returns.loc[
        pd.to_datetime(evaluation.daily_returns["date"]).between(
            pd.Timestamp(fold.test_start),
            pd.Timestamp(fold.test_end),
        )
    ].copy()
    if test_daily.empty:
        raise ValueError(f"Fold {fold.fold_id} generated no test-period returns.")
    test_rebalances = _fold_rebalances(
        evaluation.rebalances,
        fold.test_start,
        fold.test_end,
    )
    test_daily, metrics = _spy_metrics(test_daily, test_rebalances)
    write_csv(test_daily, output / "test_daily_returns.csv")
    write_parquet(test_daily, output / "test_daily_returns.parquet")
    write_csv(test_rebalances, output / "test_rebalances.csv")
    write_csv(
        pd.DataFrame(
            [
                {"category": "replay", "metric": metric, "value": value}
                for metric, value in metrics.items()
            ]
        ),
        output / "test_metrics.csv",
    )
    return (
        {
            "candidate_id": definition.candidate_id,
            "fold": fold.fold_id,
            "development_start": fold.development_start,
            "development_end": fold.development_end,
            "test_start": fold.test_start,
            "test_end": fold.test_end,
            "status": "valid",
            "net_total_return": metrics.get("net_total_return"),
            "spy_total_return": metrics.get("benchmark_total_return"),
            "active_total_return": (
                metrics.get("net_total_return", 0.0)
                - metrics.get("benchmark_total_return", 0.0)
            ),
            "sharpe_ratio": metrics.get("sharpe_ratio"),
            "max_drawdown": metrics.get("max_drawdown"),
            "average_turnover": metrics.get("average_turnover"),
            "cost_drag": metrics.get("cost_drag"),
        },
        test_daily,
        test_rebalances,
    )


def _development_diagnostics(
    definition: CandidateDefinition,
    folds: list[ReplayFold],
    factors: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    forward = forward_returns_for_factors(factors, prices, 21)
    rows: list[pd.DataFrame] = []
    for fold in folds:
        mask = pd.to_datetime(factors["date"]).between(
            pd.Timestamp(fold.development_start),
            pd.Timestamp(fold.development_end),
        )
        development_factors = factors.loc[mask].copy()
        development_forward = forward.loc[
            pd.to_datetime(forward["date"]).between(
                pd.Timestamp(fold.development_start),
                pd.Timestamp(fold.development_end),
            )
        ].copy()
        diagnostics, _ = factor_diagnostics(
            development_factors,
            development_forward,
        )
        if diagnostics.empty:
            continue
        diagnostics.insert(0, "fold", fold.fold_id)
        diagnostics.insert(0, "candidate_id", definition.candidate_id)
        rows.append(diagnostics)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _candidate_summary(
    definition: CandidateDefinition,
    fold_metrics: pd.DataFrame,
    combined_daily: pd.DataFrame,
    combined_rebalances: pd.DataFrame,
    required_folds: int,
) -> dict[str, Any]:
    combined_daily, combined = _spy_metrics(
        combined_daily,
        combined_rebalances,
    )
    valid_folds = int(fold_metrics["status"].eq("valid").sum())
    annual_sharpes = pd.to_numeric(fold_metrics["sharpe_ratio"], errors="coerce")
    return {
        "candidate_id": definition.candidate_id,
        "description": definition.description,
        "candidate_file": str(definition.path),
        "candidate_file_sha256": definition.file_sha256,
        "config_hash": config_hash(definition.config),
        "valid_folds": valid_folds,
        "all_folds_valid": valid_folds == required_folds,
        "positive_years": int(
            pd.to_numeric(fold_metrics["net_total_return"], errors="coerce").gt(0).sum()
        ),
        "spy_outperformance_years": int(
            pd.to_numeric(fold_metrics["active_total_return"], errors="coerce").gt(0).sum()
        ),
        "median_annual_sharpe": float(annual_sharpes.median()),
        "combined_net_total_return": combined.get("net_total_return"),
        "combined_spy_total_return": combined.get("benchmark_total_return"),
        "combined_sharpe": combined.get("sharpe_ratio"),
        "combined_max_drawdown": combined.get("max_drawdown"),
        "combined_average_turnover": combined.get("average_turnover"),
        "combined_cost_drag": combined.get("cost_drag"),
    }


def _extract_cached_frames(
    cache,
    candidate_config: AppConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    namespace = cache.factor_snapshots.get(config_hash(candidate_config), {})
    if not namespace:
        raise ValueError("Candidate source run did not populate factor snapshots.")
    factors = pd.concat(
        [event["factors"] for event in namespace.values()],
        ignore_index=True,
    )
    scores = pd.concat(
        [event["scores"] for event in namespace.values()],
        ignore_index=True,
    )
    factors = factors.drop_duplicates(
        ["date", "symbol", "factor_name"],
        keep="last",
    ).sort_values(["date", "symbol", "factor_name"])
    scores = scores.drop_duplicates(
        ["date", "symbol"],
        keep="last",
    ).sort_values(["date", "rank"])
    return factors, scores


def _continuous_evaluation(
    scores: pd.DataFrame,
    prices: pd.DataFrame,
    config: AppConfig,
    start_date: str,
    end_date: str,
    evaluation_start: str,
    *,
    exact_target_count: bool = False,
    market_data=None,
) -> tuple[PortfolioEvaluation, pd.DataFrame, dict[str, float]]:
    evaluation = simulate_score_portfolio(
        scores.loc[
            pd.to_datetime(scores["date"]).between(
                pd.Timestamp(start_date),
                pd.Timestamp(end_date),
            )
        ],
        prices,
        config,
        start_date=start_date,
        end_date=end_date,
        exact_target_count=exact_target_count,
        market_data=market_data,
    )
    daily = evaluation.daily_returns.loc[
        pd.to_datetime(evaluation.daily_returns["date"]).between(
            pd.Timestamp(evaluation_start),
            pd.Timestamp(end_date),
        )
    ].copy()
    rebalances = _fold_rebalances(
        evaluation.rebalances,
        evaluation_start,
        end_date,
    )
    daily, metrics = _spy_metrics(daily, rebalances)
    return evaluation, daily, metrics


def _simulate_targets(
    targets: dict[pd.Timestamp, dict],
    prices: pd.DataFrame,
    config: AppConfig,
    start_date: str,
    end_date: str,
    market_data,
) -> PortfolioEvaluation:
    daily, rebalances, holdings, trades, _ = _simulate_ledger(
        BacktestRunSpec(
            start_date=start_date,
            end_date=end_date,
            initial_capital=config.backtest.initial_capital,
            execution_lag_days=config.backtest.rebalance_execution_lag_days,
            delisting_return=0.0,
        ),
        config,
        prices,
        targets,
        config.backtest.primary_benchmark,
        market_data=market_data,
    )
    daily = _attach_reference_benchmarks(
        daily,
        prices,
        targets,
        config.backtest.secondary_benchmark,
    )
    return PortfolioEvaluation(
        daily_returns=daily,
        metrics=compute_backtest_metrics(daily, rebalances),
        rebalances=rebalances,
        holdings=holdings,
        trades=trades,
    )


def _robustness_matrix(
    replay_root: Path,
    replay: CandidateReplay,
    cache,
    rules: SelectionRules,
    full_start: str,
    full_end: str,
    evaluation_start: str,
) -> pd.DataFrame:
    definition = replay.definition
    config = definition.config
    prices = cache.prices
    market_data = cache.market_data or _prepare_ledger_market_data(prices)
    cache.market_data = market_data
    matrix: list[dict[str, Any]] = []

    base_daily = replay.source_daily.loc[
        pd.to_datetime(replay.source_daily["date"]) >= pd.Timestamp(evaluation_start)
    ].copy()
    base_rebalances = _fold_rebalances(
        replay.source_rebalances,
        evaluation_start,
        full_end,
    )
    _, base_metrics = _spy_metrics(base_daily, base_rebalances)
    base_sharpe = float(base_metrics.get("sharpe_ratio", 0.0))
    matrix.append(
        {
            "candidate_id": definition.candidate_id,
            "test": "base",
            "setting": "configured",
            "status": "valid",
            "net_total_return": base_metrics.get("net_total_return"),
            "sharpe_ratio": base_sharpe,
            "max_drawdown": base_metrics.get("max_drawdown"),
            "sharpe_decline": 0.0,
        }
    )

    base_targets = targets_from_scores(
        replay.scores,
        prices,
        config,
        exact_target_count=False,
        market_data=market_data,
    )
    for cost_bps in config.robustness.cost_sensitivity_bps:
        variant = config.model_copy(deep=True)
        variant.backtest.transaction_cost.commission_bps = 0.0
        variant.backtest.transaction_cost.slippage_bps = float(cost_bps)
        variant.backtest.transaction_cost.market_impact_coefficient = 0.0
        evaluation = _simulate_targets(
            base_targets,
            prices,
            variant,
            full_start,
            full_end,
            market_data,
        )
        daily = evaluation.daily_returns.loc[
            pd.to_datetime(evaluation.daily_returns["date"]) >= pd.Timestamp(evaluation_start)
        ].copy()
        rebalances = _fold_rebalances(
            evaluation.rebalances,
            evaluation_start,
            full_end,
        )
        _, metrics = _spy_metrics(daily, rebalances)
        matrix.append(
            {
                "candidate_id": definition.candidate_id,
                "test": "cost_sensitivity",
                "setting": str(float(cost_bps)),
                "status": "valid",
                "net_total_return": metrics.get("net_total_return"),
                "sharpe_ratio": metrics.get("sharpe_ratio"),
                "max_drawdown": metrics.get("max_drawdown"),
                "sharpe_decline": (
                    (base_sharpe - float(metrics.get("sharpe_ratio", 0.0)))
                    / max(abs(base_sharpe), 1e-12)
                ),
            }
        )

    def evaluate_top_n(top_n: int) -> dict[str, Any]:
        variant = config.model_copy(deep=True)
        variant.optimizer.constraints.target_num_holdings = int(top_n)
        try:
            _, _, metrics = _continuous_evaluation(
                replay.scores,
                prices,
                variant,
                full_start,
                full_end,
                evaluation_start,
                exact_target_count=True,
                market_data=market_data,
            )
            status = "valid"
            error = ""
        except (ValueError, RuntimeError) as exc:
            metrics = {}
            status = "invalid"
            error = str(exc)
        variant_sharpe = float(metrics.get("sharpe_ratio", float("nan")))
        return {
            "candidate_id": definition.candidate_id,
            "test": "top_n_sensitivity",
            "setting": str(top_n),
            "status": status,
            "net_total_return": metrics.get("net_total_return"),
            "sharpe_ratio": metrics.get("sharpe_ratio"),
            "max_drawdown": metrics.get("max_drawdown"),
            "sharpe_decline": (
                (base_sharpe - variant_sharpe) / max(abs(base_sharpe), 1e-12)
                if pd.notna(variant_sharpe)
                else float("nan")
            ),
            "error": error,
        }

    top_n_values = [int(value) for value in config.robustness.top_n_values]
    top_n_workers = max(
        1,
        min(int(config.robustness.parallel_workers), len(top_n_values)),
    )
    with ThreadPoolExecutor(max_workers=top_n_workers) as executor:
        matrix.extend(executor.map(evaluate_top_n, top_n_values))

    for shift in config.robustness.rebalance_day_shifts:
        if shift == 0:
            continue
        try:
            result = run_backtest(
                full_start,
                full_end,
                config,
                publish_latest=False,
                enforce_data_gate=False,
                signal_day_shift=int(shift),
                data_cache=cache,
                artifact_level="metrics",
                exact_target_count=False,
                research_metadata={"evaluation_scope": REPLAY_SCOPE},
            )
            daily = result.daily_returns.loc[
                pd.to_datetime(result.daily_returns["date"])
                >= pd.Timestamp(evaluation_start)
            ].copy()
            rebalances = _fold_rebalances(
                result.rebalances,
                evaluation_start,
                full_end,
            )
            _, metrics = _spy_metrics(daily, rebalances)
            status = "valid"
            error = ""
            run_id = result.run_id
        except (ValueError, RuntimeError) as exc:
            metrics = {}
            status = "invalid"
            error = str(exc)
            run_id = ""
        variant_sharpe = float(metrics.get("sharpe_ratio", float("nan")))
        matrix.append(
            {
                "candidate_id": definition.candidate_id,
                "test": "rebalance_day_shift",
                "setting": str(shift),
                "status": status,
                "run_id": run_id,
                "net_total_return": metrics.get("net_total_return"),
                "sharpe_ratio": metrics.get("sharpe_ratio"),
                "max_drawdown": metrics.get("max_drawdown"),
                "sharpe_decline": (
                    (base_sharpe - variant_sharpe) / max(abs(base_sharpe), 1e-12)
                    if pd.notna(variant_sharpe)
                    else float("nan")
                ),
                "error": error,
            }
        )

    frame = pd.DataFrame(matrix)
    write_csv(
        frame,
        replay_root
        / "candidates"
        / definition.candidate_id
        / "robustness_matrix.csv",
    )
    return frame


def _selection_status(
    summary: dict[str, Any],
    control: dict[str, Any],
    robustness: pd.DataFrame,
    rules: SelectionRules,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if not summary["all_folds_valid"]:
        failures.append("not_all_folds_valid")
    if summary["positive_years"] < rules.minimum_positive_years:
        failures.append("insufficient_positive_years")
    if summary["spy_outperformance_years"] < rules.minimum_spy_outperformance_years:
        failures.append("insufficient_spy_outperformance_years")
    if (
        summary["combined_sharpe"]
        < control["combined_sharpe"] + rules.minimum_sharpe_improvement
    ):
        failures.append("insufficient_sharpe_improvement")
    if (
        summary["combined_max_drawdown"]
        < control["combined_max_drawdown"] - rules.maximum_drawdown_deterioration
    ):
        failures.append("drawdown_deterioration")
    required_cost = robustness.loc[
        (robustness["test"] == "cost_sensitivity")
        & (
            pd.to_numeric(robustness["setting"], errors="coerce")
            == rules.required_cost_bps
        )
    ]
    if (
        required_cost.empty
        or required_cost.iloc[0]["status"] != "valid"
        or float(required_cost.iloc[0]["net_total_return"]) <= 0
    ):
        failures.append("required_cost_scenario_failed")
    stability = robustness.loc[
        robustness["test"].isin(["top_n_sensitivity", "rebalance_day_shift"])
    ]
    if (
        stability.empty
        or not stability["status"].eq("valid").all()
        or pd.to_numeric(stability["sharpe_decline"], errors="coerce")
        .gt(rules.robustness_maximum_sharpe_decline)
        .any()
    ):
        failures.append("robustness_decline_or_invalid")
    return not failures, failures


def _ranking_key(summary: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(summary["median_annual_sharpe"]),
        float(summary["combined_max_drawdown"]),
        -float(summary["combined_average_turnover"]),
    )


def _render_comparison(
    summaries: pd.DataFrame,
    decision: dict[str, Any],
) -> str:
    columns = [
        "candidate_id",
        "valid_folds",
        "positive_years",
        "spy_outperformance_years",
        "median_annual_sharpe",
        "combined_net_total_return",
        "combined_sharpe",
        "combined_max_drawdown",
        "combined_average_turnover",
        "selection_passed",
    ]
    table_frame = summaries.reindex(columns=columns).fillna("")
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in table_frame.itertuples(index=False, name=None)
    ]
    table = "\n".join([header, divider, *rows])
    return "\n".join(
        [
            "# v2 Historical Replay",
            "",
            f"- Evaluation scope: `{REPLAY_SCOPE}`",
            f"- Decision: `{decision['decision']}`",
            f"- Selected strategy: `{decision.get('selected_strategy_id') or 'none'}`",
            f"- Challenger: `{decision.get('challenger_strategy_id') or 'none'}`",
            "",
            table,
            "",
            "These results are exploratory historical replay evidence. They do not "
            "replace the frozen confirmatory v1 decision.",
            "",
        ]
    )


def run_historical_replay(
    base_config: AppConfig,
    suite_path: str | Path,
) -> HistoricalReplayResult:
    suite = load_replay_suite(suite_path)
    definitions = [
        load_candidate_definition(base_config, path)
        for path in suite.candidates
    ]
    identifiers = [definition.candidate_id for definition in definitions]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Candidate IDs must be unique.")
    if CONTROL_ID not in identifiers:
        raise ValueError(f"Replay suite must contain {CONTROL_ID}.")

    full_start = min(fold.development_start for fold in suite.folds)
    full_end = max(fold.test_end for fold in suite.folds)
    evaluation_start = min(fold.test_start for fold in suite.folds)
    context = create_run_context(
        base_config,
        "historical-replay",
        full_end,
    )
    snapshot = build_data_snapshot(base_config)
    write_data_snapshot(snapshot, context.path("data_snapshot.json"))
    context.path("suite.json").write_text(
        suite.model_dump_json(indent=2),
        encoding="utf-8",
    )
    cache = load_backtest_data(base_config)
    replays: dict[str, CandidateReplay] = {}
    fold_rows: list[dict[str, Any]] = []
    diagnostic_frames: list[pd.DataFrame] = []
    timing_rows: list[dict[str, Any]] = []

    try:
        for definition in definitions:
            candidate_started = time.perf_counter()
            candidate_root = context.path("candidates", definition.candidate_id)
            candidate_root.mkdir(parents=True, exist_ok=True)
            candidate_root.joinpath("resolved_config.json").write_text(
                definition.config.model_dump_json(indent=2),
                encoding="utf-8",
            )
            candidate_root.joinpath("identity.json").write_text(
                json.dumps(
                    {
                        "candidate_id": definition.candidate_id,
                        "description": definition.description,
                        "candidate_file": str(definition.path),
                        "candidate_file_sha256": definition.file_sha256,
                        "config_hash": config_hash(definition.config),
                        "data_snapshot_id": snapshot["snapshot_id"],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            try:
                source_started = time.perf_counter()
                source = run_backtest(
                    full_start,
                    full_end,
                    definition.config,
                    publish_latest=False,
                    enforce_data_gate=definition.candidate_id == CONTROL_ID,
                    data_cache=cache,
                    artifact_level="metrics",
                    exact_target_count=False,
                    research_metadata={
                        "evaluation_scope": REPLAY_SCOPE,
                        "data_snapshot_id": snapshot["snapshot_id"],
                        "data_snapshot": snapshot,
                    },
                )
                timing_rows.append(
                    {
                        "candidate_id": definition.candidate_id,
                        "stage": "source_backtest",
                        "elapsed_seconds": time.perf_counter() - source_started,
                        "status": "valid",
                    }
                )
                factors, scores = _extract_cached_frames(cache, definition.config)
                write_parquet(factors, candidate_root / "feature_snapshot.parquet")
                write_parquet(scores, candidate_root / "score_snapshot.parquet")
                diagnostics_started = time.perf_counter()
                diagnostics = _development_diagnostics(
                    definition,
                    suite.folds,
                    factors,
                    cache.prices,
                )
                if not diagnostics.empty:
                    diagnostic_frames.append(diagnostics)
                    write_csv(
                        diagnostics,
                        candidate_root / "development_factor_diagnostics.csv",
                    )
                timing_rows.append(
                    {
                        "candidate_id": definition.candidate_id,
                        "stage": "development_diagnostics",
                        "elapsed_seconds": time.perf_counter() - diagnostics_started,
                        "status": "valid",
                    }
                )
                candidate_fold_rows: list[dict[str, Any]] = []
                candidate_daily: list[pd.DataFrame] = []
                candidate_rebalances: list[pd.DataFrame] = []

                def evaluate_fold(fold: ReplayFold):
                    try:
                        return _evaluate_fold(
                            context.root,
                            definition,
                            fold,
                            scores,
                            cache.prices,
                            cache.market_data,
                        )
                    except (ValueError, RuntimeError) as exc:
                        return (
                            {
                                "candidate_id": definition.candidate_id,
                                "fold": fold.fold_id,
                                "development_start": fold.development_start,
                                "development_end": fold.development_end,
                                "test_start": fold.test_start,
                                "test_end": fold.test_end,
                                "status": "invalid",
                                "error": str(exc),
                            },
                            pd.DataFrame(),
                            pd.DataFrame(),
                        )

                folds_started = time.perf_counter()
                parallel_workers = max(
                    1,
                    min(
                        int(definition.config.robustness.parallel_workers),
                        len(suite.folds),
                    ),
                )
                with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                    fold_results = list(executor.map(evaluate_fold, suite.folds))
                timing_rows.append(
                    {
                        "candidate_id": definition.candidate_id,
                        "stage": "annual_folds",
                        "elapsed_seconds": time.perf_counter() - folds_started,
                        "status": "valid",
                        "parallel_workers": parallel_workers,
                    }
                )
                for row, daily, rebalances in fold_results:
                    candidate_fold_rows.append(row)
                    fold_rows.append(row)
                    if not daily.empty:
                        candidate_daily.append(daily)
                    if not rebalances.empty:
                        candidate_rebalances.append(rebalances)
                fold_frame = pd.DataFrame(candidate_fold_rows)
                if not candidate_daily:
                    raise ValueError("Candidate produced no valid annual test folds.")
                combined_daily = pd.concat(candidate_daily, ignore_index=True)
                combined_daily["date"] = pd.to_datetime(combined_daily["date"])
                if combined_daily["date"].duplicated().any():
                    raise ValueError("Candidate test folds contain duplicate dates.")
                combined_rebalances = (
                    pd.concat(candidate_rebalances, ignore_index=True)
                    if candidate_rebalances
                    else pd.DataFrame()
                )
                summary = _candidate_summary(
                    definition,
                    fold_frame,
                    combined_daily,
                    combined_rebalances,
                    len(suite.folds),
                )
                summary["source_run_id"] = source.run_id
                summary["elapsed_seconds"] = time.perf_counter() - candidate_started
                replays[definition.candidate_id] = CandidateReplay(
                    definition=definition,
                    source_run_id=source.run_id,
                    source_run_path=source.run_path,
                    source_daily=source.daily_returns,
                    source_rebalances=source.rebalances,
                    scores=scores,
                    factors=factors,
                    fold_metrics=fold_frame,
                    combined_daily=combined_daily,
                    combined_rebalances=combined_rebalances,
                    summary=summary,
                )
                write_csv(fold_frame, candidate_root / "fold_metrics.csv")
                write_csv(
                    combined_daily,
                    candidate_root / "combined_test_daily_returns.csv",
                )
            except (ValueError, RuntimeError) as exc:
                timing_rows.append(
                    {
                        "candidate_id": definition.candidate_id,
                        "stage": "candidate_total",
                        "elapsed_seconds": time.perf_counter() - candidate_started,
                        "status": "invalid",
                        "error": str(exc),
                    }
                )
                fold_rows.extend(
                    {
                        "candidate_id": definition.candidate_id,
                        "fold": fold.fold_id,
                        "development_start": fold.development_start,
                        "development_end": fold.development_end,
                        "test_start": fold.test_start,
                        "test_end": fold.test_end,
                        "status": "invalid",
                        "error": str(exc),
                    }
                    for fold in suite.folds
                )

        if CONTROL_ID not in replays:
            raise ValueError("The v1 control replay is invalid; selection cannot proceed.")

        challenger_pool = [
            replay
            for candidate_id, replay in replays.items()
            if candidate_id != CONTROL_ID
        ]
        ranked = sorted(
            challenger_pool,
            key=lambda replay: _ranking_key(replay.summary),
            reverse=True,
        )
        robustness_frames: list[pd.DataFrame] = []
        for replay in ranked[:2]:
            robustness_started = time.perf_counter()
            robustness_frame = _robustness_matrix(
                    context.root,
                    replay,
                    cache,
                    suite.selection,
                    full_start,
                    full_end,
                    evaluation_start,
                )
            robustness_frames.append(robustness_frame)
            timing_rows.append(
                {
                    "candidate_id": replay.definition.candidate_id,
                    "stage": "robustness_matrix",
                    "elapsed_seconds": time.perf_counter() - robustness_started,
                    "status": "valid",
                }
            )
        robustness = (
            pd.concat(robustness_frames, ignore_index=True)
            if robustness_frames
            else pd.DataFrame()
        )
        write_csv(robustness, context.path("robustness_matrix.csv"))

        control = replays[CONTROL_ID].summary
        passed: list[CandidateReplay] = []
        for replay in challenger_pool:
            candidate_robustness = robustness.loc[
                robustness["candidate_id"] == replay.definition.candidate_id
            ].copy()
            selection_passed, failures = _selection_status(
                replay.summary,
                control,
                candidate_robustness,
                suite.selection,
            )
            replay.summary["selection_passed"] = selection_passed
            replay.summary["selection_failures"] = json.dumps(failures)
            if selection_passed:
                passed.append(replay)
        control["selection_passed"] = False
        control["selection_failures"] = json.dumps(["control_only"])

        if passed:
            selected = max(passed, key=lambda replay: _ranking_key(replay.summary))
            selected_strategy_id = selected.definition.candidate_id
            challenger_strategy_id = None
            decision_name = "selected_v2"
        else:
            valid_ranked = [
                replay
                for replay in ranked
                if replay.summary.get("all_folds_valid")
            ]
            challenger = valid_ranked[0] if valid_ranked else None
            selected_strategy_id = None
            challenger_strategy_id = (
                challenger.definition.candidate_id if challenger else None
            )
            decision_name = (
                "challenger_only" if challenger_strategy_id else "no_valid_challenger"
            )

        summaries = pd.DataFrame(
            [replay.summary for replay in replays.values()]
        ).sort_values("candidate_id")
        decision = {
            "study_id": suite.study_id,
            "evaluation_scope": REPLAY_SCOPE,
            "decision": decision_name,
            "selected_strategy_id": selected_strategy_id,
            "challenger_strategy_id": challenger_strategy_id,
            "control_strategy_id": CONTROL_ID,
            "forward_strategy_id": selected_strategy_id or challenger_strategy_id,
            "data_snapshot_id": snapshot["snapshot_id"],
        }
        write_csv(pd.DataFrame(fold_rows), context.path("fold_metrics.csv"))
        write_csv(summaries, context.path("candidate_summary.csv"))
        write_csv(pd.DataFrame(timing_rows), context.path("timings.csv"))
        if diagnostic_frames:
            write_csv(
                pd.concat(diagnostic_frames, ignore_index=True),
                context.path("development_factor_diagnostics.csv"),
            )
        context.path("selection.json").write_text(
            json.dumps(decision, indent=2),
            encoding="utf-8",
        )
        context.path("comparison.md").write_text(
            _render_comparison(summaries, decision),
            encoding="utf-8",
        )
        context.update(
            status="valid",
            data_snapshot_id=snapshot["snapshot_id"],
            evidence_status=REPLAY_SCOPE,
            quality_gates={
                "candidate_identity_complete": True,
                "test_folds_non_overlapping": True,
                "confirmatory_v1_untouched": True,
                "robustness_candidates": len(robustness_frames),
                "fold_parallel_workers": max(
                    int(definition.config.robustness.parallel_workers)
                    for definition in definitions
                ),
            },
            notes=[
                "Historical replay evidence is exploratory and does not replace "
                "the frozen confirmatory v1 decision."
            ],
        )
        return HistoricalReplayResult(
            run_id=context.manifest.run_id,
            run_path=context.root,
            selected_strategy_id=selected_strategy_id,
            challenger_strategy_id=challenger_strategy_id,
            decision=decision_name,
        )
    except Exception as exc:
        if timing_rows:
            write_csv(pd.DataFrame(timing_rows), context.path("timings.csv"))
        context.update(status="invalid", notes=[str(exc)])
        raise
