from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yaml

from qss.backtest.engine import _prepare_ledger_market_data
from qss.config.schema import AppConfig, MLConfig
from qss.data.storage import resolve_path, write_csv, write_parquet
from qss.ingestion.fama_french import load_fama_french_daily
from qss.model.evaluation import evaluate_walk_forward
from qss.research.portfolio_evaluation import simulate_score_portfolio
from qss.research.statistics import (
    block_bootstrap_summary,
    deflated_sharpe_probability,
    fama_french_style_regression,
)

DEFAULT_SOURCE_RUN_ID = "20260613T115415Z-backtest-01e41c77"
DEFAULT_OUTPUT_DIR = Path("reports/research/ml_strategy_development")

BEST_VALUE_LOW_RISK_FACTORS = [
    "book_to_market",
    "earnings_yield",
    "free_cash_flow_yield",
    "sales_yield",
    "beta_to_spy",
    "max_drawdown_252d",
    "realized_vol_252d",
    "realized_vol_60d",
]

STYLE_EXPOSURES = [
    "book_to_market",
    "earnings_yield",
    "free_cash_flow_yield",
    "sales_yield",
    "beta_to_spy",
    "realized_vol_252d",
    "realized_vol_60d",
]

BEST_LIGHTGBM_PARAMETERS = {
    "n_estimators": 80,
    "learning_rate": 0.03,
    "num_leaves": 7,
    "min_child_samples": 30,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.1,
    "reg_lambda": 1.5,
    "random_state": 42,
    "verbosity": -1,
}


@dataclass(frozen=True)
class LabelVariant:
    name: str
    description: str
    builder: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]


def factor_panel(
    factor_values: pd.DataFrame,
    *,
    factors: list[str] | None = None,
) -> pd.DataFrame:
    """Return one row per date-symbol with factor columns and metadata."""
    if factor_values.empty:
        return pd.DataFrame()
    frame = factor_values.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    if factors is not None:
        frame = frame.loc[frame["factor_name"].isin(factors)].copy()
    values = frame.pivot_table(
        index=["date", "symbol"],
        columns="factor_name",
        values="processed_value",
        aggfunc="last",
    ).reset_index()
    metadata_columns = [
        column
        for column in ["date", "symbol", "sector", "market_cap"]
        if column in frame.columns
    ]
    metadata = (
        frame[metadata_columns].drop_duplicates(["date", "symbol"])
        if metadata_columns
        else frame[["date", "symbol"]].drop_duplicates()
    )
    return values.merge(metadata, on=["date", "symbol"], how="left")


def _numeric_design_columns(
    cross_section: pd.DataFrame,
    exposure_columns: list[str],
) -> pd.DataFrame:
    columns: dict[str, pd.Series] = {}
    for column in exposure_columns:
        if column not in cross_section:
            continue
        values = pd.to_numeric(cross_section[column], errors="coerce")
        if values.notna().sum() < 3:
            continue
        filled = values.fillna(values.median())
        if float(filled.std(ddof=0)) <= 1e-12:
            continue
        columns[column] = filled.astype(float)
    return pd.DataFrame(columns, index=cross_section.index)


def residualize_by_date(
    frame: pd.DataFrame,
    *,
    value_column: str,
    exposure_columns: list[str],
    include_sector: bool = True,
    residual_column: str = "residual_value",
) -> pd.DataFrame:
    """Cross-sectionally residualize values using same-date exposures only."""
    if frame.empty:
        return frame.copy()
    required = {"date", value_column}
    if not required.issubset(frame.columns):
        raise ValueError(f"Residualization requires columns: {sorted(required)}")
    rows: list[pd.DataFrame] = []
    working = frame.copy()
    working["date"] = pd.to_datetime(working["date"]).dt.normalize()
    for _, cross in working.groupby("date", sort=True):
        cross = cross.copy()
        y = pd.to_numeric(cross[value_column], errors="coerce").astype(float)
        valid = y.notna()
        residual = pd.Series(np.nan, index=cross.index, dtype=float)
        if int(valid.sum()) < 3 or float(y.loc[valid].std(ddof=0)) <= 1e-12:
            cross[residual_column] = residual
            rows.append(cross)
            continue
        design = _numeric_design_columns(cross.loc[valid], exposure_columns)
        if include_sector and "sector" in cross:
            sectors = cross.loc[valid, "sector"].fillna("Unknown").astype(str)
            dummies = pd.get_dummies(sectors, prefix="sector", drop_first=True, dtype=float)
            design = pd.concat([design, dummies], axis=1)
        x = design.to_numpy(dtype=float) if not design.empty else np.empty((int(valid.sum()), 0))
        x = np.column_stack([np.ones(int(valid.sum())), x])
        coefficients, *_ = np.linalg.lstsq(x, y.loc[valid].to_numpy(dtype=float), rcond=None)
        residual.loc[valid] = y.loc[valid].to_numpy(dtype=float) - x @ coefficients
        cross[residual_column] = residual
        rows.append(cross)
    return pd.concat(rows, ignore_index=True)


def _rank_within_date(values: pd.Series) -> pd.Series:
    return values.rank(method="average", pct=True)


def style_residual_rank_labels(
    factor_values: pd.DataFrame,
    forward_labels: pd.DataFrame,
) -> pd.DataFrame:
    panel = factor_panel(factor_values, factors=BEST_VALUE_LOW_RISK_FACTORS)
    labels = forward_labels.loc[forward_labels["label_name"] == "forward_return"].copy()
    labels["date"] = pd.to_datetime(labels["date"]).dt.normalize()
    merged = labels.merge(panel, on=["date", "symbol"], how="inner")
    residualized = residualize_by_date(
        merged,
        value_column="label_value",
        exposure_columns=STYLE_EXPOSURES,
        include_sector=True,
        residual_column="style_residual_return",
    )
    residualized["label_value"] = residualized.groupby("date", group_keys=False)[
        "style_residual_return"
    ].apply(_rank_within_date)
    residualized["label_name"] = "cross_sectional_rank"
    residualized["version"] = "exploratory_style_residual_rank_v1"
    return residualized.loc[:, forward_labels.columns].dropna(subset=["label_value"])


def sector_relative_rank_labels(
    factor_values: pd.DataFrame,
    forward_labels: pd.DataFrame,
) -> pd.DataFrame:
    panel = factor_panel(factor_values, factors=BEST_VALUE_LOW_RISK_FACTORS)
    labels = forward_labels.loc[forward_labels["label_name"] == "forward_return"].copy()
    labels["date"] = pd.to_datetime(labels["date"]).dt.normalize()
    merged = labels.merge(panel[["date", "symbol", "sector"]], on=["date", "symbol"], how="inner")
    merged["sector"] = merged["sector"].fillna("Unknown").astype(str)
    sector_mean = merged.groupby(["date", "sector"], observed=True)["label_value"].transform("mean")
    merged["sector_relative_return"] = merged["label_value"] - sector_mean
    merged["label_value"] = merged.groupby("date", group_keys=False)[
        "sector_relative_return"
    ].apply(_rank_within_date)
    merged["label_name"] = "cross_sectional_rank"
    merged["version"] = "exploratory_sector_relative_rank_v1"
    return merged.loc[:, forward_labels.columns].dropna(subset=["label_value"])


def style_neutralized_score_frame(
    predictions: pd.DataFrame,
    factor_values: pd.DataFrame,
    *,
    score_column: str = "prediction",
    factors: list[str] | None = None,
    exposure_columns: list[str] | None = None,
    include_sector: bool = True,
) -> pd.DataFrame:
    panel = factor_panel(factor_values, factors=factors or BEST_VALUE_LOW_RISK_FACTORS)
    exposures = exposure_columns or STYLE_EXPOSURES
    merged = predictions.copy()
    merged["date"] = pd.to_datetime(merged["date"]).dt.normalize()
    merged = merged.drop(
        columns=[column for column in ["sector", "market_cap"] if column in merged.columns]
    )
    merged = merged.merge(panel, on=["date", "symbol"], how="inner")
    residualized = residualize_by_date(
        merged,
        value_column=score_column,
        exposure_columns=exposures,
        include_sector=include_sector,
        residual_column="style_neutral_score",
    )
    residualized["total_score"] = residualized["style_neutral_score"]
    return residualized[
        ["date", "symbol", "total_score", "sector", "market_cap", "style_neutral_score"]
    ].dropna(subset=["total_score"])


def score_frame_from_predictions(
    predictions: pd.DataFrame,
    factor_values: pd.DataFrame,
    *,
    score_column: str = "prediction",
) -> pd.DataFrame:
    panel = factor_panel(factor_values, factors=BEST_VALUE_LOW_RISK_FACTORS)
    merged = predictions.copy()
    merged["date"] = pd.to_datetime(merged["date"]).dt.normalize()
    merged = merged.merge(panel[["date", "symbol", "sector", "market_cap"]], on=["date", "symbol"], how="inner")
    merged["total_score"] = pd.to_numeric(merged[score_column], errors="coerce")
    return merged[["date", "symbol", "total_score", "sector", "market_cap"]].dropna(
        subset=["total_score"]
    )


def load_source_config(run_root: Path) -> AppConfig:
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    return AppConfig.model_validate(manifest["config"])


def configured_ml() -> MLConfig:
    return MLConfig(
        enabled=True,
        model_type="lightgbm",
        target="cross_sectional_rank",
        parameters=BEST_LIGHTGBM_PARAMETERS,
        portfolio_top_n=25,
        transaction_cost_bps=10.0,
        walk_forward={
            "train_periods": 60,
            "min_train_periods": 12,
            "test_periods": 3,
            "step_periods": 3,
            "rolling": True,
            "purge": True,
            "embargo_days": 5,
        },
    )


def configured_portfolio_config(
    source_config: AppConfig,
    *,
    max_sector_weight: float = 0.25,
    target_num_holdings: int = 25,
    max_weight: float = 0.05,
) -> AppConfig:
    config = deepcopy(source_config)
    config.ml = configured_ml()
    config.optimizer.constraints.target_num_holdings = target_num_holdings
    config.optimizer.constraints.max_weight = max_weight
    config.optimizer.constraints.max_sector_weight = max_sector_weight
    config.optimizer.constraints.max_turnover_per_rebalance = 0.30
    return config


def _metric_map(metrics: pd.DataFrame) -> dict[str, float]:
    if metrics.empty or not {"metric", "value"}.issubset(metrics.columns):
        return {}
    return {
        str(row.metric): float(row.value)
        for row in metrics.itertuples(index=False)
        if pd.notna(row.value)
    }


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _markdown_table(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    floatfmt: str = ".4f",
) -> str:
    if frame.empty:
        return ""
    selected = frame.loc[:, columns].copy()
    rows = []
    for row in selected.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, float | np.floating):
                values.append("" if np.isnan(value) else format(float(value), floatfmt))
            else:
                values.append(str(value))
        rows.append(values)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def _daily_diagnostics(
    daily_returns: pd.DataFrame,
    config: AppConfig,
    *,
    trial_count: int,
    output_dir: Path,
) -> dict[str, float | bool]:
    bootstrap = block_bootstrap_summary(
        daily_returns,
        primary_metric="sharpe_ratio",
        block_size=config.research_validation.bootstrap_block_days,
        samples=config.research_validation.bootstrap_samples,
        seed=config.research_validation.bootstrap_seed,
        confidence_level=config.research_validation.confidence_level,
    )
    write_csv(bootstrap, output_dir / "bootstrap_summary.csv")
    dsp = deflated_sharpe_probability(daily_returns["portfolio_return"], trial_count)
    _write_json(output_dir / "deflated_sharpe.json", dsp)
    style_factors = load_fama_french_daily(config.research_validation.style_factor_cache)
    exposures, style_summary = fama_french_style_regression(daily_returns, style_factors)
    write_csv(exposures, output_dir / "style_factor_exposures.csv")
    _write_json(output_dir / "style_factor_summary.json", style_summary)
    lower = np.nan
    if not bootstrap.empty:
        sharpe_row = bootstrap.loc[bootstrap["metric"] == "sharpe_ratio"]
        if not sharpe_row.empty:
            lower = float(sharpe_row.iloc[0]["one_sided_lower_95"])
    return {
        "bootstrap_sharpe_one_sided_lower_95": lower,
        "deflated_sharpe_probability": float(dsp.get("probability", np.nan)),
        "deflated_sharpe_trial_count": float(dsp.get("trial_count", trial_count)),
        "ff_alpha_annualized": float(style_summary.get("alpha_annualized", np.nan)),
        "ff_alpha_t_stat": float(style_summary.get("alpha_t_stat", np.nan)),
        "ff_r_squared": float(style_summary.get("r_squared", np.nan)),
        "passes_daily_dsp_gate": bool(
            float(dsp.get("probability", np.nan))
            >= config.research_validation.deflated_sharpe_probability
        ),
    }


def run_daily_simulation(
    score_frame: pd.DataFrame,
    prices: pd.DataFrame,
    source_config: AppConfig,
    *,
    name: str,
    output_dir: Path,
    end_date: str,
    max_sector_weight: float,
    target_num_holdings: int = 25,
    max_weight: float = 0.05,
    trial_count: int,
    market_data=None,
) -> dict[str, float | str | bool]:
    sim_config = configured_portfolio_config(
        source_config,
        max_sector_weight=max_sector_weight,
        target_num_holdings=target_num_holdings,
        max_weight=max_weight,
    )
    start = str(pd.to_datetime(score_frame["date"]).min().date())
    simulation_dir = output_dir / "simulations" / name
    evaluation = simulate_score_portfolio(
        score_frame,
        prices,
        sim_config,
        start_date=start,
        end_date=end_date,
        output_path=simulation_dir,
        market_data=market_data,
    )
    metrics = _metric_map(evaluation.metrics)
    diagnostics = _daily_diagnostics(
        evaluation.daily_returns,
        sim_config,
        trial_count=trial_count,
        output_dir=simulation_dir,
    )
    row: dict[str, float | str | bool] = {
        "candidate": name,
        "kind": "daily_simulation",
        "start_date": start,
        "end_date": end_date,
        "max_sector_weight": max_sector_weight,
        "target_num_holdings": target_num_holdings,
        "max_weight": max_weight,
        "total_return": metrics.get("total_return", np.nan),
        "net_total_return": metrics.get("net_total_return", np.nan),
        "sharpe_ratio": metrics.get("sharpe_ratio", np.nan),
        "max_drawdown": metrics.get("max_drawdown", np.nan),
        "beta": metrics.get("beta", np.nan),
        "alpha_annualized": metrics.get("alpha_annualized", np.nan),
        "information_ratio": metrics.get("information_ratio", np.nan),
        "average_turnover": metrics.get("average_turnover", np.nan),
        "average_number_of_holdings": metrics.get("average_number_of_holdings", np.nan),
    }
    row.update(diagnostics)
    _write_json(simulation_dir / "simulation_summary.json", row)
    return row


def _evaluation_summary(name: str, result: dict[str, pd.DataFrame]) -> dict[str, float | str]:
    aggregate = result["aggregate_metrics"].iloc[0].to_dict()
    portfolio = result["portfolio_metrics"].iloc[0].to_dict()
    rank_ic = result["fold_metrics"]["rank_ic"].dropna()
    return {
        "candidate": name,
        "kind": "walk_forward_label_experiment",
        "folds": float(aggregate.get("folds", np.nan)),
        "mean_rank_ic": float(aggregate.get("mean_rank_ic", np.nan)),
        "positive_rank_ic_share": float((rank_ic > 0).mean()) if not rank_ic.empty else np.nan,
        "periods": float(portfolio.get("periods", np.nan)),
        "net_total_return": float(portfolio.get("net_total_return", np.nan)),
        "net_sharpe": float(portfolio.get("net_sharpe", np.nan)),
        "average_turnover": float(portfolio.get("average_turnover", np.nan)),
    }


def _load_best_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    return frame


def _report_markdown(
    rows: list[dict],
    *,
    source_run_id: str,
    output_dir: Path,
    audit: dict,
) -> str:
    results = pd.DataFrame(rows)
    daily = results.loc[results["kind"] == "daily_simulation"].copy()
    walk = results.loc[results["kind"] == "walk_forward_label_experiment"].copy()
    lines = [
        "# ML Strategy Development",
        "",
        f"- Source run: `{source_run_id}`",
        "- Research stage: `exploratory`",
        "- Validation conditions changed: `false`",
        "- Official confirmatory claim: `false`",
        f"- New exploratory candidates counted: `{audit['new_exploratory_candidates']}`",
        f"- Cumulative exploratory trial count used for DSP: `{audit['cumulative_trial_count']}`",
        "",
        "## Walk-Forward Label Experiments",
        "",
    ]
    if walk.empty:
        lines.append("_No walk-forward label experiments were produced._")
    else:
        lines.append(
            _markdown_table(
                walk,
                [
                    "candidate",
                    "mean_rank_ic",
                    "positive_rank_ic_share",
                    "net_sharpe",
                    "net_total_return",
                    "average_turnover",
                ],
            )
        )
    lines.extend(["", "## Daily Portfolio Simulations", ""])
    if daily.empty:
        lines.append("_No daily simulations were produced._")
    else:
        ordered = daily.sort_values("sharpe_ratio", ascending=False)
        lines.append(
            _markdown_table(
                ordered,
                [
                    "candidate",
                    "sharpe_ratio",
                    "net_total_return",
                    "max_drawdown",
                    "beta",
                    "target_num_holdings",
                    "max_sector_weight",
                    "ff_alpha_annualized",
                    "ff_alpha_t_stat",
                    "deflated_sharpe_probability",
                    "passes_daily_dsp_gate",
                ],
            )
        )
    lines.extend(
        [
            "",
            "## Audit",
            "",
            f"- Uses closed v1 holdout as confirmation: `{audit['uses_closed_v1_holdout_as_confirmation']}`",
            f"- Official confirmatory claim: `{audit['official_confirmatory_claim']}`",
            f"- Purge enabled: `{audit['purge_enabled']}`",
            f"- Embargo days: `{audit['embargo_days']}`",
            f"- Label leakage audit: `{audit['label_leakage_audit']}`",
            f"- Output directory: `{output_dir.as_posix()}`",
            "",
            "## Research Readout",
            "",
            "- These experiments are only development evidence.",
            "- Passing the daily DSP gate here would still not be a confirmation because the strategy was selected after exploratory search.",
            "- A strategy can move to v2 only after clean-git reproducibility and a fresh preregistered forward holdout.",
        ]
    )
    return "\n".join(lines) + "\n"


def _label_gap_audit(
    labels: pd.DataFrame,
    ml_config: MLConfig,
    *,
    factor_values: pd.DataFrame,
) -> dict[str, bool | str]:
    from qss.model.evaluation import build_model_dataset
    from qss.research.walk_forward import walk_forward_splits

    dataset, _ = build_model_dataset(factor_values, labels, ml_config.target)
    folds = walk_forward_splits(dataset, ml_config.walk_forward)
    passed = True
    for fold, train_index, _ in folds:
        train = dataset.loc[train_index]
        cutoff = fold.test_start - pd.Timedelta(days=ml_config.walk_forward.embargo_days)
        if not (pd.to_datetime(train["label_end_time"]) < cutoff).all():
            passed = False
            break
    return {
        "passed": bool(passed),
        "rule": "max train label_end_time < test_start - embargo_days for every fold",
    }


def run_ml_strategy_development(
    *,
    source_run_id: str = DEFAULT_SOURCE_RUN_ID,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path | pd.DataFrame | dict]:
    source_root = resolve_path(Path("reports/runs") / source_run_id)
    output_root = resolve_path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    source_config = load_source_config(source_root)
    run_manifest = json.loads((source_root / "manifest.json").read_text(encoding="utf-8"))
    existing_audit_path = resolve_path("reports/research/ml_optimization_search/audit.json")
    existing_trials = 59
    if existing_audit_path.exists():
        existing_trials = int(json.loads(existing_audit_path.read_text(encoding="utf-8")).get("search_candidates", 59))

    factors = pd.read_parquet(source_root / "feature_snapshot.parquet")
    factors["date"] = pd.to_datetime(factors["date"]).dt.normalize()
    model_factors = factors.loc[factors["factor_name"].isin(BEST_VALUE_LOW_RISK_FACTORS)].copy()
    forward_labels = pd.read_parquet(source_root / "labels_forward_return.parquet")
    prices = pd.read_parquet(resolve_path(Path(source_config.paths.silver_data) / "prices" / "prices_daily.parquet"))
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()

    ml_config = configured_ml()
    variants = [
        LabelVariant(
            name="style_residual_rank_lgbm",
            description="Predict same-date sector and value/low-risk residual forward-return ranks.",
            builder=style_residual_rank_labels,
        ),
        LabelVariant(
            name="sector_relative_rank_lgbm",
            description="Predict sector-demeaned forward-return ranks.",
            builder=sector_relative_rank_labels,
        ),
    ]
    rows: list[dict] = []
    label_gap_results: dict[str, dict] = {}
    market_data = _prepare_ledger_market_data(prices)
    data_cutoff = str(run_manifest.get("data_cutoff", source_config.backtest.end_date or "2026-06-11"))
    new_candidate_count = 0

    for variant in variants:
        labels = variant.builder(model_factors, forward_labels)
        labels_for_evaluation = pd.concat(
            [
                labels,
                forward_labels.loc[forward_labels["label_name"] == "forward_return"].copy(),
            ],
            ignore_index=True,
        )
        variant_dir = output_root / "label_experiments" / variant.name
        write_parquet(labels, variant_dir / "labels.parquet")
        write_csv(labels, variant_dir / "labels.csv")
        write_parquet(labels_for_evaluation, variant_dir / "evaluation_labels.parquet")
        label_gap_results[variant.name] = _label_gap_audit(
            labels_for_evaluation,
            ml_config,
            factor_values=model_factors,
        )
        result = evaluate_walk_forward(
            model_factors,
            labels_for_evaluation,
            ml_config,
            variant_dir / "walk_forward",
        )
        summary = _evaluation_summary(variant.name, result)
        summary["description"] = variant.description
        rows.append(summary)
        predictions = result["predictions"]
        raw_scores = score_frame_from_predictions(predictions, model_factors)
        neutral_scores = style_neutralized_score_frame(predictions, model_factors)
        score_variants = [
            (f"{variant.name}__raw_score", raw_scores, 0.25, 25),
            (f"{variant.name}__style_neutral_score", neutral_scores, 0.25, 25),
        ]
        if variant.name == "style_residual_rank_lgbm":
            score_variants.extend(
                [
                    (f"{variant.name}__style_neutral_score__sector_cap_20", neutral_scores, 0.20, 25),
                    (f"{variant.name}__style_neutral_score__sector_cap_15", neutral_scores, 0.15, 25),
                    (
                        f"{variant.name}__style_neutral_score__holdings35_sector_cap20",
                        neutral_scores,
                        0.20,
                        35,
                    ),
                ]
            )
        for score_name, score_frame, sector_cap, target_holdings in score_variants:
            new_candidate_count += 1
            rows.append(
                run_daily_simulation(
                    score_frame,
                    prices,
                    source_config,
                    name=score_name,
                    output_dir=output_root,
                    end_date=data_cutoff,
                    max_sector_weight=sector_cap,
                    target_num_holdings=target_holdings,
                    trial_count=existing_trials + new_candidate_count,
                    market_data=market_data,
                )
            )

    best_predictions_path = resolve_path("reports/research/ml_optimization_search/best_predictions.csv")
    if best_predictions_path.exists():
        best_predictions = _load_best_predictions(best_predictions_path)
        neutral_best = style_neutralized_score_frame(best_predictions, model_factors)
        for score_name, sector_cap, target_holdings in [
            ("current_best__style_neutral_score", 0.25, 25),
            ("current_best__style_neutral_score__sector_cap_20", 0.20, 25),
            ("current_best__style_neutral_score__sector_cap_15", 0.15, 25),
            ("current_best__style_neutral_score__holdings35_sector_cap20", 0.20, 35),
        ]:
            new_candidate_count += 1
            rows.append(
                run_daily_simulation(
                    neutral_best,
                    prices,
                    source_config,
                    name=score_name,
                    output_dir=output_root,
                    end_date=data_cutoff,
                    max_sector_weight=sector_cap,
                    target_num_holdings=target_holdings,
                    trial_count=existing_trials + new_candidate_count,
                    market_data=market_data,
                )
            )

    results = pd.DataFrame(rows)
    write_csv(results, output_root / "experiment_results.csv")
    audit = {
        "source_run": source_run_id,
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "research_stage": "exploratory",
        "validation_conditions_changed": False,
        "official_confirmatory_claim": False,
        "uses_closed_v1_holdout_as_confirmation": False,
        "existing_ml_search_candidates": existing_trials,
        "new_exploratory_candidates": new_candidate_count,
        "cumulative_trial_count": existing_trials + new_candidate_count,
        "purge_enabled": ml_config.walk_forward.purge,
        "embargo_days": ml_config.walk_forward.embargo_days,
        "label_leakage_audit": label_gap_results,
        "known_limits": [
            "Development search only; not official confirmatory evidence.",
            "Fresh v2 forward holdout starts 2026-07-01 and is unavailable on 2026-06-16.",
            "Score neutralization reduces direct style ranking exposure but does not impose a hard beta-neutral optimizer constraint.",
        ],
    }
    _write_json(output_root / "audit.json", audit)
    report = _report_markdown(rows, source_run_id=source_run_id, output_dir=output_root, audit=audit)
    (output_root / "strategy_development_report.md").write_text(report, encoding="utf-8")
    config_payload = {
        "model_factors": BEST_VALUE_LOW_RISK_FACTORS,
        "style_exposures": STYLE_EXPOSURES,
        "lightgbm_parameters": BEST_LIGHTGBM_PARAMETERS,
        "ml_config": ml_config.model_dump(mode="json"),
    }
    (output_root / "development_config.yaml").write_text(
        yaml.safe_dump(config_payload, sort_keys=False),
        encoding="utf-8",
    )
    return {
        "output_dir": output_root,
        "results": results,
        "audit": audit,
    }


def main() -> None:
    result = run_ml_strategy_development()
    print(result["output_dir"])


if __name__ == "__main__":
    main()
