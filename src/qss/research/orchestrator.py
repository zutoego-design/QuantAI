from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from shutil import copyfile

import pandas as pd
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from qss.backtest.engine import load_backtest_data, run_backtest
from qss.backtest.metrics import comprehensive_factor_diagnostics
from qss.config.schema import AppConfig
from qss.data.storage import resolve_path
from qss.data.validation import validate_research_data
from qss.experiments.registry import ExperimentRegistry, registry_record_from_run
from qss.ingestion.fama_french import load_fama_french_daily
from qss.model.evaluation import fit_holdout_predictions
from qss.model.scoring import compute_alpha_scores
from qss.research.decision import (
    research_evidence_decision,
    write_research_decision,
)
from qss.research.governance import confirmatory_rerun_guard
from qss.research.portfolio_evaluation import simulate_score_portfolio
from qss.research.protocol import (
    ResearchProtocol,
    StudyStatus,
    exploratory_protocol,
    validate_label_gap,
)
from qss.research.snapshot import build_data_snapshot, write_data_snapshot
from qss.research.statistics import (
    block_bootstrap_summary,
    deflated_sharpe_probability,
    fama_french_style_regression,
)
from qss.runs.manifest import create_run_context


class HypothesisFamily(BaseModel):
    family_id: str | None = None
    trial_family: str | None = None
    factors: list[str] = Field(default_factory=list)
    expected_direction: str
    score_orientation: str
    trial_budget: int | None = Field(default=None, ge=1)
    primary_metric: str | None = None
    decision_rule: str | None = None


class ExperimentSpec(BaseModel):
    hypothesis: str
    universe: str = "sp500_historical_standard"
    factors: list[str] = Field(default_factory=list)
    preprocessing: dict = Field(default_factory=dict)
    portfolio: dict = Field(default_factory=dict)
    costs: dict = Field(default_factory=dict)
    model: dict = Field(default_factory=dict)
    robustness_tests: list[str] = Field(
        default_factory=lambda: [
            "subperiod",
            "cost_sensitivity",
            "top_n_sensitivity",
            "rebalance_day_shift",
        ]
    )
    start_date: str
    end_date: str
    seed: int = 42
    max_years: int = 20
    study_id: str | None = None
    research_stage: str = "exploratory"
    study_status: StudyStatus = "active"
    development_start: str | None = None
    development_end: str | None = None
    holdout_start: str | None = None
    holdout_end: str | None = None
    primary_metric: str = "sharpe_ratio"
    primary_metric_threshold: float = 0.0
    null_hypothesis: str = (
        "The strategy has no positive out-of-sample investment value."
    )
    trial_family: str | None = None
    trial_budget: int | None = Field(default=None, ge=1)
    require_clean_git: bool = True
    factor_evidence_mode: str = "factor_level"
    hypothesis_families: dict[str, HypothesisFamily] = Field(default_factory=dict)
    primary_hypothesis_family: str | None = None
    forward_validation: dict = Field(default_factory=dict)
    factor_directions: dict[str, int] = Field(default_factory=dict)

    @field_validator("end_date")
    @classmethod
    def validate_dates(cls, value: str, info):
        start = info.data.get("start_date")
        if start and pd.Timestamp(value) <= pd.Timestamp(start):
            raise ValueError("end_date must be after start_date")
        return value

    @model_validator(mode="after")
    def validate_protocol_fields(self) -> "ExperimentSpec":
        if self.research_stage not in {"exploratory", "confirmatory"}:
            raise ValueError("research_stage must be exploratory or confirmatory")
        if self.research_stage == "confirmatory":
            required = {
                "study_id": self.study_id,
                "development_start": self.development_start,
                "development_end": self.development_end,
                "holdout_start": self.holdout_start,
                "holdout_end": self.holdout_end,
                "trial_family": self.trial_family,
            }
            missing = [name for name, value in required.items() if not value]
            if missing:
                raise ValueError(
                    f"Confirmatory experiments require: {', '.join(missing)}"
                )
            required_robustness = {
                "subperiod",
                "cost_sensitivity",
                "top_n_sensitivity",
                "rebalance_day_shift",
            }
            missing_robustness = sorted(
                required_robustness - set(self.robustness_tests)
            )
            if missing_robustness:
                raise ValueError(
                    "Confirmatory experiments require robustness tests: "
                    f"{', '.join(missing_robustness)}"
                )
            if (
                self.primary_hypothesis_family
                and self.primary_hypothesis_family not in self.hypothesis_families
            ):
                raise ValueError(
                    "primary_hypothesis_family must be defined in hypothesis_families"
                )
            start_period = pd.Timestamp(self.start_date).to_period("M")
            end_period = pd.Timestamp(self.end_date).to_period("M")
            if end_period.ordinal - start_period.ordinal + 1 < 24:
                raise ValueError(
                    "Confirmatory experiments require at least 24 months "
                    "for subperiod robustness."
                )
        return self

    @property
    def spec_hash(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def legacy_spec_hash_without_governance(self) -> str:
        payload = json.dumps(
            self.model_dump(
                mode="json",
                exclude={
                    "study_status",
                    "trial_budget",
                    "require_clean_git",
                    "factor_evidence_mode",
                    "hypothesis_families",
                    "primary_hypothesis_family",
                    "forward_validation",
                },
            ),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def research_protocol(self) -> ResearchProtocol:
        if not self.study_id:
            return exploratory_protocol(
                study_id=f"legacy-{self.spec_hash[:12]}",
                start_date=self.start_date,
                end_date=self.end_date,
                factors=self.factors,
            )
        return ResearchProtocol(
            study_id=self.study_id,
            stage=self.research_stage,
            study_status=self.study_status,
            development_start=self.development_start or self.start_date,
            development_end=self.development_end or self.end_date,
            holdout_start=self.holdout_start,
            holdout_end=self.holdout_end,
            primary_metric=self.primary_metric,
            primary_metric_threshold=self.primary_metric_threshold,
            null_hypothesis=self.null_hypothesis,
            trial_family=self.trial_family or self.study_id,
            trial_budget=self.trial_budget,
            clean_git_required=(
                self.require_clean_git if self.research_stage == "confirmatory" else False
            ),
            factor_evidence_mode=self.factor_evidence_mode,
            hypothesis_families={
                name: family.model_dump(mode="json")
                for name, family in self.hypothesis_families.items()
            },
            factor_directions=self.factor_directions,
        )


def _flatten_mapping(value: dict, prefix: str = "") -> dict[str, object]:
    flattened: dict[str, object] = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, dict):
            flattened.update(_flatten_mapping(item, path))
        else:
            flattened[path] = item
    return flattened


def _config_diff(base: AppConfig, variant: AppConfig) -> dict[str, dict]:
    left = _flatten_mapping(base.model_dump(mode="json"))
    right = _flatten_mapping(variant.model_dump(mode="json"))
    return {
        key: {"before": left.get(key), "after": right.get(key)}
        for key in sorted(set(left) | set(right))
        if left.get(key) != right.get(key)
    }


def _factor_evidence(
    diagnostics: pd.DataFrame,
    protocol: ResearchProtocol,
    fdr_alpha: float,
) -> tuple[pd.DataFrame, list[str]]:
    if diagnostics.empty:
        return pd.DataFrame(), ["Factor diagnostics are missing."]
    configured = diagnostics.loc[
        diagnostics["factor_name"].isin(protocol.factor_directions)
    ].copy()
    if not protocol.factor_directions:
        configured["preregistered_direction"] = pd.NA
        configured["direction_matches"] = pd.NA
        configured["evidence_status"] = "not_preregistered"
        return configured, []
    configured["preregistered_direction"] = configured["factor_name"].map(
        protocol.factor_directions
    )
    # Diagnostics use direction-adjusted processed values, so positive IC is the
    # preregistered direction after transformation.
    configured["direction_matches"] = (
        pd.to_numeric(configured["rank_ic"], errors="coerce") > 0
    )
    configured["evidence_status"] = (
        configured["direction_matches"]
        & (pd.to_numeric(configured["fdr_q_value"], errors="coerce") <= fdr_alpha)
    ).map({True: "supported", False: "unsupported"})
    missing = sorted(
        set(protocol.factor_directions) - set(configured["factor_name"])
    )
    unsupported = configured.loc[
        configured["evidence_status"] != "supported",
        "factor_name",
    ].astype(str).tolist()
    blockers = []
    if missing:
        blockers.append(f"Preregistered factor diagnostics are missing: {missing}.")
    if unsupported:
        blockers.append(
            "Preregistered factor evidence did not survive direction and FDR "
            f"checks: {sorted(unsupported)}."
        )
    return configured, blockers


def _holdout_factor_diagnostics(
    factors: pd.DataFrame,
    prices: pd.DataFrame,
    holdout_start: str,
    holdout_end: str,
) -> tuple[dict[str, pd.DataFrame], dict]:
    start = pd.Timestamp(holdout_start)
    end = pd.Timestamp(holdout_end)
    factor_dates = pd.to_datetime(factors["date"]).dt.normalize()
    holdout_factors = factors.loc[factor_dates.between(start, end)].copy()
    price_dates = pd.to_datetime(prices["date"]).dt.normalize()
    holdout_prices = prices.loc[price_dates <= end].copy()
    diagnostics = comprehensive_factor_diagnostics(
        holdout_factors,
        holdout_prices,
    )
    scope = {
        "evaluation_scope": "holdout",
        "holdout_start": holdout_start,
        "holdout_end": holdout_end,
        "factor_date_min": (
            str(pd.to_datetime(holdout_factors["date"]).min().date())
            if not holdout_factors.empty
            else None
        ),
        "factor_date_max": (
            str(pd.to_datetime(holdout_factors["date"]).max().date())
            if not holdout_factors.empty
            else None
        ),
        "price_date_max": (
            str(pd.to_datetime(holdout_prices["date"]).max().date())
            if not holdout_prices.empty
            else None
        ),
    }
    return diagnostics, scope


class ResearchOrchestrator:
    """Bounded research runner. It never writes raw inputs or baseline artifacts."""

    def __init__(self, config: AppConfig):
        self.config = config

    def _configured_experiment(self, spec: ExperimentSpec) -> AppConfig:
        config = self.config.model_copy(deep=True)
        if spec.universe != config.universe.name:
            raise ValueError(
                f"Experiment universe {spec.universe!r} is not the configured "
                f"universe {config.universe.name!r}."
            )
        if spec.factors:
            requested = set(spec.factors)
            available = {
                name
                for group in config.factor_groups.values()
                for name in group.factors
            }
            unknown = requested - available
            if unknown:
                raise ValueError(f"Unknown factors requested: {sorted(unknown)}")
            for group_name in list(config.factor_groups):
                group = config.factor_groups[group_name]
                group.factors = {
                    name: definition
                    for name, definition in group.factors.items()
                    if name in requested
                }
                if not group.factors:
                    del config.factor_groups[group_name]

        allowed_preprocessing = {
            "winsorize": config.factor_processing.winsorize,
            "neutralization": config.factor_processing.neutralization,
        }
        for section, values in spec.preprocessing.items():
            if section not in allowed_preprocessing or not isinstance(values, dict):
                raise ValueError(f"Unsupported preprocessing override: {section}")
            target = allowed_preprocessing[section]
            for key, value in values.items():
                if not hasattr(target, key):
                    raise ValueError(f"Unsupported preprocessing field: {section}.{key}")
                setattr(target, key, value)

        allowed_portfolio = {
            "target_num_holdings",
            "max_weight",
            "max_sector_weight",
            "max_turnover_per_rebalance",
        }
        for key, value in spec.portfolio.items():
            if key not in allowed_portfolio:
                raise ValueError(f"Unsupported portfolio override: {key}")
            setattr(config.optimizer.constraints, key, value)

        allowed_costs = {
            "commission_bps",
            "slippage_bps",
            "market_impact_coefficient",
            "max_adv_participation",
        }
        for key, value in spec.costs.items():
            if key not in allowed_costs:
                raise ValueError(f"Unsupported cost override: {key}")
            setattr(config.backtest.transaction_cost, key, value)
        if spec.model:
            allowed_model = {
                "enabled",
                "model_type",
                "target",
                "parameters",
                "portfolio_top_n",
                "transaction_cost_bps",
                "walk_forward",
            }
            unknown_model = set(spec.model) - allowed_model
            if unknown_model:
                raise ValueError(f"Unsupported model fields: {sorted(unknown_model)}")
            config.ml.enabled = bool(spec.model.get("enabled", True))
            for key in [
                "model_type",
                "target",
                "parameters",
                "portfolio_top_n",
                "transaction_cost_bps",
            ]:
                if key in spec.model:
                    setattr(config.ml, key, spec.model[key])
            for key, value in spec.model.get("walk_forward", {}).items():
                if not hasattr(config.ml.walk_forward, key):
                    raise ValueError(f"Unsupported walk-forward field: {key}")
                setattr(config.ml.walk_forward, key, value)
        if spec.research_stage == "confirmatory":
            if len(set(config.robustness.cost_sensitivity_bps)) < 3:
                raise ValueError(
                    "Confirmatory cost sensitivity requires at least three "
                    "distinct cost scenarios."
                )
            if len(set(config.robustness.top_n_values)) < 2:
                raise ValueError(
                    "Confirmatory holding-count sensitivity requires at least "
                    "two distinct target counts."
                )
            nonzero_shifts = {
                shift
                for shift in config.robustness.rebalance_day_shifts
                if shift != 0
            }
            if len(nonzero_shifts) < 2:
                raise ValueError(
                    "Confirmatory rebalance-day sensitivity requires at least "
                    "two non-zero shifts."
                )
        return config

    def _confirmatory_evidence(
        self,
        *,
        spec: ExperimentSpec,
        protocol: ResearchProtocol,
        config: AppConfig,
        data_cache,
        full_result,
        context,
        trial_number: int,
        robustness_rows: list[dict],
    ) -> tuple[dict, pd.DataFrame]:
        assert protocol.holdout_start is not None
        assert protocol.holdout_end is not None
        factors = pd.read_parquet(full_result.run_path / "feature_snapshot.parquet")
        label_frames = []
        for name in [
            "labels_forward_return.parquet",
            "labels_cross_sectional_rank.parquet",
        ]:
            path = full_result.run_path / name
            if path.exists():
                label_frames.append(pd.read_parquet(path))
        labels = (
            pd.concat(label_frames, ignore_index=True)
            if label_frames
            else pd.DataFrame()
        )
        rule_scores = compute_alpha_scores(factors, config)
        rule_scores = rule_scores.loc[
            pd.to_datetime(rule_scores["date"]).between(
                pd.Timestamp(protocol.holdout_start),
                pd.Timestamp(protocol.holdout_end),
            )
        ]
        holdout_root = context.path("holdout_evaluation")
        rule_evaluation = simulate_score_portfolio(
            rule_scores,
            data_cache.prices,
            config,
            start_date=protocol.holdout_start,
            end_date=protocol.holdout_end,
            output_path=holdout_root / "rule_score",
        )
        selected_evaluation = rule_evaluation
        selected_model = "rule_score"
        if config.ml.enabled:
            predictions, metadata = fit_holdout_predictions(
                factors,
                labels,
                config.ml,
                development_end=protocol.development_end,
                holdout_start=protocol.holdout_start,
                holdout_end=protocol.holdout_end,
            )
            factor_metadata = (
                factors.groupby(["date", "symbol"], as_index=False)
                .agg({"sector": "first", "market_cap": "first"})
            )
            model_scores = predictions.rename(
                columns={"prediction": "total_score"}
            ).merge(
                factor_metadata,
                on=["date", "symbol"],
                how="left",
            )
            model_root = holdout_root / config.ml.model_type
            selected_evaluation = simulate_score_portfolio(
                model_scores,
                data_cache.prices,
                config,
                start_date=protocol.holdout_start,
                end_date=protocol.holdout_end,
                output_path=model_root,
            )
            (model_root / "model_evaluation.json").write_text(
                json.dumps(metadata, indent=2),
                encoding="utf-8",
            )
            predictions.to_csv(model_root / "predictions.csv", index=False)
            selected_model = config.ml.model_type

        selected_metrics = selected_evaluation.metrics.set_index("metric")[
            "value"
        ].to_dict()
        pd.DataFrame(
            [
                {
                    "model_type": selected_model,
                    "evaluation_scope": "holdout",
                    "net_total_return": selected_metrics.get("net_total_return"),
                    "net_sharpe": selected_metrics.get("sharpe_ratio"),
                    "cagr": selected_metrics.get("cagr"),
                    "max_drawdown": selected_metrics.get("max_drawdown"),
                    "average_turnover": selected_metrics.get("average_turnover"),
                }
            ]
        ).to_csv(holdout_root / "portfolio_metrics.csv", index=False)

        validation = config.research_validation
        bootstrap = block_bootstrap_summary(
            selected_evaluation.daily_returns,
            primary_metric=protocol.primary_metric,
            block_size=validation.bootstrap_block_days,
            samples=validation.bootstrap_samples,
            seed=validation.bootstrap_seed,
            confidence_level=validation.confidence_level,
        )
        bootstrap.to_csv(context.path("bootstrap_summary.csv"), index=False)
        deflated = deflated_sharpe_probability(
            selected_evaluation.daily_returns["portfolio_return"],
            trial_number,
        )
        context.path("deflated_sharpe.json").write_text(
            json.dumps(deflated, indent=2),
            encoding="utf-8",
        )

        blockers: list[str] = []
        try:
            style_factors = load_fama_french_daily(
                validation.style_factor_cache
            )
            exposures, style_summary = fama_french_style_regression(
                selected_evaluation.daily_returns,
                style_factors,
            )
            style_summary["coverage"] = (
                float(style_summary.get("observations", 0.0))
                / max(len(selected_evaluation.daily_returns), 1)
            )
            exposures.to_csv(context.path("style_factor_exposures.csv"), index=False)
            context.path("style_factor_summary.json").write_text(
                json.dumps(style_summary, indent=2),
                encoding="utf-8",
            )
            if exposures.empty:
                blockers.append(
                    "Fama-French 5-factor plus Momentum regression had insufficient overlap."
                )
            elif style_summary["coverage"] < 0.95:
                blockers.append(
                    "Fama-French style-factor coverage is below 95% of holdout days."
                )
        except (OSError, RuntimeError, ValueError) as exc:
            context.path("style_factor_error.txt").write_text(
                str(exc),
                encoding="utf-8",
            )
            if validation.require_style_regression:
                blockers.append(
                    "Fama-French 5-factor plus Momentum regression is unavailable."
                )

        holdout_diagnostics, diagnostic_scope = _holdout_factor_diagnostics(
            factors,
            data_cache.prices,
            protocol.holdout_start,
            protocol.holdout_end,
        )
        diagnostic_artifacts = {
            "summary": "factor_diagnostics.csv",
            "quantiles": "factor_quantiles.csv",
            "decay": "factor_decay.csv",
            "correlation": "factor_correlation.csv",
        }
        for key, name in diagnostic_artifacts.items():
            holdout_diagnostics[key].to_csv(holdout_root / name, index=False)
            holdout_diagnostics[key].to_csv(context.path(name), index=False)
        context.path("factor_diagnostics_scope.json").write_text(
            json.dumps(diagnostic_scope, indent=2),
            encoding="utf-8",
        )
        diagnostics = holdout_diagnostics["summary"]
        factor_evidence, factor_blockers = _factor_evidence(
            diagnostics,
            protocol,
            validation.fdr_alpha,
        )
        factor_evidence.to_csv(context.path("factor_evidence.csv"), index=False)
        blockers.extend(factor_blockers)
        robustness = pd.DataFrame(robustness_rows)
        required_robustness = {
            "subperiod",
            "cost_sensitivity",
            "top_n_sensitivity",
            "rebalance_day_shift",
        }
        observed_robustness = (
            set(robustness["test"]) if not robustness.empty else set()
        )
        missing_robustness = sorted(required_robustness - observed_robustness)
        invalid_robustness = (
            robustness.loc[
                robustness["test"].isin(required_robustness)
                & (robustness["status"] != "valid"),
                ["test", "setting"],
            ].to_dict("records")
            if not robustness.empty
            else []
        )
        if missing_robustness:
            blockers.append(
                f"Required robustness tests are missing: {missing_robustness}."
            )
        if invalid_robustness:
            blockers.append(
                f"Required robustness tests are invalid: {invalid_robustness}."
            )
        decision = research_evidence_decision(
            stage=protocol.stage,
            primary_metric=protocol.primary_metric,
            threshold=protocol.primary_metric_threshold,
            bootstrap_summary=bootstrap,
            deflated_sharpe=deflated,
            net_total_return=float(
                selected_metrics.get("net_total_return", float("nan"))
            ),
            required_probability=validation.deflated_sharpe_probability,
            blockers=blockers,
        )
        decision["selected_model"] = selected_model
        write_research_decision(decision, context.root)
        return decision, selected_evaluation.metrics

    def run(self, spec: ExperimentSpec):
        years = (pd.Timestamp(spec.end_date) - pd.Timestamp(spec.start_date)).days / 365.25
        if years > spec.max_years:
            raise ValueError(f"Experiment spans {years:.1f} years; limit is {spec.max_years}.")
        start_period = pd.Timestamp(spec.start_date).to_period("M")
        end_period = pd.Timestamp(spec.end_date).to_period("M")
        sample_months = end_period.ordinal - start_period.ordinal + 1
        experiment_config = self._configured_experiment(spec)
        protocol = spec.research_protocol()
        registry = (
            ExperimentRegistry.from_config(experiment_config)
            if experiment_config.registry.enabled
            else None
        )
        trial_number = (
            registry.next_trial_number(protocol.trial_family)
            if registry is not None
            else 1
        )
        confirmatory_rerun_guard(
            protocol,
            trial_number=trial_number,
            trial_budget=spec.trial_budget,
            registry_enabled=registry is not None,
            require_clean_git=spec.require_clean_git,
        )
        data_snapshot = build_data_snapshot(experiment_config)
        if registry is not None:
            prior_snapshot = registry.data_snapshot_for_spec(spec.spec_hash)
            if (
                prior_snapshot is not None
                and prior_snapshot != data_snapshot["snapshot_id"]
            ):
                raise ValueError(
                    "The preregistered spec was previously run against a different "
                    "data snapshot. Change the spec or explicitly create a new study."
                )
        context = create_run_context(experiment_config, "experiment", spec.end_date)
        context.path("experiment_spec.json").write_text(
            spec.model_dump_json(indent=2), encoding="utf-8"
        )
        context.path("research_protocol.json").write_text(
            protocol.model_dump_json(indent=2),
            encoding="utf-8",
        )
        write_data_snapshot(data_snapshot, context.path("data_snapshot.json"))
        context.update(
            research_protocol=protocol.model_dump(mode="json"),
            spec_hash=spec.spec_hash,
            data_snapshot_id=data_snapshot["snapshot_id"],
            trial_number=trial_number,
            study_status=protocol.study_status,
            trial_budget=protocol.trial_budget,
        )

        def register_experiment(
            status: str,
            *,
            metrics: pd.DataFrame | None = None,
            evidence_status: str | None = None,
            approval_status: str = "draft",
        ) -> None:
            if registry is None:
                return
            registry.upsert(
                registry_record_from_run(
                    experiment_config,
                    context.manifest.run_id,
                    "experiment",
                    context.root,
                    status=status,
                    created_at=context.manifest.created_at,
                    config_hash=context.manifest.config_hash,
                    start_date=spec.start_date,
                    end_date=spec.end_date,
                    metrics=metrics,
                    approval_status=approval_status,
                    research_protocol=protocol.model_dump(mode="json"),
                    spec_hash=spec.spec_hash,
                    data_snapshot_id=data_snapshot["snapshot_id"],
                    trial_number=trial_number,
                    evidence_status=evidence_status,
                    evaluation_scope=(
                        "holdout"
                        if protocol.stage == "confirmatory"
                        else "full_sample"
                    ),
                )
            )

        register_experiment("running")
        try:
            validation = validate_research_data(
                experiment_config, spec.start_date, spec.end_date, context=context
            )
            if validation.status != "valid":
                context.update(
                    status="invalid",
                    notes=["Data gate failed; backtest and promotion were not executed."],
                )
                register_experiment("invalid")
                return context
            context.update(status="running")
            data_cache = load_backtest_data(experiment_config)
            if protocol.stage == "confirmatory":
                benchmark_dates = data_cache.prices.loc[
                    data_cache.prices["symbol"]
                    == experiment_config.backtest.primary_benchmark,
                    "date",
                ]
                validate_label_gap(
                    protocol,
                    pd.DatetimeIndex(benchmark_dates),
                    experiment_config.labels.horizon_days,
                )
            research_metadata = {
                "research_protocol": protocol.model_dump(mode="json"),
                "spec_hash": spec.spec_hash,
                "data_snapshot_id": data_snapshot["snapshot_id"],
                "data_snapshot": data_snapshot,
                "trial_number": trial_number,
                "evaluation_scope": (
                    "holdout"
                    if protocol.stage == "confirmatory"
                    else "full_sample"
                ),
            }
            full_result = run_backtest(
                spec.start_date,
                spec.end_date,
                experiment_config,
                publish_latest=False,
                enforce_data_gate=False,
                data_cache=data_cache,
                research_metadata=research_metadata,
            )
            child_runs = [{"period": "full", "run_id": full_result.run_id}]
            robustness_rows: list[dict] = []

            def add_robustness_result(
                test: str,
                setting: str,
                result,
                config_diff: dict | None = None,
            ) -> None:
                values = result.metrics.set_index("metric")["value"].to_dict()
                robustness_rows.append(
                    {
                        "test": test,
                        "setting": setting,
                        "run_id": result.run_id,
                        "cagr": values.get("cagr"),
                        "sharpe_ratio": values.get("sharpe_ratio"),
                        "max_drawdown": values.get("max_drawdown"),
                        "status": "valid",
                        "config_diff": json.dumps(
                            config_diff or {},
                            sort_keys=True,
                        ),
                    }
                )

            def add_robustness_invalid(
                test: str,
                setting: str,
                error: Exception,
                config_diff: dict | None = None,
            ) -> None:
                robustness_rows.append(
                    {
                        "test": test,
                        "setting": setting,
                        "run_id": "",
                        "status": "invalid",
                        "error": str(error),
                        "config_diff": json.dumps(
                            config_diff or {},
                            sort_keys=True,
                        ),
                    }
                )

            add_robustness_result("base", "configured", full_result)
            if "cost_sensitivity" in spec.robustness_tests:
                cost_path = full_result.run_path / "cost_sensitivity.csv"
                cost_frame = (
                    pd.read_csv(cost_path)
                    if cost_path.exists()
                    else pd.DataFrame()
                )
                if cost_frame.empty:
                    add_robustness_invalid(
                        "cost_sensitivity",
                        "all",
                        ValueError("Cost sensitivity artifact is missing or empty."),
                    )
                else:
                    for row in cost_frame.to_dict("records"):
                        robustness_rows.append(
                            {
                                "test": "cost_sensitivity",
                                "setting": f"{float(row['cost_bps']):g}_bps",
                                "run_id": full_result.run_id,
                                "status": "valid",
                                "total_return": row.get("total_return"),
                                "transaction_cost_paid": row.get(
                                    "transaction_cost_paid"
                                ),
                                "config_diff": "{}",
                            }
                        )
            start = pd.Timestamp(spec.start_date)
            end = pd.Timestamp(spec.end_date)
            midpoint = start + (end - start) / 2
            robustness_tasks: list[dict] = []
            if sample_months >= 24 and "subperiod" in spec.robustness_tests:
                robustness_tasks.append(
                    {
                        "test": "subperiod",
                        "setting": "first_half",
                        "period": "first_half",
                        "start_date": str(start.date()),
                        "end_date": str(midpoint.date()),
                        "config": experiment_config,
                        "shift": 0,
                        "blocking": True,
                        "exact_target_count": False,
                        "config_diff": {},
                    }
                )
                robustness_tasks.append(
                    {
                        "test": "subperiod",
                        "setting": "second_half",
                        "period": "second_half",
                        "start_date": str(
                            (midpoint + pd.Timedelta(days=1)).date()
                        ),
                        "end_date": str(end.date()),
                        "config": experiment_config,
                        "shift": 0,
                        "blocking": True,
                        "exact_target_count": False,
                        "config_diff": {},
                    }
                )
            if "top_n_sensitivity" in spec.robustness_tests:
                for top_n in experiment_config.robustness.top_n_values:
                    variant = experiment_config.model_copy(deep=True)
                    variant.optimizer.constraints.target_num_holdings = top_n
                    difference = _config_diff(experiment_config, variant)
                    allowed = {"optimizer.constraints.target_num_holdings"}
                    unexpected = sorted(set(difference) - allowed)
                    if unexpected:
                        raise ValueError(
                            "Top-N robustness changed unsupported fields: "
                            f"{unexpected}"
                        )
                    robustness_tasks.append(
                        {
                            "test": "top_n_sensitivity",
                            "setting": str(top_n),
                            "period": f"top_n_{top_n}",
                            "start_date": spec.start_date,
                            "end_date": spec.end_date,
                            "config": variant,
                            "shift": 0,
                            "blocking": False,
                            "exact_target_count": True,
                            "config_diff": difference,
                        }
                    )
            if "rebalance_day_shift" in spec.robustness_tests:
                for shift in experiment_config.robustness.rebalance_day_shifts:
                    if shift == 0:
                        continue
                    robustness_tasks.append(
                        {
                            "test": "rebalance_day_shift",
                            "setting": str(shift),
                            "period": f"rebalance_shift_{shift:+d}",
                            "start_date": spec.start_date,
                            "end_date": spec.end_date,
                            "config": experiment_config,
                            "shift": shift,
                            "blocking": False,
                            "exact_target_count": False,
                            "config_diff": {},
                        }
                    )

            def run_robustness_task(task: dict):
                try:
                    return run_backtest(
                        task["start_date"],
                        task["end_date"],
                        task["config"],
                        publish_latest=False,
                        enforce_data_gate=False,
                        signal_day_shift=task["shift"],
                        data_cache=data_cache,
                        artifact_level="metrics",
                        exact_target_count=task["exact_target_count"],
                        research_metadata=research_metadata,
                    )
                except (ValueError, RuntimeError) as exc:
                    return exc

            if robustness_tasks:
                workers = min(
                    experiment_config.robustness.parallel_workers,
                    len(robustness_tasks),
                )
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    outcomes = list(
                        executor.map(run_robustness_task, robustness_tasks)
                    )
                for task, outcome in zip(
                    robustness_tasks,
                    outcomes,
                    strict=True,
                ):
                    if isinstance(outcome, Exception):
                        if task["blocking"]:
                            raise outcome
                        add_robustness_invalid(
                            task["test"],
                            task["setting"],
                            outcome,
                            task["config_diff"],
                        )
                        continue
                    child_runs.append(
                        {
                            "period": task["period"],
                            "run_id": outcome.run_id,
                        }
                    )
                    add_robustness_result(
                        task["test"],
                        task["setting"],
                        outcome,
                        task["config_diff"],
                    )
            context.path("child_runs.json").write_text(
                json.dumps(child_runs, indent=2), encoding="utf-8"
            )
            pd.DataFrame(robustness_rows).to_csv(
                context.path("robustness_matrix.csv"),
                index=False,
            )
            for artifact in [
                "factor_metadata.json",
                "feature_snapshot.parquet",
                "label_config.json",
                "label_validation.csv",
                "bias_review.md",
                "bias_review.json",
                "final_report.md",
            ]:
                source = full_result.run_path / artifact
                if source.exists():
                    copyfile(source, context.path(artifact))
            if protocol.stage == "confirmatory":
                for artifact in [
                    "factor_diagnostics.csv",
                    "factor_quantiles.csv",
                    "factor_decay.csv",
                    "factor_correlation.csv",
                ]:
                    source = full_result.run_path / artifact
                    if source.exists():
                        copyfile(
                            source,
                            context.path(f"full_sample_{artifact}"),
                        )
            ml_source = full_result.run_path / "ml_evaluation"
            if ml_source.exists():
                for source in ml_source.iterdir():
                    if source.is_file():
                        copyfile(source, context.path("ml_evaluation", source.name))

            selected_metrics_frame = full_result.metrics
            if protocol.stage == "confirmatory":
                decision, selected_metrics_frame = self._confirmatory_evidence(
                    spec=spec,
                    protocol=protocol,
                    config=experiment_config,
                    data_cache=data_cache,
                    full_result=full_result,
                    context=context,
                    trial_number=trial_number,
                    robustness_rows=robustness_rows,
                )
            else:
                decision = research_evidence_decision(
                    stage=protocol.stage,
                    primary_metric=protocol.primary_metric,
                    threshold=protocol.primary_metric_threshold,
                    bootstrap_summary=pd.DataFrame(),
                    deflated_sharpe={},
                    net_total_return=float(
                        full_result.metrics.set_index("metric")["value"].get(
                            "net_total_return",
                            float("nan"),
                        )
                    ),
                    required_probability=(
                        experiment_config.research_validation
                        .deflated_sharpe_probability
                    ),
                )
                write_research_decision(decision, context.root)
            metrics = selected_metrics_frame.set_index("metric")["value"].to_dict()
            baseline_path = (
                resolve_path(experiment_config.paths.reports)
                / "backtest"
                / "backtest_metrics.csv"
            )
            comparison = pd.DataFrame()
            if baseline_path.exists():
                baseline = pd.read_csv(baseline_path).rename(
                    columns={"value": "legacy_demo_value"}
                )
                current = full_result.metrics[["metric", "value"]].rename(
                    columns={"value": "experiment_value"}
                )
                comparison = current.merge(baseline, on="metric", how="left")
                comparison["difference"] = (
                    comparison["experiment_value"] - comparison["legacy_demo_value"]
                )
            comparison.to_csv(context.path("baseline_comparison.csv"), index=False)
            memo = [
                "# Research Memo",
                "",
                f"## Hypothesis\n{spec.hypothesis}",
                "",
                "## Data Gate\nPassed all configured point-in-time and coverage checks.",
                "",
                "## Workflow",
                (
                    "- Single-factor diagnostics: confirmatory evidence uses "
                    "holdout-only diagnostics."
                    if protocol.stage == "confirmatory"
                    else "- Single-factor diagnostics: generated in the full child run."
                ),
                "- Portfolio backtest: completed.",
                "- Subperiod checks: completed when the sample spans at least two years.",
                "- Cost and delisting sensitivity: generated in every child run.",
                "- Legacy baseline comparison: generated when the legacy metric file is available.",
                "",
                f"- Research stage: `{protocol.stage}`",
                f"- Trial family: `{protocol.trial_family}`",
                f"- Trial number: `{trial_number}`",
                "",
                "## Result",
                f"- Full child backtest run: `{full_result.run_id}`",
                f"- CAGR: `{metrics.get('cagr', float('nan')):.4f}`",
                f"- Sharpe: `{metrics.get('sharpe_ratio', float('nan')):.4f}`",
                f"- Max drawdown: `{metrics.get('max_drawdown', float('nan')):.4f}`",
                "",
                "## Research Evidence Decision",
                f"- `{decision['status']}`",
                "- Artifact validity and research evidence are separate decisions.",
            ]
            context.path("research_memo.md").write_text("\n".join(memo), encoding="utf-8")
            context.update(
                status="valid",
                notes=[f"Completed {len(child_runs)} bounded child backtest runs."],
                evidence_status=decision["status"],
            )
            register_experiment(
                context.manifest.status,
                metrics=selected_metrics_frame,
                evidence_status=decision["status"],
                approval_status="review_required",
            )
            return context
        except KeyboardInterrupt:
            context.update(
                status="invalid",
                notes=["Experiment interrupted before completion."],
            )
            register_experiment("invalid")
            raise
        except Exception as exc:
            context.update(status="invalid", notes=[str(exc)])
            register_experiment("invalid")
            raise


def load_experiment_spec(path: str | Path) -> ExperimentSpec:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    payload = json.loads(text) if target.suffix.lower() == ".json" else yaml.safe_load(text)
    return ExperimentSpec.model_validate(payload)
