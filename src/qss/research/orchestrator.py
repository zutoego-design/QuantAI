from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml
from pydantic import BaseModel, Field, field_validator

from qss.backtest.engine import run_backtest
from qss.config.schema import AppConfig
from qss.data.storage import resolve_path
from qss.data.validation import validate_research_data
from qss.runs.manifest import create_run_context


class ExperimentSpec(BaseModel):
    hypothesis: str
    universe: str = "nasdaq_operating_equities_pit"
    factors: list[str] = Field(default_factory=list)
    preprocessing: dict = Field(default_factory=dict)
    portfolio: dict = Field(default_factory=dict)
    costs: dict = Field(default_factory=dict)
    start_date: str
    end_date: str
    seed: int = 42
    max_years: int = 20

    @field_validator("end_date")
    @classmethod
    def validate_dates(cls, value: str, info):
        start = info.data.get("start_date")
        if start and pd.Timestamp(value) <= pd.Timestamp(start):
            raise ValueError("end_date must be after start_date")
        return value


class ResearchOrchestrator:
    """Bounded research runner. It never writes raw inputs or baseline artifacts."""

    def __init__(self, config: AppConfig):
        self.config = config

    def _configured_experiment(self, spec: ExperimentSpec) -> AppConfig:
        config = self.config.model_copy(deep=True)
        if spec.universe != config.universe.name:
            raise ValueError(
                f"Experiment universe {spec.universe!r} is not the configured "
                f"point-in-time universe {config.universe.name!r}."
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
        return config

    def run(self, spec: ExperimentSpec):
        years = (pd.Timestamp(spec.end_date) - pd.Timestamp(spec.start_date)).days / 365.25
        if years > spec.max_years:
            raise ValueError(f"Experiment spans {years:.1f} years; limit is {spec.max_years}.")
        experiment_config = self._configured_experiment(spec)
        context = create_run_context(experiment_config, "experiment", spec.end_date)
        context.path("experiment_spec.json").write_text(
            spec.model_dump_json(indent=2), encoding="utf-8"
        )
        try:
            validation = validate_research_data(
                experiment_config, spec.start_date, spec.end_date, context=context
            )
            if validation.status != "valid":
                context.update(
                    status="invalid",
                    notes=["Data gate failed; backtest and promotion were not executed."],
                )
                return context
            full_result = run_backtest(
                spec.start_date,
                spec.end_date,
                experiment_config,
                publish_latest=False,
                enforce_data_gate=False,
            )
            child_runs = [{"period": "full", "run_id": full_result.run_id}]
            start = pd.Timestamp(spec.start_date)
            end = pd.Timestamp(spec.end_date)
            midpoint = start + (end - start) / 2
            if years >= 2:
                first_result = run_backtest(
                    str(start.date()),
                    str(midpoint.date()),
                    experiment_config,
                    publish_latest=False,
                    enforce_data_gate=False,
                )
                second_result = run_backtest(
                    str((midpoint + pd.Timedelta(days=1)).date()),
                    str(end.date()),
                    experiment_config,
                    publish_latest=False,
                    enforce_data_gate=False,
                )
                child_runs.extend(
                    [
                        {"period": "first_half", "run_id": first_result.run_id},
                        {"period": "second_half", "run_id": second_result.run_id},
                    ]
                )
            context.path("child_runs.json").write_text(
                json.dumps(child_runs, indent=2), encoding="utf-8"
            )

            metrics = full_result.metrics.set_index("metric")["value"].to_dict()
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
                "- Single-factor diagnostics: generated in the full child run.",
                "- Portfolio backtest: completed.",
                "- Subperiod checks: completed when the sample spans at least two years.",
                "- Cost and delisting sensitivity: generated in every child run.",
                "- Legacy baseline comparison: generated when the legacy metric file is available.",
                "",
                "## Result",
                f"- Full child backtest run: `{full_result.run_id}`",
                f"- CAGR: `{metrics.get('cagr', float('nan')):.4f}`",
                f"- Sharpe: `{metrics.get('sharpe_ratio', float('nan')):.4f}`",
                f"- Max drawdown: `{metrics.get('max_drawdown', float('nan')):.4f}`",
                "",
                "## Promotion Decision",
                "Eligible for human review. No automated strategy promotion is performed.",
            ]
            context.path("research_memo.md").write_text("\n".join(memo), encoding="utf-8")
            context.update(
                status="valid",
                notes=[f"Completed {len(child_runs)} bounded child backtest runs."],
            )
            return context
        except Exception as exc:
            context.update(status="invalid", notes=[str(exc)])
            raise


def load_experiment_spec(path: str | Path) -> ExperimentSpec:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    payload = json.loads(text) if target.suffix.lower() == ".json" else yaml.safe_load(text)
    return ExperimentSpec.model_validate(payload)
