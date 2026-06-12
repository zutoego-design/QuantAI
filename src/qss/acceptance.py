from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from qss.backtest.metrics import compounded_monthly_returns
from qss.config.schema import AppConfig
from qss.data.storage import resolve_path, write_csv
from qss.reporting.service import report_bundle
from qss.runs.manifest import create_run_context


def run_acceptance_checks(config: AppConfig, run_path: str | Path | None = None):
    if run_path is None:
        latest_path = resolve_path(config.paths.reports) / "latest_run.json"
        if not latest_path.exists():
            raise ValueError("No latest run pointer exists.")
        run_path = json.loads(latest_path.read_text(encoding="utf-8"))["path"]
    bundle = report_bundle(run_path)
    context = create_run_context(config, "acceptance")
    checks: list[dict] = []

    missing = bundle.validate()
    checks.append({"check": "report_bundle_complete", "passed": not missing, "details": str(missing)})
    manifest = json.loads(bundle.manifest.read_text(encoding="utf-8"))
    structured = json.loads(bundle.structured_report.read_text(encoding="utf-8"))
    checks.append(
        {
            "check": "source_run_valid",
            "passed": manifest.get("status") == "valid",
            "details": manifest.get("status"),
        }
    )
    checks.append(
        {
            "check": "report_schema_version",
            "passed": structured.get("schema_version")
            == manifest.get("report_schema_version"),
            "details": structured.get("schema_version"),
        }
    )
    daily = pd.read_csv(bundle.daily_returns)
    metrics = pd.read_csv(bundle.metrics)
    rebalances = pd.read_csv(bundle.root / "rebalances.csv")
    sensitivity = pd.read_csv(bundle.root / "delisting_sensitivity.csv")
    saved_monthly = pd.read_csv(bundle.root / "monthly_returns.csv")
    recalculated = compounded_monthly_returns(daily)
    monthly_match = np.allclose(
        saved_monthly["portfolio_return"],
        recalculated["portfolio_return"],
        atol=1e-12,
        equal_nan=True,
    )
    checks.append({"check": "monthly_returns_compounded", "passed": monthly_match, "details": ""})
    checks.append(
        {
            "check": "daily_returns_complete",
            "passed": not daily[["portfolio_return", "benchmark_return"]].isna().any().any(),
            "details": str(daily[["portfolio_return", "benchmark_return"]].isna().sum().to_dict()),
        }
    )
    expected_holdings = config.optimizer.constraints.target_num_holdings
    holdings_ok = (
        not rebalances.empty
        and bool((rebalances["holding_count"] == expected_holdings).all())
    )
    checks.append(
        {
            "check": "target_holding_count",
            "passed": holdings_ok,
            "details": f"expected={expected_holdings}",
        }
    )
    scenarios = set(np.round(sensitivity["delisting_return"], 2))
    checks.append(
        {
            "check": "delisting_sensitivity_complete",
            "passed": scenarios == {0.0, -0.3, -1.0},
            "details": str(sorted(scenarios)),
        }
    )
    required_metrics = {
        "cagr",
        "annualized_volatility",
        "downside_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "omega_ratio",
        "var_95_daily",
        "cvar_95_daily",
        "alpha_annualized",
        "beta",
        "correlation",
        "r_squared",
        "tracking_error",
        "information_ratio",
        "up_capture",
        "down_capture",
    }
    missing_metrics = required_metrics - set(metrics["metric"])
    checks.append(
        {
            "check": "professional_metrics_complete",
            "passed": not missing_metrics,
            "details": str(sorted(missing_metrics)),
        }
    )
    frame = pd.DataFrame(checks)
    status = "valid" if bool(frame["passed"].all()) else "invalid"
    write_csv(frame, context.path("acceptance_checks.csv"))
    context.update(
        status=status,
        quality_gates={row["check"]: bool(row["passed"]) for row in checks},
        notes=[f"Validated source run {bundle.run_id}."],
    )
    return frame, context
