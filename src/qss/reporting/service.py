from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from qss.reporting.backtest_report import render_backtest_report
from qss.reporting.schema import ReportBundle


def report_bundle(run_path: str | Path) -> ReportBundle:
    root = Path(run_path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    return ReportBundle(
        run_id=manifest["run_id"],
        root=root,
        manifest=root / "manifest.json",
        html_report=root / "report.html",
        metrics=root / "metrics.csv",
        daily_returns=root / "daily_returns.csv",
        structured_report=root / "report.json",
    )


def render_saved_backtest(run_path: str | Path) -> ReportBundle:
    bundle = report_bundle(run_path)
    daily = pd.read_csv(bundle.daily_returns)
    metrics = pd.read_csv(bundle.metrics)
    rebalances = pd.read_csv(bundle.root / "rebalances.csv")
    drawdowns = pd.read_csv(bundle.root / "drawdown_episodes.csv")
    sectors = pd.read_csv(bundle.root / "sector_exposure.csv")
    factor_diagnostics = pd.read_csv(bundle.root / "factor_diagnostics.csv")
    data_diagnostics = pd.read_csv(bundle.root / "data_diagnostics.csv")
    sensitivity = pd.read_csv(bundle.root / "delisting_sensitivity.csv")
    sector_attribution_path = bundle.root / "sector_return_attribution_summary.csv"
    sector_attribution = (
        pd.read_csv(sector_attribution_path)
        if sector_attribution_path.exists()
        else pd.DataFrame()
    )
    manifest = json.loads(bundle.manifest.read_text(encoding="utf-8"))
    bundle.html_report.write_text(
        render_backtest_report(
            daily,
            metrics,
            rebalances,
            drawdowns=drawdowns,
            holdings=sectors,
            factor_diagnostics=factor_diagnostics,
            data_diagnostics=data_diagnostics,
            delisting_sensitivity=sensitivity,
            sector_attribution=sector_attribution,
            manifest=manifest,
        ),
        encoding="utf-8",
    )
    return bundle
