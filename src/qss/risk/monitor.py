from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from qss.config.schema import AppConfig
from qss.data.storage import write_parquet
from qss.macro.regime import compute_macro_regime
from qss.reporting.risk_report import render_risk_report
from qss.risk.alerts import generate_alerts
from qss.risk.drawdown import current_drawdown
from qss.risk.exposures import beta_to_benchmark, sector_exposure, single_name_concentration
from qss.runs.manifest import create_run_context


@dataclass
class RiskReport:
    metrics: dict[str, float]
    alerts: pd.DataFrame
    sector_exposure: pd.DataFrame
    macro_summary: pd.DataFrame
    run_id: str
    run_path: Path


def run_daily_risk_monitor(as_of_date: pd.Timestamp, config: AppConfig) -> RiskReport:
    as_of_date = pd.Timestamp(as_of_date).normalize()
    context = create_run_context(config, "risk", as_of_date)
    try:
        return _run_daily_risk_monitor(as_of_date, config, context)
    except Exception as exc:
        context.update(status="invalid", notes=[str(exc)])
        raise


def _run_daily_risk_monitor(as_of_date, config, context) -> RiskReport:
    weights = pd.read_parquet(Path(config.paths.gold_data) / "portfolios" / "portfolio_weights.parquet")
    prices = pd.read_parquet(Path(config.paths.silver_data) / "prices" / "prices_daily.parquet")
    latest_run = Path(config.paths.reports) / "latest_run.json"
    if latest_run.exists():
        run_root = Path(json.loads(latest_run.read_text(encoding="utf-8"))["path"])
        backtest = pd.read_csv(run_root / "daily_returns.csv")
    else:
        if config.runtime.research_mode:
            raise ValueError("Risk monitor requires a valid versioned backtest run.")
        backtest = pd.read_parquet(
            Path(config.paths.gold_data) / "backtests" / "daily_portfolio_returns.parquet"
        )
    macro = pd.read_parquet(Path(config.paths.silver_data) / "macro" / "macro_observations.parquet")
    for frame in [weights, prices, backtest, macro]:
        for column in ["date", "available_date"]:
            if column in frame:
                frame[column] = pd.to_datetime(frame[column]).dt.tz_localize(None)

    latest_date = weights.loc[weights["date"] <= as_of_date, "date"].max()
    portfolio = weights.loc[weights["date"] == latest_date].copy()
    sector_df = sector_exposure(portfolio)

    price_window = prices.loc[prices["date"] <= as_of_date].copy()
    market_date = price_window["date"].max()
    latest_px = price_window.loc[
        price_window["date"] == market_date, ["symbol", "return_1d"]
    ]
    missing_symbols = set(portfolio["symbol"]) - set(
        latest_px.loc[latest_px["return_1d"].notna(), "symbol"]
    )
    if missing_symbols:
        raise ValueError(
            f"Risk monitor is missing current returns for held securities: {sorted(missing_symbols)}"
        )
    daily_return = float(
        portfolio.set_index("symbol")["target_weight"]
        .mul(latest_px.set_index("symbol")["return_1d"])
        .sum()
    )
    history = backtest.loc[backtest["date"] <= as_of_date].copy().sort_values("date")
    benchmark_history = history["benchmark_return"] if not history.empty else pd.Series(dtype="float64")
    portfolio_history = history["portfolio_return"] if not history.empty else pd.Series(dtype="float64")
    realized_vol = float(portfolio_history.tail(60).std(ddof=0) * np.sqrt(252)) if len(portfolio_history) >= 20 else 0.0
    drawdown = current_drawdown(portfolio_history) if not portfolio_history.empty else 0.0
    beta = beta_to_benchmark(portfolio_history.tail(252), benchmark_history.tail(252)) if len(history) >= 20 else 0.0
    tracking_error = float((portfolio_history.tail(252) - benchmark_history.tail(252)).std(ddof=0) * np.sqrt(252)) if len(history) >= 20 else 0.0

    metrics = {
        "daily_loss": daily_return,
        "drawdown": drawdown,
        "realized_vol": realized_vol,
        "beta": beta,
        "single_name_weight": single_name_concentration(portfolio),
        "tracking_error": tracking_error,
        "portfolio_daily_return": daily_return,
        "benchmark_daily_return": float(benchmark_history.iloc[-1]) if not benchmark_history.empty else 0.0,
        "active_return": daily_return - (float(benchmark_history.iloc[-1]) if not benchmark_history.empty else 0.0),
    }
    alerts = generate_alerts(metrics, sector_df, config.risk_limits)
    macro_summary = compute_macro_regime(as_of_date, macro, prices, config)
    report_date = f"{as_of_date:%Y%m%d}"
    alerts_path = context.path("alerts.csv")
    alerts.to_csv(alerts_path, index=False)
    html = render_risk_report(as_of_date, metrics, portfolio, sector_df, alerts, macro_summary)
    html_path = context.path("report.html")
    html_path.write_text(html, encoding="utf-8")
    write_parquet(
        pd.DataFrame([metrics]),
        Path(config.paths.gold_data)
        / "risk_reports"
        / f"risk_report_{report_date}.parquet",
    )
    pd.DataFrame([metrics]).to_csv(context.path("metrics.csv"), index=False)
    context.update(status="valid", quality_gates={"held_returns_complete": True})
    return RiskReport(
        metrics=metrics,
        alerts=alerts,
        sector_exposure=sector_df,
        macro_summary=macro_summary,
        run_id=context.manifest.run_id,
        run_path=context.root,
    )
