from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from qss.config.schema import AppConfig
from qss.data.storage import append_or_replace_parquet, read_parquet, write_csv
from qss.data.validation import validate_research_data
from qss.logging_utils import logger
from qss.portfolio.constraints import validate_weights
from qss.portfolio.optimizer import OptimizationResult, optimize_portfolio_with_status
from qss.portfolio.orders import build_orders
from qss.reporting.rebalance_report import render_rebalance_report
from qss.risk.covariance import estimate_covariance_from_prices
from qss.runs.manifest import create_run_context


@dataclass
class RebalanceRun:
    portfolio: pd.DataFrame
    orders: pd.DataFrame
    optimizer_status: str
    warning: str | None
    run_id: str
    run_path: Path


def run_rebalance(
    as_of_date: pd.Timestamp,
    config: AppConfig,
    enforce_data_gate: bool = True,
) -> RebalanceRun:
    as_of_date = pd.Timestamp(as_of_date).normalize()
    context = create_run_context(config, "rebalance", as_of_date)
    try:
        if config.runtime.research_mode and enforce_data_gate:
            validation = validate_research_data(
                config,
                config.backtest.start_date,
                str(as_of_date.date()),
                context=context,
            )
            if validation.status != "valid":
                raise ValueError(
                    f"Research data gate failed; review {context.path('checks.csv')}."
                )
        return _run_rebalance(as_of_date, config, context)
    except Exception as exc:
        context.update(status="invalid", notes=[str(exc)])
        raise


def _run_rebalance(as_of_date, config, context) -> RebalanceRun:
    scores = read_parquet(Path(config.paths.gold_data) / "scores" / "alpha_scores.parquet")
    prices = read_parquet(Path(config.paths.silver_data) / "prices" / "prices_daily.parquet")
    existing_weights = read_parquet(Path(config.paths.gold_data) / "portfolios" / "portfolio_weights.parquet")
    scores = scores.loc[scores["date"] == as_of_date].copy()
    if scores.empty:
        raise ValueError(f"No alpha scores found for {as_of_date:%Y-%m-%d}")

    previous = pd.Series(dtype="float64")
    if not existing_weights.empty:
        latest_date = existing_weights.loc[existing_weights["date"] < as_of_date, "date"].max()
        if pd.notna(latest_date):
            previous = existing_weights.loc[existing_weights["date"] == latest_date].set_index("symbol")["target_weight"]

    covariance = estimate_covariance_from_prices(prices, scores["symbol"].tolist(), as_of_date, config.optimizer.covariance)
    opt_result: OptimizationResult = optimize_portfolio_with_status(
        scores=scores,
        covariance=covariance,
        previous_weights=previous,
        sector_map=scores.set_index("symbol")["sector"],
        config=config.optimizer,
    )
    portfolio = opt_result.weights.copy()
    expected_holdings = min(
        config.optimizer.constraints.target_num_holdings, len(scores)
    )
    if config.runtime.research_mode and opt_result.status == "fallback":
        raise ValueError(f"Optimizer fallback is not publishable: {opt_result.warning}")
    if config.runtime.research_mode and len(portfolio) != expected_holdings:
        raise ValueError(
            f"Portfolio has {len(portfolio)} holdings; expected {expected_holdings}."
        )
    portfolio["date"] = as_of_date
    portfolio["strategy_name"] = config.strategy.name
    portfolio = portfolio[["date", "strategy_name", "symbol", "target_weight", "previous_weight", "trade_weight", "sector", "alpha_score"]]
    validate_weights(portfolio, config.optimizer.constraints.max_weight, config.optimizer.constraints.max_sector_weight)
    orders = build_orders(portfolio)

    append_or_replace_parquet(
        portfolio,
        Path(config.paths.gold_data) / "portfolios" / "portfolio_weights.parquet",
        ["date", "strategy_name", "symbol"],
    )
    write_csv(portfolio, context.path("target_weights.csv"))
    write_csv(orders, context.path("orders.csv"))
    html = render_rebalance_report(
        as_of_date=as_of_date,
        portfolio=portfolio,
        optimizer_status=opt_result.status,
        warning=opt_result.warning,
        universe_size=len(scores),
    )
    report_path = context.path("report.html")
    report_path.write_text(html, encoding="utf-8")
    context.update(
        status="valid",
        quality_gates={
            "target_holding_count": len(portfolio) == expected_holdings,
            "optimizer_no_fallback": opt_result.status != "fallback",
        },
    )
    logger.info("Rebalance report written to {}", report_path)
    return RebalanceRun(
        portfolio=portfolio,
        orders=orders,
        optimizer_status=opt_result.status,
        warning=opt_result.warning,
        run_id=context.manifest.run_id,
        run_path=context.root,
    )
