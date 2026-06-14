from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from qss.backtest.engine import (
    BacktestRunSpec,
    _attach_reference_benchmarks,
    _drift_target_weights,
    _LedgerMarketData,
    _prepare_ledger_market_data,
    _simulate_ledger,
)
from qss.backtest.metrics import compute_backtest_metrics
from qss.config.schema import AppConfig
from qss.data.calendar import next_trading_day
from qss.data.storage import write_csv, write_parquet
from qss.portfolio.optimizer import (
    optimize_portfolio_to_target_count,
    optimize_portfolio_with_status,
)
from qss.risk.covariance import estimate_covariance


@dataclass
class PortfolioEvaluation:
    daily_returns: pd.DataFrame
    metrics: pd.DataFrame
    rebalances: pd.DataFrame
    holdings: pd.DataFrame
    trades: pd.DataFrame


def _drift_weights(
    previous_weights: pd.Series,
    previous_execution: pd.Timestamp | None,
    signal_date: pd.Timestamp,
    returns: pd.DataFrame,
    research_mode: bool,
) -> pd.Series:
    return _drift_target_weights(
        previous_weights,
        previous_execution,
        signal_date,
        returns,
        research_mode,
    )


def targets_from_scores(
    score_frame: pd.DataFrame,
    prices: pd.DataFrame,
    config: AppConfig,
    *,
    exact_target_count: bool = True,
    market_data: _LedgerMarketData | None = None,
) -> dict[pd.Timestamp, dict]:
    required = {"date", "symbol", "total_score"}
    if score_frame.empty or not required.issubset(score_frame.columns):
        raise ValueError(f"Score frame must contain {sorted(required)}.")
    scores = score_frame.copy()
    scores["date"] = pd.to_datetime(scores["date"]).dt.normalize()
    if "sector" not in scores:
        scores["sector"] = "Unknown"
    if "market_cap" not in scores:
        scores["market_cap"] = 1.0
    scores["market_cap"] = pd.to_numeric(
        scores["market_cap"],
        errors="coerce",
    ).fillna(1.0)
    benchmark_dates = prices.loc[
        prices["symbol"] == config.backtest.primary_benchmark,
        "date",
    ]
    calendar_source = benchmark_dates if not benchmark_dates.empty else prices["date"]
    trading_dates = pd.DatetimeIndex(sorted(pd.to_datetime(calendar_source).unique()))
    prepared_market_data = market_data or _prepare_ledger_market_data(prices)
    returns = prepared_market_data.returns
    targets: dict[pd.Timestamp, dict] = {}
    previous_target = pd.Series(dtype=float)
    previous_execution: pd.Timestamp | None = None
    for signal_date, cross_section in scores.groupby("date", sort=True):
        signal_date = pd.Timestamp(signal_date)
        cross_section = cross_section.dropna(subset=["total_score"]).copy()
        if cross_section.empty:
            continue
        target_count = config.optimizer.constraints.target_num_holdings
        if config.runtime.research_mode and len(cross_section) < target_count:
            raise ValueError(
                f"Only {len(cross_section)} scores on {signal_date.date()}; "
                f"{target_count} are required."
            )
        symbols = cross_section["symbol"].tolist()
        covariance = estimate_covariance(
            returns.loc[returns.index <= signal_date].reindex(columns=symbols),
            config.optimizer.covariance,
        )
        covariance = covariance.reindex(
            index=symbols,
            columns=symbols,
        ).fillna(0.0)
        pretrade = _drift_weights(
            previous_target,
            previous_execution,
            signal_date,
            returns,
            config.runtime.research_mode,
        )
        optimizer = (
            optimize_portfolio_to_target_count
            if exact_target_count
            else optimize_portfolio_with_status
        )
        result = optimizer(
            scores=cross_section,
            covariance=covariance,
            previous_weights=pretrade,
            sector_map=cross_section.set_index("symbol")["sector"],
            config=config.optimizer,
        )
        if result.status == "fallback":
            raise ValueError(
                f"Portfolio optimizer failed on {signal_date.date()}: {result.warning}"
            )
        execution_date = next_trading_day(
            signal_date,
            trading_dates,
            config.backtest.rebalance_execution_lag_days,
        )
        universe = cross_section[
            ["symbol", "sector", "market_cap"]
        ].drop_duplicates("symbol")
        universe["included"] = True
        targets[execution_date] = {
            "signal_date": signal_date,
            "weights": result.weights.set_index("symbol")["target_weight"],
            "sectors": result.weights.set_index("symbol")["sector"],
            "optimizer_status": result.status,
            "warning": result.warning,
            "factors": pd.DataFrame(),
            "universe": universe,
        }
        previous_target = result.weights.set_index("symbol")["target_weight"]
        previous_execution = execution_date
    if not targets:
        raise ValueError("No portfolio targets were generated from scores.")
    return targets


def simulate_score_portfolio(
    score_frame: pd.DataFrame,
    prices: pd.DataFrame,
    config: AppConfig,
    *,
    start_date: str,
    end_date: str,
    output_path: str | Path | None = None,
    exact_target_count: bool = True,
    market_data: _LedgerMarketData | None = None,
) -> PortfolioEvaluation:
    prepared_market_data = market_data or _prepare_ledger_market_data(prices)
    targets = targets_from_scores(
        score_frame,
        prices,
        config,
        exact_target_count=exact_target_count,
        market_data=prepared_market_data,
    )
    targets = {
        date: event
        for date, event in targets.items()
        if pd.Timestamp(start_date) <= date <= pd.Timestamp(end_date)
    }
    if not targets:
        raise ValueError("No score targets fall inside the evaluation window.")
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
        market_data=prepared_market_data,
    )
    daily = _attach_reference_benchmarks(
        daily,
        prices,
        targets,
        config.backtest.secondary_benchmark,
    )
    metrics = compute_backtest_metrics(daily, rebalances)
    result = PortfolioEvaluation(
        daily_returns=daily,
        metrics=metrics,
        rebalances=rebalances,
        holdings=holdings,
        trades=trades,
    )
    if output_path is not None:
        root = Path(output_path)
        root.mkdir(parents=True, exist_ok=True)
        for name, frame in [
            ("daily_returns", daily),
            ("metrics", metrics),
            ("rebalances", rebalances),
            ("holdings", holdings),
            ("trades", trades),
        ]:
            write_csv(frame, root / f"{name}.csv")
            write_parquet(frame, root / f"{name}.parquet")
    return result
