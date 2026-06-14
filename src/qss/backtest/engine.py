from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel

from qss.backtest.accounting import compute_drawdown
from qss.backtest.metrics import (
    compounded_monthly_returns,
    comprehensive_factor_diagnostics,
    compute_backtest_metrics,
    drawdown_episodes,
)
from qss.backtest.transaction_costs import estimate_trade_cost
from qss.config.schema import AppConfig
from qss.data.calendar import month_end_dates, next_trading_day
from qss.data.fundamentals import latest_fundamentals_as_of
from qss.data.storage import read_parquet, resolve_path, write_csv, write_parquet
from qss.data.validation import failed_check_summary, validate_research_data
from qss.experiments.registry import ExperimentRegistry, registry_record_from_run
from qss.factors.metadata import write_factor_metadata_snapshot
from qss.factors.registry import compute_factor_values_for_date
from qss.labels.builders import (
    build_cross_sectional_rank_labels,
    build_forward_return_labels,
)
from qss.labels.schema import LabelDefinition
from qss.labels.storage import persist_labels, write_run_label_config
from qss.labels.validation import validate_label_artifact
from qss.model.scoring import compute_alpha_scores
from qss.portfolio.constraints import validate_weights
from qss.portfolio.optimizer import (
    OptimizationResult,
    build_equal_weight_portfolio,
    optimize_portfolio_to_target_count,
    optimize_portfolio_with_status,
)
from qss.reporting.backtest_report import render_backtest_report
from qss.research.audit import build_bias_review, write_bias_review
from qss.risk.covariance import estimate_covariance
from qss.runs.manifest import RunContext, config_hash, create_run_context
from qss.universe.builder import build_universe


class BacktestRunSpec(BaseModel):
    start_date: str
    end_date: str
    initial_capital: float
    execution_lag_days: int = 1
    delisting_return: float = 0.0


@dataclass
class BacktestResult:
    daily_returns: pd.DataFrame
    metrics: pd.DataFrame
    rebalances: pd.DataFrame
    holdings: pd.DataFrame
    trades: pd.DataFrame
    sensitivity: pd.DataFrame
    run_id: str
    run_path: Path


@dataclass
class _LedgerMarketData:
    returns: pd.DataFrame
    last_dates: pd.Series
    histories: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]


@dataclass
class BacktestDataCache:
    prices: pd.DataFrame
    fundamentals: pd.DataFrame
    security_master: pd.DataFrame
    membership: pd.DataFrame
    filings: pd.DataFrame
    factor_snapshots: dict[str, dict[pd.Timestamp, dict]] = field(default_factory=dict)
    market_data: _LedgerMarketData | None = None


def load_backtest_data(config: AppConfig) -> BacktestDataCache:
    prices = read_parquet(
        Path(config.paths.silver_data) / "prices" / "prices_daily.parquet"
    )
    observations = read_parquet(
        Path(config.paths.silver_data)
        / "fundamentals"
        / "fundamental_observations.parquet"
    )
    fundamentals = (
        observations
        if not observations.empty
        else read_parquet(
            Path(config.paths.silver_data)
            / "fundamentals"
            / "fundamentals_quarterly.parquet"
        )
    )
    security_master = read_parquet(
        Path(config.paths.silver_data) / "universe" / "security_master.parquet"
    )
    membership = read_parquet(
        Path(config.paths.silver_data) / "universe" / "universe_membership.parquet"
    )
    filings = read_parquet(
        Path(config.paths.silver_data) / "events" / "sec_filings.parquet"
    )
    for frame in [prices, fundamentals, membership, filings]:
        for column in [
            "date",
            "available_date",
            "period_end_date",
            "filing_date",
        ]:
            if column in frame:
                frame[column] = pd.to_datetime(frame[column]).dt.tz_localize(None)
    return BacktestDataCache(
        prices=prices,
        fundamentals=fundamentals,
        security_master=security_master,
        membership=membership,
        filings=filings,
    )


def _forbid_synthetic(config: AppConfig, *datasets: pd.DataFrame) -> None:
    if not config.runtime.research_mode or config.runtime.allow_synthetic:
        return
    violations = []
    for frame in datasets:
        if "source" in frame:
            count = int(frame["source"].astype(str).str.contains("synthetic", case=False).sum())
            if count:
                violations.append(count)
    if violations:
        raise ValueError(
            f"Research mode forbids synthetic inputs; found {sum(violations)} synthetic rows."
        )


def _point_in_time_master(
    signal_date: pd.Timestamp,
    membership: pd.DataFrame,
    security_master: pd.DataFrame,
    research_mode: bool,
) -> pd.DataFrame:
    if membership.empty or "date" not in membership:
        if research_mode:
            raise ValueError("Point-in-time universe membership is required in research mode.")
        return security_master.copy()
    frame = membership.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    available = frame.loc[frame["date"] <= signal_date]
    if available.empty:
        if research_mode:
            raise ValueError(f"No point-in-time universe snapshot exists on or before {signal_date.date()}.")
        return security_master.copy()
    snapshot_date = available["date"].max()
    if research_mode and (signal_date - snapshot_date).days > 35:
        raise ValueError(
            f"Universe snapshot {snapshot_date.date()} is stale for signal date {signal_date.date()}."
        )
    snapshot = available.loc[(available["date"] == snapshot_date) & available["included"]].copy()
    if "security_id" in snapshot and "security_id" in security_master:
        metadata = security_master.drop_duplicates("security_id")
        snapshot = snapshot.merge(metadata, on="security_id", how="left", suffixes=("", "_master"))
        if "symbol_master" in snapshot:
            snapshot["symbol"] = snapshot["symbol"].fillna(snapshot["symbol_master"])
    elif "symbol" in security_master:
        snapshot = snapshot.merge(
            security_master.drop_duplicates("symbol"), on="symbol", how="left", suffixes=("", "_master")
        )
    for column, default in [
        ("sector", "Unknown"),
        ("security_type", "Common Stock"),
        ("exchange", "XNAS"),
    ]:
        if column not in snapshot:
            snapshot[column] = default
        snapshot[column] = snapshot[column].fillna(default)
    return snapshot


def _drift_target_weights(
    previous_weights: pd.Series,
    previous_execution: pd.Timestamp | None,
    signal_date: pd.Timestamp,
    returns: pd.DataFrame,
    research_mode: bool,
) -> pd.Series:
    if previous_weights.empty or previous_execution is None:
        return previous_weights
    window = returns.loc[
        (returns.index > previous_execution) & (returns.index <= signal_date)
    ].reindex(columns=previous_weights.index)
    growth_values: dict[str, float] = {}
    for symbol in previous_weights.index:
        series = window[symbol]
        valid = series.dropna()
        if valid.empty:
            growth_values[str(symbol)] = 1.0
            continue
        last_valid = valid.index.max()
        if series.loc[series.index <= last_valid].isna().any() and research_mode:
            raise ValueError(
                f"Cannot derive pre-trade weight for {symbol}; "
                "an intermediate return is missing."
            )
        growth_values[str(symbol)] = float((1.0 + series.fillna(0.0)).prod())
    drifted = previous_weights * pd.Series(growth_values).reindex(
        previous_weights.index
    )
    return drifted / drifted.sum()


def _target_schedule(
    start_date: str,
    end_date: str,
    config: AppConfig,
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    security_master: pd.DataFrame,
    membership: pd.DataFrame,
    filings: pd.DataFrame | None = None,
    signal_day_shift: int = 0,
    progress_callback: Callable[[str, float, str], None] | None = None,
    progress_start: float = 0.12,
    progress_end: float = 0.62,
    factor_snapshots: dict[pd.Timestamp, dict] | None = None,
    exact_target_count: bool = False,
    market_data: _LedgerMarketData | None = None,
) -> dict[pd.Timestamp, dict]:
    benchmark_dates = prices.loc[
        prices["symbol"] == config.backtest.primary_benchmark,
        "date",
    ]
    calendar_source = benchmark_dates if not benchmark_dates.empty else prices["date"]
    trading_dates = pd.DatetimeIndex(sorted(calendar_source.unique()))
    signal_dates = []
    for month_end in month_end_dates(start_date, end_date):
        anchor = int(trading_dates.searchsorted(month_end, side="right") - 1)
        shifted = anchor + signal_day_shift
        if anchor >= 0 and 0 <= shifted < len(trading_dates):
            signal_date = pd.Timestamp(trading_dates[shifted]).normalize()
            if pd.Timestamp(start_date) <= signal_date <= pd.Timestamp(end_date):
                signal_dates.append(signal_date)
    targets: dict[pd.Timestamp, dict] = {}
    previous_target = pd.Series(dtype=float)
    previous_execution: pd.Timestamp | None = None
    indexed_prices = prices.sort_values("date").set_index("date", drop=False)
    returns = (
        market_data.returns
        if market_data is not None
        else prices.pivot_table(
            index="date",
            columns="symbol",
            values="return_1d",
            aggfunc="last",
        )
    )
    scheduled_dates = sorted(set(signal_dates))
    total_dates = len(scheduled_dates)
    first_membership_date = (
        pd.to_datetime(membership["date"]).min()
        if not membership.empty and "date" in membership
        else pd.NaT
    )
    for index, signal_date in enumerate(scheduled_dates, start=1):
        if progress_callback is not None:
            fraction = (index - 1) / max(total_dates, 1)
            progress_callback(
                "targets",
                progress_start + (progress_end - progress_start) * fraction,
                f"Generating rebalance {index}/{total_dates} for {signal_date.date()}",
            )
        if pd.notna(first_membership_date) and signal_date < first_membership_date:
            continue
        snapshot = (
            factor_snapshots.get(signal_date)
            if factor_snapshots is not None
            else None
        )
        if snapshot is None:
            price_window_start = signal_date - pd.Timedelta(days=550)
            price_window = indexed_prices.loc[
                price_window_start:signal_date
            ].reset_index(drop=True)
            latest_fundamentals = latest_fundamentals_as_of(
                fundamentals,
                signal_date,
            )
            pit_master = _point_in_time_master(
                signal_date,
                membership,
                security_master,
                config.runtime.research_mode,
            )
            universe = build_universe(
                signal_date,
                price_window,
                fundamentals,
                pit_master,
                config.universe,
                latest_fundamentals=latest_fundamentals,
            )
            factors = compute_factor_values_for_date(
                signal_date,
                price_window,
                fundamentals,
                universe,
                config,
                filings=filings,
                latest_fundamentals=latest_fundamentals,
            )
            scores = compute_alpha_scores(factors, config)
            symbols = scores["symbol"].tolist()
            covariance = estimate_covariance(
                returns.loc[returns.index <= signal_date].reindex(columns=symbols),
                config.optimizer.covariance,
            )
            covariance = covariance.reindex(
                index=symbols,
                columns=symbols,
            ).fillna(0.0)
            snapshot = {
                "factors": factors,
                "scores": scores,
                "covariance": covariance,
                "universe": universe.loc[universe["included"]].copy(),
            }
            if factor_snapshots is not None:
                factor_snapshots[signal_date] = snapshot
        factors = snapshot["factors"]
        scores = snapshot["scores"]
        covariance = snapshot["covariance"]
        universe = snapshot["universe"]
        target_count = config.optimizer.constraints.target_num_holdings
        if config.runtime.research_mode and len(scores) < target_count:
            raise ValueError(
                f"Only {len(scores)} eligible securities on {signal_date.date()}; "
                f"{target_count} are required."
            )
        if scores.empty:
            continue
        pretrade_weights = _drift_target_weights(
            previous_target,
            previous_execution,
            signal_date,
            returns,
            config.runtime.research_mode,
        )
        if config.optimizer.enabled:
            optimizer = (
                optimize_portfolio_to_target_count
                if exact_target_count
                else optimize_portfolio_with_status
            )
            result = optimizer(
                scores=scores,
                covariance=covariance,
                previous_weights=pretrade_weights,
                sector_map=scores.set_index("symbol")["sector"],
                config=config.optimizer,
            )
        else:
            equal_weight = build_equal_weight_portfolio(
                scores,
                target_count,
                max_sector_weight=config.optimizer.constraints.max_sector_weight,
            )
            validate_weights(
                equal_weight,
                config.optimizer.constraints.max_weight,
                config.optimizer.constraints.max_sector_weight,
            )
            result = OptimizationResult(
                weights=equal_weight,
                status="top_n_equal_weight",
            )
        if config.runtime.research_mode and result.status == "fallback":
            raise ValueError(
                f"Optimizer fallback is not publishable on {signal_date.date()}: {result.warning}"
            )
        if result.weights.empty:
            continue
        execution_date = next_trading_day(
            signal_date, trading_dates, config.backtest.rebalance_execution_lag_days
        )
        targets[execution_date] = {
            "signal_date": signal_date,
            "weights": result.weights.set_index("symbol")["target_weight"],
            "sectors": result.weights.set_index("symbol")["sector"],
            "optimizer_status": result.status,
            "warning": result.warning,
            "factors": factors,
            "universe": universe,
        }
        previous_target = result.weights.set_index("symbol")["target_weight"]
        previous_execution = execution_date
    if progress_callback is not None:
        progress_callback(
            "targets",
            progress_end,
            f"Generated {len(targets)} rebalance targets",
        )
    return targets


def _prepare_ledger_market_data(prices: pd.DataFrame) -> _LedgerMarketData:
    returns = prices.pivot_table(
        index="date",
        columns="symbol",
        values="return_1d",
        aggfunc="last",
    )
    last_dates = prices.groupby("symbol", observed=True)["date"].max()
    histories: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    ordered = prices.loc[
        :,
        ["symbol", "date", "adj_close", "volume", "return_1d"],
    ].sort_values(["symbol", "date"])
    for symbol, history in ordered.groupby("symbol", sort=False, observed=True):
        histories[str(symbol)] = (
            history["date"].to_numpy(dtype="datetime64[ns]"),
            (
                pd.to_numeric(history["adj_close"], errors="coerce")
                * pd.to_numeric(history["volume"], errors="coerce")
            ).to_numpy(dtype=float),
            pd.to_numeric(history["return_1d"], errors="coerce").to_numpy(dtype=float),
        )
    return _LedgerMarketData(
        returns=returns,
        last_dates=last_dates,
        histories=histories,
    )


def _latest_market_inputs(
    histories: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    symbol: str,
    as_of: pd.Timestamp,
) -> tuple[float | None, float | None]:
    history = histories.get(symbol)
    if history is None:
        return None, None
    dates, dollar_volume, daily_returns = history
    end = int(
        dates.searchsorted(
            np.datetime64(pd.Timestamp(as_of).to_datetime64()),
            side="right",
        )
    )
    if end == 0:
        return None, None
    adv_values = dollar_volume[max(0, end - 20) : end]
    return_values = daily_returns[max(0, end - 60) : end]
    adv = float(np.nanmean(adv_values)) if np.isfinite(adv_values).any() else np.nan
    volatility = (
        float(np.nanstd(return_values, ddof=0) * np.sqrt(252))
        if np.isfinite(return_values).any()
        else np.nan
    )
    return adv, volatility


def _simulate_ledger(
    spec: BacktestRunSpec,
    config: AppConfig,
    prices: pd.DataFrame,
    targets: dict[pd.Timestamp, dict],
    benchmark_symbol: str,
    market_data: _LedgerMarketData | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    in_window = prices.loc[
        (prices["date"] >= pd.Timestamp(spec.start_date))
        & (prices["date"] <= pd.Timestamp(spec.end_date))
    ]
    benchmark_dates = in_window.loc[
        in_window["symbol"] == benchmark_symbol,
        "date",
    ]
    calendar_source = benchmark_dates if not benchmark_dates.empty else in_window["date"]
    dates = pd.DatetimeIndex(sorted(calendar_source.unique()))
    market_data = market_data or _prepare_ledger_market_data(prices)
    returns = market_data.returns
    last_dates = market_data.last_dates
    benchmark = returns[benchmark_symbol] if benchmark_symbol in returns else pd.Series(dtype=float)
    if config.runtime.research_mode and benchmark.empty:
        raise ValueError(f"Benchmark {benchmark_symbol} has no price history.")

    holdings: dict[str, float] = {}
    cash = float(spec.initial_capital)
    previous_value = float(spec.initial_capital)
    liquidated: set[str] = set()
    daily_rows: list[dict] = []
    rebalance_rows: list[dict] = []
    holding_rows: list[dict] = []
    trade_rows: list[dict] = []
    sector_contribution_rows: list[dict] = []
    current_sectors: pd.Series = pd.Series(dtype=object)

    for day in dates:
        gross_start = previous_value
        missing_return_fills = 0
        daily_sector_contributions: dict[str, float] = {}
        for symbol in list(holdings):
            value = holdings[symbol]
            asset_return = returns.at[day, symbol] if symbol in returns and day in returns.index else np.nan
            if pd.isna(asset_return):
                last_date = last_dates.get(symbol, pd.NaT)
                if pd.notna(last_date) and day > last_date:
                    sector = str(current_sectors.get(symbol, "Unknown") or "Unknown")
                    contribution = (
                        value / gross_start * spec.delisting_return if gross_start else 0.0
                    )
                    daily_sector_contributions[sector] = (
                        daily_sector_contributions.get(sector, 0.0) + contribution
                    )
                    value *= 1.0 + spec.delisting_return
                    cash += value
                    del holdings[symbol]
                    liquidated.add(symbol)
                    continue
                if config.runtime.research_mode:
                    raise ValueError(
                        f"Missing return for held security {symbol} on {day.date()} "
                        "before its last trading date."
                    )
                asset_return = 0.0
                missing_return_fills += 1
            sector = str(current_sectors.get(symbol, "Unknown") or "Unknown")
            contribution = value / gross_start * float(asset_return) if gross_start else 0.0
            daily_sector_contributions[sector] = (
                daily_sector_contributions.get(sector, 0.0) + contribution
            )
            holdings[symbol] = value * (1.0 + float(asset_return))

        pretrade_value = cash + sum(holdings.values())
        gross_return = pretrade_value / gross_start - 1.0 if gross_start else 0.0
        turnover = 0.0
        total_cost = 0.0
        max_participation = 0.0

        if day in targets:
            event = targets[day]
            target_weights = event["weights"]
            current_weights = pd.Series(
                {symbol: value / pretrade_value for symbol, value in holdings.items()},
                dtype=float,
            )
            all_symbols = current_weights.index.union(target_weights.index)
            trades = (
                target_weights.reindex(all_symbols).fillna(0.0)
                - current_weights.reindex(all_symbols).fillna(0.0)
            ) * pretrade_value
            turnover = float(trades.abs().sum() / pretrade_value) if pretrade_value else 0.0
            if (
                config.runtime.research_mode
                and not current_weights.empty
                and turnover
                > config.optimizer.constraints.max_turnover_per_rebalance + 1e-6
            ):
                raise ValueError(
                    f"Realized turnover {turnover:.2%} on {day.date()} exceeds "
                    f"{config.optimizer.constraints.max_turnover_per_rebalance:.2%}."
                )
            commission = slippage = market_impact = 0.0
            for symbol, notional in trades.abs().items():
                adv, volatility = _latest_market_inputs(
                    market_data.histories,
                    symbol,
                    event["signal_date"],
                )
                estimate = estimate_trade_cost(
                    float(notional),
                    pretrade_value,
                    adv,
                    volatility,
                    config.backtest.transaction_cost.commission_bps,
                    config.backtest.transaction_cost.slippage_bps,
                    config.backtest.transaction_cost.market_impact_coefficient,
                )
                if (
                    config.runtime.research_mode
                    and estimate.adv_participation
                    > config.backtest.transaction_cost.max_adv_participation
                ):
                    raise ValueError(
                        f"Trade in {symbol} requires {estimate.adv_participation:.2%} of ADV, "
                        f"above the configured limit."
                    )
                commission += estimate.commission
                slippage += estimate.slippage
                market_impact += estimate.market_impact
                max_participation = max(max_participation, estimate.adv_participation)
                trade_rows.append(
                    {
                        "signal_date": event["signal_date"],
                        "execution_date": day,
                        "symbol": symbol,
                        "trade_notional": float(trades.loc[symbol]),
                        "current_weight": float(current_weights.get(symbol, 0.0)),
                        "target_weight": float(target_weights.get(symbol, 0.0)),
                        "commission": estimate.commission,
                        "slippage": estimate.slippage,
                        "market_impact": estimate.market_impact,
                        "total_cost": estimate.total,
                        "adv_participation": estimate.adv_participation,
                    }
                )
            total_cost = commission + slippage + market_impact
            investable = max(pretrade_value - total_cost, 0.0)
            holdings = {
                symbol: investable * float(weight)
                for symbol, weight in target_weights.items()
                if weight > 0
            }
            cash = investable - sum(holdings.values())
            weights = pd.Series(holdings) / investable if investable else pd.Series(dtype=float)
            hhi = float((weights**2).sum()) if not weights.empty else 0.0
            rebalance_rows.append(
                {
                    "signal_date": event["signal_date"],
                    "execution_date": day,
                    "optimizer_status": event["optimizer_status"],
                    "warning": event["warning"],
                    "turnover": turnover,
                    "holding_count": len(holdings),
                    "commission": commission,
                    "slippage": slippage,
                    "market_impact": market_impact,
                    "total_cost": total_cost,
                    "cost_fraction": total_cost / pretrade_value if pretrade_value else 0.0,
                    "max_adv_participation": max_participation,
                    "hhi": hhi,
                    "effective_holdings": 1 / hhi if hhi else 0.0,
                }
            )
            current_sectors = event["sectors"]

        portfolio_value = cash + sum(holdings.values())
        if total_cost:
            daily_sector_contributions["Transaction Costs"] = (
                daily_sector_contributions.get("Transaction Costs", 0.0)
                - total_cost / gross_start
                if gross_start
                else 0.0
            )
        sector_contribution_rows.extend(
            {
                "date": day,
                "sector": sector,
                "portfolio_contribution": contribution,
            }
            for sector, contribution in daily_sector_contributions.items()
        )
        for symbol, value in holdings.items():
            holding_rows.append(
                {
                    "date": day,
                    "symbol": symbol,
                    "weight": float(value / portfolio_value) if portfolio_value else 0.0,
                    "sector": current_sectors.get(symbol, "Unknown"),
                    "position_value": value,
                }
            )
        net_return = portfolio_value / previous_value - 1.0 if previous_value else 0.0
        if day in benchmark.index:
            benchmark_return = benchmark.loc[day]
            if pd.isna(benchmark_return):
                if day == dates[0]:
                    benchmark_return = 0.0
                else:
                    raise ValueError(f"Benchmark return is missing on {day.date()}.")
        else:
            if config.runtime.research_mode:
                raise ValueError(f"Benchmark date {day.date()} is missing.")
            benchmark_return = 0.0
        daily_rows.append(
            {
                "date": day,
                "portfolio_return": net_return,
                "gross_return": gross_return,
                "benchmark_return": float(benchmark_return),
                "active_return": net_return - float(benchmark_return),
                "portfolio_value": portfolio_value,
                "turnover": turnover,
                "transaction_cost": total_cost,
                "cash": cash,
                "delisting_liquidations": len(liquidated),
                "missing_return_fills": missing_return_fills,
            }
        )
        previous_value = portfolio_value

    daily = pd.DataFrame(daily_rows)
    daily["drawdown"] = compute_drawdown(daily["portfolio_return"])
    daily["benchmark_value"] = (
        (1 + daily["benchmark_return"]).cumprod() * spec.initial_capital
    )
    return (
        daily,
        pd.DataFrame(rebalance_rows),
        pd.DataFrame(holding_rows),
        pd.DataFrame(trade_rows),
        pd.DataFrame(sector_contribution_rows),
    )


def _attach_reference_benchmarks(
    daily: pd.DataFrame,
    prices: pd.DataFrame,
    targets: dict[pd.Timestamp, dict],
    secondary_symbol: str,
) -> pd.DataFrame:
    frame = daily.copy()
    returns = prices.pivot_table(
        index="date", columns="symbol", values="return_1d", aggfunc="last"
    )
    executions = sorted(targets)
    equal_returns: list[float] = []
    cap_returns: list[float] = []
    coverage_rows: list[float] = []
    secondary_returns: list[float] = []
    for day in pd.to_datetime(frame["date"]):
        prior = [execution for execution in executions if execution <= day]
        if not prior or day not in returns.index:
            equal_returns.append(0.0)
            cap_returns.append(0.0)
            coverage_rows.append(0.0)
        else:
            universe = targets[prior[-1]]["universe"]
            symbols = universe["symbol"].drop_duplicates()
            cross = returns.loc[day].reindex(symbols)
            valid = cross.dropna()
            coverage_rows.append(len(valid) / len(symbols) if len(symbols) else 0.0)
            equal_returns.append(float(valid.mean()) if not valid.empty else 0.0)
            caps = universe.drop_duplicates("symbol").set_index("symbol")["market_cap"].reindex(valid.index)
            caps = caps.where(caps > 0).dropna()
            if caps.empty:
                cap_returns.append(float(valid.mean()) if not valid.empty else 0.0)
            else:
                weights = caps / caps.sum()
                cap_returns.append(float((valid.reindex(weights.index) * weights).sum()))
        secondary = (
            returns.at[day, secondary_symbol]
            if day in returns.index and secondary_symbol in returns
            else np.nan
        )
        secondary_returns.append(float(secondary) if pd.notna(secondary) else np.nan)
    frame["secondary_benchmark_return"] = secondary_returns
    frame["internal_equal_weight_return"] = equal_returns
    frame["internal_cap_weight_return"] = cap_returns
    frame["internal_benchmark_coverage"] = coverage_rows
    return frame


def _sector_return_attribution(
    portfolio_contributions: pd.DataFrame,
    daily: pd.DataFrame,
    prices: pd.DataFrame,
    targets: dict[pd.Timestamp, dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    portfolio = portfolio_contributions.copy()
    if portfolio.empty:
        portfolio = pd.DataFrame(
            columns=["date", "sector", "portfolio_contribution"]
        )
    portfolio["date"] = pd.to_datetime(portfolio["date"])
    portfolio = (
        portfolio.groupby(["date", "sector"], as_index=False)["portfolio_contribution"]
        .sum()
    )

    returns = prices.pivot_table(
        index="date",
        columns="symbol",
        values="return_1d",
        aggfunc="last",
    )
    executions = sorted(targets)
    current_execution: pd.Timestamp | None = None
    execution_index = 0
    benchmark_rows: list[dict] = []
    for day in pd.to_datetime(daily["date"]):
        while (
            execution_index < len(executions)
            and executions[execution_index] <= day
        ):
            current_execution = executions[execution_index]
            execution_index += 1
        if current_execution is None or day not in returns.index:
            continue
        universe = targets[current_execution]["universe"].drop_duplicates("symbol").copy()
        universe["sector"] = universe["sector"].fillna("Unknown").replace("", "Unknown")
        cross = returns.loc[day].reindex(universe["symbol"]).dropna()
        if cross.empty:
            continue
        indexed = universe.set_index("symbol")
        caps = pd.to_numeric(indexed["market_cap"], errors="coerce").reindex(cross.index)
        valid_caps = caps.where(caps > 0).dropna()
        if valid_caps.empty:
            weights = pd.Series(1.0 / len(cross), index=cross.index)
        else:
            weights = valid_caps / valid_caps.sum()
            cross = cross.reindex(weights.index)
        sectors = indexed["sector"].reindex(weights.index).fillna("Unknown")
        contributions = (cross * weights).groupby(sectors).sum()
        benchmark_rows.extend(
            {
                "date": day,
                "sector": str(sector or "Unknown"),
                "internal_cap_benchmark_contribution": float(contribution),
            }
            for sector, contribution in contributions.items()
        )

    benchmark = pd.DataFrame(benchmark_rows)
    if benchmark.empty:
        benchmark = pd.DataFrame(
            columns=[
                "date",
                "sector",
                "internal_cap_benchmark_contribution",
            ]
        )
    attribution = portfolio.merge(
        benchmark,
        on=["date", "sector"],
        how="outer",
    ).fillna(
        {
            "portfolio_contribution": 0.0,
            "internal_cap_benchmark_contribution": 0.0,
        }
    )
    attribution["active_contribution"] = (
        attribution["portfolio_contribution"]
        - attribution["internal_cap_benchmark_contribution"]
    )

    daily_index = daily.copy()
    daily_index["date"] = pd.to_datetime(daily_index["date"])
    daily_index["portfolio_start_wealth"] = (
        (1.0 + daily_index["portfolio_return"]).cumprod().shift(fill_value=1.0)
    )
    daily_index["benchmark_start_wealth"] = (
        (1.0 + daily_index["internal_cap_weight_return"])
        .cumprod()
        .shift(fill_value=1.0)
    )
    attribution = attribution.merge(
        daily_index[
            ["date", "portfolio_start_wealth", "benchmark_start_wealth"]
        ],
        on="date",
        how="left",
        validate="many_to_one",
    )
    attribution["portfolio_linked_contribution"] = (
        attribution["portfolio_contribution"] * attribution["portfolio_start_wealth"]
    )
    attribution["benchmark_linked_contribution"] = (
        attribution["internal_cap_benchmark_contribution"]
        * attribution["benchmark_start_wealth"]
    )
    attribution["active_linked_contribution"] = (
        attribution["portfolio_linked_contribution"]
        - attribution["benchmark_linked_contribution"]
    )
    attribution = attribution.sort_values(["date", "sector"]).reset_index(drop=True)
    summary = (
        attribution.groupby("sector", as_index=False)[
            [
                "portfolio_linked_contribution",
                "benchmark_linked_contribution",
                "active_linked_contribution",
            ]
        ]
        .sum()
        .sort_values("active_linked_contribution", ascending=False)
        .reset_index(drop=True)
    )
    return attribution, summary


def _data_diagnostics(
    targets: dict[pd.Timestamp, dict],
    membership: pd.DataFrame,
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    daily: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []
    prior_symbols: set[str] | None = None
    for execution, event in sorted(targets.items()):
        universe = event["universe"]
        symbols = set(universe["symbol"])
        listed = len(symbols - prior_symbols) if prior_symbols is not None else len(symbols)
        delisted = len(prior_symbols - symbols) if prior_symbols is not None else 0
        price_symbols = set(
            prices.loc[prices["date"] == event["signal_date"], "symbol"]
        )
        fundamental_symbols = (
            set(
                fundamentals.loc[
                    pd.to_datetime(fundamentals["available_date"])
                    <= event["signal_date"],
                    "symbol",
                ]
            )
            if {"symbol", "available_date"}.issubset(fundamentals.columns)
            else set()
        )
        rows.append(
            {
                "signal_date": event["signal_date"],
                "execution_date": execution,
                "universe_size": len(symbols),
                "listed_since_prior": listed,
                "delisted_since_prior": delisted,
                "price_coverage": len(symbols & price_symbols) / len(symbols) if symbols else 0.0,
                "fundamental_coverage": len(symbols & fundamental_symbols) / len(symbols)
                if symbols
                else 0.0,
            }
        )
        prior_symbols = symbols
    result = pd.DataFrame(rows)
    if not result.empty:
        result["minimum_internal_benchmark_coverage"] = float(
            daily["internal_benchmark_coverage"].min()
        )
        result["delisting_liquidations"] = int(daily["delisting_liquidations"].max())
    return result


def _portfolio_diagnostics(
    holdings: pd.DataFrame,
    trades: pd.DataFrame,
    targets: dict[pd.Timestamp, dict],
    initial_capital: float,
    max_adv_participation: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    concentration_rows: list[dict] = []
    if not holdings.empty:
        for date, cross in holdings.groupby("date"):
            weights = cross["weight"].sort_values(ascending=False)
            hhi = float((weights**2).sum())
            concentration_rows.append(
                {
                    "date": date,
                    "hhi": hhi,
                    "effective_holdings": 1 / hhi if hhi else 0.0,
                    "largest_weight": float(weights.iloc[0]),
                    "top_10_weight": float(weights.head(10).sum()),
                }
            )
    concentration = pd.DataFrame(concentration_rows)

    exposure_rows: list[dict] = []
    for execution, event in targets.items():
        portfolio = holdings.loc[holdings["date"] == execution]
        portfolio_sector = (
            portfolio.groupby("sector")["weight"].sum() if not portfolio.empty else pd.Series(dtype=float)
        )
        universe = event["universe"].drop_duplicates("symbol").copy()
        universe["market_cap"] = pd.to_numeric(universe["market_cap"], errors="coerce")
        valid = universe.loc[universe["market_cap"] > 0]
        if valid.empty:
            benchmark_sector = pd.Series(dtype=float)
        else:
            benchmark_sector = (
                valid.groupby("sector")["market_cap"].sum() / valid["market_cap"].sum()
            )
        for sector in portfolio_sector.index.union(benchmark_sector.index):
            portfolio_weight = float(portfolio_sector.get(sector, 0.0))
            benchmark_weight = float(benchmark_sector.get(sector, 0.0))
            exposure_rows.append(
                {
                    "date": execution,
                    "sector": sector,
                    "portfolio_weight": portfolio_weight,
                    "internal_cap_benchmark_weight": benchmark_weight,
                    "active_weight": portfolio_weight - benchmark_weight,
                }
            )
    exposures = pd.DataFrame(exposure_rows)

    holding_period = 0.0
    if not holdings.empty:
        spans = holdings.groupby("symbol")["date"].agg(["min", "max"])
        holding_period = float(
            (pd.to_datetime(spans["max"]) - pd.to_datetime(spans["min"])).dt.days.mean()
        )
    capacity = np.nan
    if not trades.empty:
        participation = trades["adv_participation"].replace([np.inf, -np.inf], np.nan).dropna()
        participation = participation.loc[participation > 0]
        if not participation.empty:
            capacity = float(
                (initial_capital * max_adv_participation / participation).min()
            )
    summary = {
        "average_holding_period_days": holding_period,
        "estimated_capacity": capacity,
        "average_hhi": float(concentration["hhi"].mean()) if not concentration.empty else 0.0,
        "average_effective_holdings": float(concentration["effective_holdings"].mean())
        if not concentration.empty
        else 0.0,
    }
    return concentration, exposures, summary


def run_backtest(
    start_date: str,
    end_date: str,
    config: AppConfig,
    publish_latest: bool = True,
    enforce_data_gate: bool = True,
    signal_day_shift: int = 0,
    progress_callback: Callable[[str, float, str], None] | None = None,
    data_cache: BacktestDataCache | None = None,
    artifact_level: Literal["full", "metrics"] = "full",
    exact_target_count: bool = False,
    research_metadata: dict | None = None,
) -> BacktestResult:
    context: RunContext = create_run_context(config, "backtest", end_date)
    if research_metadata:
        context.update(
            research_protocol=research_metadata.get("research_protocol"),
            spec_hash=research_metadata.get("spec_hash"),
            data_snapshot_id=research_metadata.get("data_snapshot_id"),
            trial_number=research_metadata.get("trial_number"),
        )
        if research_metadata.get("data_snapshot") is not None:
            context.path("data_snapshot.json").write_text(
                json.dumps(research_metadata["data_snapshot"], indent=2),
                encoding="utf-8",
            )
    try:
        if progress_callback is not None:
            progress_callback("validation", 0.02, "Validating research data")
        if config.runtime.research_mode and enforce_data_gate:
            validation = validate_research_data(
                config, start_date, end_date, context=context
            )
            if validation.status != "valid":
                raise ValueError(
                    "Research data gate failed: "
                    f"{failed_check_summary(validation.checks)}. "
                    f"Review {context.path('checks.csv')}."
                )
        if progress_callback is not None:
            progress_callback("loading", 0.06, "Loading research datasets")
        cache = data_cache or load_backtest_data(config)
        prices = cache.prices
        fundamentals = cache.fundamentals
        security_master = cache.security_master
        membership = cache.membership
        filings = cache.filings
        _forbid_synthetic(config, prices, fundamentals)
        snapshot_namespace = cache.factor_snapshots.setdefault(
            config_hash(config),
            {},
        )
        if cache.market_data is None:
            cache.market_data = _prepare_ledger_market_data(prices)
        ledger_market_data = cache.market_data
        targets = _target_schedule(
            start_date,
            end_date,
            config,
            prices,
            fundamentals,
            security_master,
            membership,
            filings=filings,
            signal_day_shift=signal_day_shift,
            progress_callback=progress_callback,
            factor_snapshots=snapshot_namespace,
            exact_target_count=exact_target_count,
            market_data=ledger_market_data,
        )
        if not targets:
            raise ValueError("Backtest generated no valid rebalance targets.")

        if progress_callback is not None:
            progress_callback("ledger", 0.62, "Simulating delisting scenarios")
        scenario_results: dict[
            float,
            tuple[
                pd.DataFrame,
                pd.DataFrame,
                pd.DataFrame,
                pd.DataFrame,
                pd.DataFrame,
            ],
        ] = {}
        sensitivity_rows: list[dict] = []
        delisting_scenarios = (
            [0.0]
            if artifact_level == "metrics"
            else config.backtest.delisting_return_scenarios
        )
        for scenario_index, scenario in enumerate(delisting_scenarios, start=1):
            if progress_callback is not None:
                progress_callback(
                    "ledger",
                    0.62 + 0.08 * (scenario_index - 1) / max(len(delisting_scenarios), 1),
                    f"Simulating delisting scenario {scenario_index}/{len(delisting_scenarios)}",
                )
            spec = BacktestRunSpec(
                start_date=start_date,
                end_date=end_date,
                initial_capital=config.backtest.initial_capital,
                execution_lag_days=config.backtest.rebalance_execution_lag_days,
                delisting_return=scenario,
            )
            result = _simulate_ledger(
                spec,
                config,
                prices,
                targets,
                config.backtest.primary_benchmark,
                market_data=ledger_market_data,
            )
            scenario_results[float(scenario)] = result
            sensitivity_rows.append(
                {
                    "delisting_return": scenario,
                    "ending_value": float(result[0]["portfolio_value"].iloc[-1]),
                    "total_return": float(
                        result[0]["portfolio_value"].iloc[-1] / spec.initial_capital - 1
                    ),
                    "delisting_liquidations": int(
                        result[0]["delisting_liquidations"].iloc[-1]
                    ),
                    "validation_status": (
                        "observed"
                        if int(result[0]["delisting_liquidations"].iloc[-1]) > 0
                        else "not_observed"
                    ),
                }
            )

        daily, rebalances, holdings, trades, portfolio_sector_contributions = (
            scenario_results[0.0]
        )
        sensitivity = pd.DataFrame(sensitivity_rows)
        if artifact_level == "metrics":
            daily = _attach_reference_benchmarks(
                daily,
                prices,
                targets,
                config.backtest.secondary_benchmark,
            )
            metrics = compute_backtest_metrics(daily, rebalances)
            write_csv(daily, context.path("daily_returns.csv"))
            write_csv(metrics, context.path("metrics.csv"))
            write_csv(rebalances, context.path("rebalances.csv"))
            write_parquet(daily, context.path("daily_returns.parquet"))
            write_parquet(metrics, context.path("metrics.parquet"))
            write_parquet(rebalances, context.path("rebalances.parquet"))
            context.update(
                status="valid",
                quality_gates={
                    "artifact_level": "metrics",
                    "rebalance_targets": len(targets),
                },
                notes=[
                    "Metrics-only robustness child; full diagnostics are owned by "
                    "the experiment's canonical full child."
                ],
            )
            if config.registry.enabled:
                ExperimentRegistry.from_config(config).upsert(
                    registry_record_from_run(
                        config,
                        context.manifest.run_id,
                        "backtest",
                        context.root,
                        status=context.manifest.status,
                        created_at=context.manifest.created_at,
                        config_hash=context.manifest.config_hash,
                        start_date=start_date,
                        end_date=end_date,
                        metrics=metrics,
                        research_protocol=context.manifest.research_protocol,
                        spec_hash=context.manifest.spec_hash,
                        data_snapshot_id=context.manifest.data_snapshot_id,
                        trial_number=context.manifest.trial_number,
                        evidence_status=context.manifest.evidence_status,
                        evaluation_scope=research_metadata.get("evaluation_scope")
                        if research_metadata
                        else None,
                    )
                )
            if progress_callback is not None:
                progress_callback("complete", 1.0, "Backtest complete")
            return BacktestResult(
                daily_returns=daily,
                metrics=metrics,
                rebalances=rebalances,
                holdings=holdings,
                trades=trades,
                sensitivity=sensitivity,
                run_id=context.manifest.run_id,
                run_path=context.root,
            )
        if progress_callback is not None:
            progress_callback("costs", 0.70, "Running transaction-cost sensitivity")
        cost_sensitivity_rows = []
        cost_scenarios = config.robustness.cost_sensitivity_bps
        for cost_index, cost_bps in enumerate(cost_scenarios, start=1):
            if progress_callback is not None:
                progress_callback(
                    "costs",
                    0.70 + 0.08 * (cost_index - 1) / max(len(cost_scenarios), 1),
                    f"Testing cost scenario {cost_index}/{len(cost_scenarios)}",
                )
            cost_config = config.model_copy(deep=True)
            cost_config.backtest.transaction_cost.commission_bps = 0.0
            cost_config.backtest.transaction_cost.slippage_bps = float(cost_bps)
            cost_config.backtest.transaction_cost.market_impact_coefficient = 0.0
            cost_result = _simulate_ledger(
                BacktestRunSpec(
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=config.backtest.initial_capital,
                    execution_lag_days=config.backtest.rebalance_execution_lag_days,
                    delisting_return=0.0,
                ),
                cost_config,
                prices,
                targets,
                config.backtest.primary_benchmark,
                market_data=ledger_market_data,
            )
            cost_sensitivity_rows.append(
                {
                    "cost_bps": float(cost_bps),
                    "ending_value": float(cost_result[0]["portfolio_value"].iloc[-1]),
                    "total_return": float(
                        cost_result[0]["portfolio_value"].iloc[-1]
                        / config.backtest.initial_capital
                        - 1
                    ),
                    "transaction_cost_paid": float(
                        cost_result[0]["transaction_cost"].sum()
                    ),
                }
            )
        cost_sensitivity = pd.DataFrame(cost_sensitivity_rows)
        daily = _attach_reference_benchmarks(
            daily, prices, targets, config.backtest.secondary_benchmark
        )
        sector_attribution, sector_attribution_summary = _sector_return_attribution(
            portfolio_sector_contributions,
            daily,
            prices,
            targets,
        )
        if config.runtime.research_mode:
            invested = daily.loc[daily["date"] >= min(targets)]
            if invested["secondary_benchmark_return"].isna().any():
                raise ValueError(
                    f"Secondary benchmark {config.backtest.secondary_benchmark} is incomplete."
                )
            minimum_internal_coverage = float(
                invested["internal_benchmark_coverage"].min()
            )
            if minimum_internal_coverage < config.universe.min_long_price_coverage:
                raise ValueError(
                    "Internal benchmark coverage fell to "
                    f"{minimum_internal_coverage:.2%}, below "
                    f"{config.universe.min_long_price_coverage:.2%}."
                )
        metrics = compute_backtest_metrics(daily, rebalances)
        drawdowns = drawdown_episodes(daily["portfolio_return"], daily["date"])
        monthly = compounded_monthly_returns(daily)
        all_factors = pd.concat(
            [event["factors"] for event in targets.values()], ignore_index=True
        )
        if progress_callback is not None:
            progress_callback("labels", 0.80, "Building labels and feature snapshot")
        write_parquet(all_factors, context.path("feature_snapshot.parquet"))
        write_factor_metadata_snapshot(config, context.path("factor_metadata.json"))
        label_definitions: list[LabelDefinition] = []
        label_frames: list[pd.DataFrame] = []
        if config.labels.enabled:
            forward_definition = LabelDefinition(
                name="forward_return",
                horizon_days=config.labels.horizon_days,
                start_offset_days=config.labels.start_offset_days,
                embargo_days=config.labels.embargo_days,
                version=config.labels.version,
            )
            forward_labels = build_forward_return_labels(
                prices,
                pd.DatetimeIndex(all_factors["date"].drop_duplicates()),
                forward_definition,
                symbols=sorted(all_factors["symbol"].unique()),
            )
            if "forward_return" in config.labels.label_types:
                label_definitions.append(forward_definition)
                label_frames.append(forward_labels)
                persist_labels(forward_labels, forward_definition, config)
                write_parquet(
                    forward_labels,
                    context.path("labels_forward_return.parquet"),
                )
            if "cross_sectional_rank" in config.labels.label_types:
                rank_definition = LabelDefinition(
                    name="cross_sectional_rank",
                    horizon_days=config.labels.horizon_days,
                    start_offset_days=config.labels.start_offset_days,
                    embargo_days=config.labels.embargo_days,
                    version=config.labels.version,
                )
                rank_labels = build_cross_sectional_rank_labels(
                    forward_labels,
                    rank_definition,
                )
                label_definitions.append(rank_definition)
                label_frames.append(rank_labels)
                persist_labels(rank_labels, rank_definition, config)
                write_parquet(
                    rank_labels,
                    context.path("labels_cross_sectional_rank.parquet"),
                )
            write_run_label_config(label_definitions, context.root)
            combined_labels = (
                pd.concat(label_frames, ignore_index=True)
                if label_frames
                else pd.DataFrame()
            )
            label_checks = validate_label_artifact(combined_labels, all_factors)
            write_csv(label_checks, context.path("label_validation.csv"))
            if config.ml.enabled:
                from qss.model.evaluation import evaluate_walk_forward

                evaluate_walk_forward(
                    all_factors,
                    combined_labels,
                    config.ml,
                    context.path("ml_evaluation"),
                )
        if progress_callback is not None:
            progress_callback("diagnostics", 0.86, "Computing diagnostics")
        factor_reports = comprehensive_factor_diagnostics(all_factors, prices)
        data_diagnostics = _data_diagnostics(
            targets, membership, prices, fundamentals, daily
        )
        concentration, sector_exposure, portfolio_summary = _portfolio_diagnostics(
            holdings,
            trades,
            targets,
            config.backtest.initial_capital,
            config.backtest.transaction_cost.max_adv_participation,
        )
        metrics = pd.concat(
            [
                metrics,
                pd.DataFrame(
                    [
                        {"category": "portfolio", "metric": key, "value": value}
                        for key, value in portfolio_summary.items()
                    ]
                ),
            ],
            ignore_index=True,
        )

        if progress_callback is not None:
            progress_callback("artifacts", 0.93, "Writing reports and artifacts")
        write_csv(daily, context.path("daily_returns.csv"))
        write_csv(metrics, context.path("metrics.csv"))
        write_csv(rebalances, context.path("rebalances.csv"))
        write_csv(holdings, context.path("holdings.csv"))
        write_csv(trades, context.path("trades.csv"))
        write_csv(drawdowns, context.path("drawdown_episodes.csv"))
        write_csv(monthly, context.path("monthly_returns.csv"))
        write_csv(sensitivity, context.path("delisting_sensitivity.csv"))
        write_csv(cost_sensitivity, context.path("cost_sensitivity.csv"))
        write_csv(sector_exposure, context.path("sector_exposure.csv"))
        write_csv(
            sector_attribution,
            context.path("sector_return_attribution.csv"),
        )
        write_csv(
            sector_attribution_summary,
            context.path("sector_return_attribution_summary.csv"),
        )
        write_csv(concentration, context.path("concentration.csv"))
        write_csv(factor_reports["summary"], context.path("factor_diagnostics.csv"))
        write_csv(factor_reports["quantiles"], context.path("factor_quantiles.csv"))
        write_csv(factor_reports["decay"], context.path("factor_decay.csv"))
        write_csv(factor_reports["correlation"], context.path("factor_correlation.csv"))
        write_csv(data_diagnostics, context.path("data_diagnostics.csv"))
        write_parquet(daily, context.path("daily_returns.parquet"))
        write_parquet(rebalances, context.path("rebalances.parquet"))
        write_parquet(metrics, context.path("metrics.parquet"))
        write_parquet(holdings, context.path("holdings.parquet"))
        write_parquet(trades, context.path("trades.parquet"))
        report = render_backtest_report(
            daily,
            metrics,
            rebalances,
            drawdowns=drawdowns,
            holdings=sector_exposure,
            factor_diagnostics=factor_reports["summary"],
            data_diagnostics=data_diagnostics,
            delisting_sensitivity=sensitivity,
            sector_attribution=sector_attribution_summary,
            manifest=context.manifest.__dict__,
        )
        context.path("report.html").write_text(report, encoding="utf-8")
        context.path("report.json").write_text(
            json.dumps(
                {
                    "schema_version": context.manifest.report_schema_version,
                    "run_id": context.manifest.run_id,
                    "status": "valid",
                    "metrics": metrics.to_dict("records"),
                    "artifacts": {
                        "daily_returns": "daily_returns.parquet",
                        "rebalances": "rebalances.parquet",
                        "holdings": "holdings.parquet",
                        "trades": "trades.parquet",
                        "factor_diagnostics": "factor_diagnostics.csv",
                        "factor_metadata": "factor_metadata.json",
                        "feature_snapshot": "feature_snapshot.parquet",
                        "label_config": "label_config.json",
                        "cost_sensitivity": "cost_sensitivity.csv",
                        "sector_return_attribution": "sector_return_attribution.csv",
                        "sector_return_attribution_summary": (
                            "sector_return_attribution_summary.csv"
                        ),
                        "bias_review": "bias_review.md",
                        "data_diagnostics": "data_diagnostics.csv",
                        "html": "report.html",
                    },
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        synthetic_rows = 0
        for frame in [prices, fundamentals]:
            if "source" in frame:
                synthetic_rows += int(
                    frame["source"]
                    .astype(str)
                    .str.contains("synthetic", case=False)
                    .sum()
                )
        quickstart_mode = not config.runtime.research_mode
        current_membership_mode = (
            config.universe.membership_mode == "current_snapshot"
        )
        sp500_history_mode = (
            config.universe.membership_mode == "point_in_time"
            and config.universe.long_history_provider == "sp500_wikipedia"
        )
        context.update(
            status="valid",
            quality_gates={
                "synthetic_rows": synthetic_rows,
                "benchmark_complete": not daily["benchmark_return"].isna().any(),
                "secondary_benchmark_complete": not daily[
                    "secondary_benchmark_return"
                ].isna().any(),
                "internal_benchmark_coverage": float(
                    daily.loc[daily["date"] >= min(targets), "internal_benchmark_coverage"].min()
                ),
                "missing_returns_filled_with_zero": bool(
                    daily["missing_return_fills"].sum()
                ),
                "missing_return_fill_count": int(
                    daily["missing_return_fills"].sum()
                ),
            },
            bias_flags=[
                *(
                    ["current_membership_backfilled_survivorship_bias"]
                    if current_membership_mode
                    else ["sp500_point_in_time_wikipedia_reconstruction"]
                    if sp500_history_mode
                    else ["free_data_long_history_approximate"]
                ),
                *(
                    [
                        "quickstart_current_membership",
                        "quickstart_synthetic_fundamentals",
                    ]
                    if quickstart_mode
                    else []
                ),
            ],
            notes=(
                [
                    "Quickstart backtest: suitable for system evaluation, not trusted "
                    "point-in-time investment research."
                ]
                if quickstart_mode
                else (
                    [
                        "Current Nasdaq membership was backfilled across history. "
                        "Results contain survivorship bias and exclude historical "
                        "constituent changes and many delisted securities."
                    ]
                    if current_membership_mode
                    else (
                        [
                            "S&P 500 membership was reconstructed point-in-time "
                            "from Wikipedia constituent changes; no synthetic "
                            "prices or fundamentals were used."
                        ]
                        if sp500_history_mode
                        else None
                    )
                )
            ),
        )
        bias_review = build_bias_review(
            manifest=context.manifest.__dict__,
            factor_diagnostics=factor_reports["summary"],
            sector_exposure=sector_exposure,
            sector_attribution=sector_attribution_summary,
            concentration=concentration,
            cost_sensitivity=cost_sensitivity,
            data_diagnostics=data_diagnostics,
            delisting_sensitivity=sensitivity,
        )
        write_bias_review(bias_review, context.root)
        dominant_sector = "None"
        investable_attribution = sector_attribution_summary.loc[
            ~sector_attribution_summary["sector"].isin(
                ["Transaction Costs", "Unknown"]
            )
        ]
        if not investable_attribution.empty:
            dominant = investable_attribution.iloc[
                investable_attribution["portfolio_linked_contribution"]
                .abs()
                .argmax()
            ]
            dominant_sector = (
                f"{dominant['sector']} "
                f"({float(dominant['portfolio_linked_contribution']):.2%})"
            )
        context.path("final_report.md").write_text(
            "\n".join(
                [
                    "# Final Research Report",
                    "",
                    f"- Run ID: `{context.manifest.run_id}`",
                    f"- Status: `{context.manifest.status}`",
                    f"- Strategy: `{config.strategy.name}`",
                    f"- Universe: `{config.universe.name}`",
                    f"- Bias review recommendation: `{bias_review['recommendation']}`",
                    f"- Largest absolute sector contribution: `{dominant_sector}`",
                    "",
                    "All numeric results are sourced from deterministic run artifacts.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        if config.registry.enabled:
            ExperimentRegistry.from_config(config).upsert(
                registry_record_from_run(
                    config,
                    context.manifest.run_id,
                    "backtest",
                    context.root,
                    status=context.manifest.status,
                    created_at=context.manifest.created_at,
                    config_hash=context.manifest.config_hash,
                    start_date=start_date,
                    end_date=end_date,
                    metrics=metrics,
                    research_protocol=context.manifest.research_protocol,
                    spec_hash=context.manifest.spec_hash,
                    data_snapshot_id=context.manifest.data_snapshot_id,
                    trial_number=context.manifest.trial_number,
                    evidence_status=context.manifest.evidence_status,
                    evaluation_scope=research_metadata.get("evaluation_scope")
                    if research_metadata
                    else None,
                )
            )
        if publish_latest:
            latest = resolve_path(config.paths.reports) / "latest_run.json"
            latest.write_text(
                json.dumps(
                    {"run_id": context.manifest.run_id, "path": str(context.root)},
                    indent=2,
                ),
                encoding="utf-8",
            )
        if progress_callback is not None:
            progress_callback("complete", 1.0, "Backtest complete")
        return BacktestResult(
            daily_returns=daily,
            metrics=metrics,
            rebalances=rebalances,
            holdings=holdings,
            trades=trades,
            sensitivity=sensitivity,
            run_id=context.manifest.run_id,
            run_path=context.root,
        )
    except Exception as exc:
        if progress_callback is not None:
            progress_callback("failed", 1.0, f"Backtest failed: {exc}")
        context.update(status="invalid", notes=[str(exc)])
        raise
