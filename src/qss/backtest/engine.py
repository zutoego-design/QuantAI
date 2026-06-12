from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

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
from qss.data.storage import read_parquet, resolve_path, write_csv, write_parquet
from qss.data.validation import validate_research_data
from qss.factors.registry import compute_factor_values_for_date
from qss.model.scoring import compute_alpha_scores
from qss.portfolio.optimizer import optimize_portfolio_with_status
from qss.reporting.backtest_report import render_backtest_report
from qss.risk.covariance import estimate_covariance_from_prices
from qss.runs.manifest import RunContext, create_run_context
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


def _target_schedule(
    start_date: str,
    end_date: str,
    config: AppConfig,
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    security_master: pd.DataFrame,
    membership: pd.DataFrame,
) -> dict[pd.Timestamp, dict]:
    trading_dates = pd.DatetimeIndex(sorted(prices["date"].unique()))
    signal_dates = []
    for month_end in month_end_dates(start_date, end_date):
        prior = trading_dates[trading_dates <= month_end]
        if len(prior):
            signal_dates.append(pd.Timestamp(prior[-1]).normalize())
    targets: dict[pd.Timestamp, dict] = {}
    previous_target = pd.Series(dtype=float)
    previous_execution: pd.Timestamp | None = None
    for signal_date in sorted(set(signal_dates)):
        pit_master = _point_in_time_master(
            signal_date, membership, security_master, config.runtime.research_mode
        )
        universe = build_universe(signal_date, prices, fundamentals, pit_master, config.universe)
        factors = compute_factor_values_for_date(signal_date, prices, fundamentals, universe, config)
        scores = compute_alpha_scores(factors, config)
        target_count = config.optimizer.constraints.target_num_holdings
        if config.runtime.research_mode and len(scores) < target_count:
            raise ValueError(
                f"Only {len(scores)} eligible securities on {signal_date.date()}; "
                f"{target_count} are required."
            )
        if scores.empty:
            continue
        covariance = estimate_covariance_from_prices(
            prices, scores["symbol"].tolist(), signal_date, config.optimizer.covariance
        )
        pretrade_weights = previous_target
        if not previous_target.empty and previous_execution is not None:
            expected_dates = pd.DatetimeIndex(
                sorted(
                    prices.loc[
                        (prices["date"] > previous_execution)
                        & (prices["date"] <= signal_date),
                        "date",
                    ].unique()
                )
            )
            growth_values: dict[str, float] = {}
            for symbol in previous_target.index:
                series = (
                    prices.loc[
                        (prices["symbol"] == symbol)
                        & (prices["date"] > previous_execution)
                        & (prices["date"] <= signal_date),
                        ["date", "return_1d"],
                    ]
                    .drop_duplicates("date")
                    .set_index("date")["return_1d"]
                    .reindex(expected_dates)
                )
                valid = series.dropna()
                if valid.empty:
                    growth_values[symbol] = 1.0
                    continue
                last_valid = valid.index.max()
                if series.loc[series.index <= last_valid].isna().any():
                    raise ValueError(
                        f"Cannot derive pre-trade weight for {symbol}; "
                        "an intermediate return is missing."
                    )
                growth_values[symbol] = float((1.0 + valid).prod())
            growth = pd.Series(growth_values)
            drifted_values = previous_target * growth
            pretrade_weights = drifted_values / drifted_values.sum()
        result = optimize_portfolio_with_status(
            scores=scores,
            covariance=covariance,
            previous_weights=pretrade_weights,
            sector_map=scores.set_index("symbol")["sector"],
            config=config.optimizer,
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
            "universe": universe.loc[universe["included"]].copy(),
        }
        previous_target = result.weights.set_index("symbol")["target_weight"]
        previous_execution = execution_date
    return targets


def _latest_market_inputs(
    prices: pd.DataFrame,
    symbol: str,
    as_of: pd.Timestamp,
) -> tuple[float | None, float | None]:
    history = prices.loc[
        (prices["symbol"] == symbol) & (prices["date"] <= as_of)
    ].sort_values("date").tail(60)
    if history.empty:
        return None, None
    adv = float((history["adj_close"] * history["volume"]).tail(20).mean())
    volatility = float(history["return_1d"].std(ddof=0) * np.sqrt(252))
    return adv, volatility


def _simulate_ledger(
    spec: BacktestRunSpec,
    config: AppConfig,
    prices: pd.DataFrame,
    targets: dict[pd.Timestamp, dict],
    benchmark_symbol: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.DatetimeIndex(
        sorted(
            prices.loc[
                (prices["date"] >= pd.Timestamp(spec.start_date))
                & (prices["date"] <= pd.Timestamp(spec.end_date)),
                "date",
            ].unique()
        )
    )
    returns = prices.pivot_table(index="date", columns="symbol", values="return_1d", aggfunc="last")
    last_dates = prices.groupby("symbol")["date"].max()
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
    current_sectors: pd.Series = pd.Series(dtype=object)

    for day in dates:
        gross_start = previous_value
        for symbol in list(holdings):
            value = holdings[symbol]
            asset_return = returns.at[day, symbol] if symbol in returns and day in returns.index else np.nan
            if pd.isna(asset_return):
                last_date = last_dates.get(symbol, pd.NaT)
                if pd.notna(last_date) and day > last_date:
                    value *= 1.0 + spec.delisting_return
                    cash += value
                    del holdings[symbol]
                    liquidated.add(symbol)
                    continue
                raise ValueError(
                    f"Missing return for held security {symbol} on {day.date()} before its last trading date."
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
                adv, volatility = _latest_market_inputs(prices, symbol, event["signal_date"])
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
) -> BacktestResult:
    context: RunContext = create_run_context(config, "backtest", end_date)
    try:
        if config.runtime.research_mode and enforce_data_gate:
            validation = validate_research_data(
                config, start_date, end_date, context=context
            )
            if validation.status != "valid":
                raise ValueError(
                    f"Research data gate failed; review {context.path('checks.csv')}."
                )
        prices = read_parquet(Path(config.paths.silver_data) / "prices" / "prices_daily.parquet")
        observations = read_parquet(
            Path(config.paths.silver_data) / "fundamentals" / "fundamental_observations.parquet"
        )
        fundamentals = (
            observations
            if not observations.empty
            else read_parquet(
                Path(config.paths.silver_data) / "fundamentals" / "fundamentals_quarterly.parquet"
            )
        )
        security_master = read_parquet(
            Path(config.paths.silver_data) / "universe" / "security_master.parquet"
        )
        membership = read_parquet(
            Path(config.paths.silver_data) / "universe" / "universe_membership.parquet"
        )
        for frame in [prices, fundamentals, membership]:
            for column in ["date", "available_date", "period_end_date", "filing_date"]:
                if column in frame:
                    frame[column] = pd.to_datetime(frame[column]).dt.tz_localize(None)
        _forbid_synthetic(config, prices, fundamentals)
        targets = _target_schedule(
            start_date,
            end_date,
            config,
            prices,
            fundamentals,
            security_master,
            membership,
        )
        if not targets:
            raise ValueError("Backtest generated no valid rebalance targets.")

        scenario_results: dict[
            float, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ] = {}
        sensitivity_rows: list[dict] = []
        for scenario in config.backtest.delisting_return_scenarios:
            spec = BacktestRunSpec(
                start_date=start_date,
                end_date=end_date,
                initial_capital=config.backtest.initial_capital,
                execution_lag_days=config.backtest.rebalance_execution_lag_days,
                delisting_return=scenario,
            )
            result = _simulate_ledger(
                spec, config, prices, targets, config.backtest.primary_benchmark
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
                }
            )

        daily, rebalances, holdings, trades = scenario_results[0.0]
        daily = _attach_reference_benchmarks(
            daily, prices, targets, config.backtest.secondary_benchmark
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
        sensitivity = pd.DataFrame(sensitivity_rows)
        all_factors = pd.concat(
            [event["factors"] for event in targets.values()], ignore_index=True
        )
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

        write_csv(daily, context.path("daily_returns.csv"))
        write_csv(metrics, context.path("metrics.csv"))
        write_csv(rebalances, context.path("rebalances.csv"))
        write_csv(holdings, context.path("holdings.csv"))
        write_csv(trades, context.path("trades.csv"))
        write_csv(drawdowns, context.path("drawdown_episodes.csv"))
        write_csv(monthly, context.path("monthly_returns.csv"))
        write_csv(sensitivity, context.path("delisting_sensitivity.csv"))
        write_csv(sector_exposure, context.path("sector_exposure.csv"))
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
                        "data_diagnostics": "data_diagnostics.csv",
                        "html": "report.html",
                    },
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        context.update(
            status="valid",
            quality_gates={
                "synthetic_rows": 0,
                "benchmark_complete": True,
                "secondary_benchmark_complete": True,
                "internal_benchmark_coverage": float(
                    daily.loc[daily["date"] >= min(targets), "internal_benchmark_coverage"].min()
                ),
                "missing_returns_filled_with_zero": False,
            },
            bias_flags=["free_data_long_history_approximate"],
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
        context.update(status="invalid", notes=[str(exc)])
        raise
