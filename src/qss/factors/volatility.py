from __future__ import annotations

import numpy as np
import pandas as pd


def _max_drawdown(series: pd.Series) -> float:
    clean = series.dropna()
    if clean.empty:
        return np.nan
    wealth = clean / clean.iloc[0]
    drawdown = wealth / wealth.cummax() - 1
    return float(drawdown.min())


def _beta(asset_returns: pd.Series, benchmark_returns: pd.Series, window: int = 252) -> float:
    merged = pd.concat([asset_returns, benchmark_returns], axis=1, join="inner").dropna().tail(window)
    if len(merged) < 20:
        return np.nan
    benchmark_var = np.var(merged.iloc[:, 1], ddof=0)
    if benchmark_var == 0:
        return np.nan
    covariance = np.cov(merged.iloc[:, 0], merged.iloc[:, 1], ddof=0)[0, 1]
    return float(covariance / benchmark_var)


def compute_volatility_factors(
    as_of_date: pd.Timestamp,
    prices: pd.DataFrame,
    symbols: list[str],
    benchmark_symbol: str,
) -> pd.DataFrame:
    subset = prices.loc[
        (prices["date"] <= as_of_date)
        & (prices["symbol"].isin([*symbols, benchmark_symbol]))
    ]
    returns = (
        subset.pivot(index="date", columns="symbol", values="return_1d")
        .sort_index()
        .reindex(columns=[*symbols, benchmark_symbol])
    )
    closes = (
        subset.pivot(index="date", columns="symbol", values="adj_close")
        .sort_index()
        .reindex(columns=symbols)
    )
    assets = returns.reindex(columns=symbols)
    recent_60 = assets.tail(60)
    recent_252 = assets.tail(252)
    vol_60 = recent_60.std(ddof=0).where(recent_60.count() >= 20) * np.sqrt(252)
    vol_252 = (
        recent_252.std(ddof=0).where(recent_252.count() >= 60) * np.sqrt(252)
    )
    benchmark = returns.get(benchmark_symbol, pd.Series(dtype="float64"))

    beta_values: list[float] = []
    drawdowns: list[float] = []
    benchmark_values = benchmark.to_numpy(dtype="float64", na_value=np.nan)
    asset_values = assets.to_numpy(dtype="float64", na_value=np.nan)
    close_values = closes.to_numpy(dtype="float64", na_value=np.nan)
    for column in range(len(symbols)):
        values = asset_values[:, column]
        valid = np.isfinite(values) & np.isfinite(benchmark_values)
        paired_asset = values[valid][-252:]
        paired_benchmark = benchmark_values[valid][-252:]
        if len(paired_asset) < 20:
            beta_values.append(np.nan)
        else:
            benchmark_var = np.var(paired_benchmark, ddof=0)
            beta_values.append(
                np.nan
                if benchmark_var == 0
                else float(
                    np.cov(paired_asset, paired_benchmark, ddof=0)[0, 1]
                    / benchmark_var
                )
            )

        clean_close = close_values[:, column]
        clean_close = clean_close[np.isfinite(clean_close)][-252:]
        if len(clean_close) == 0:
            drawdowns.append(np.nan)
        else:
            wealth = clean_close / clean_close[0]
            drawdowns.append(
                float(np.min(wealth / np.maximum.accumulate(wealth) - 1))
            )

    return pd.DataFrame(
        {
            "symbol": symbols,
            "realized_vol_60d": vol_60.reindex(symbols).to_numpy(),
            "realized_vol_252d": vol_252.reindex(symbols).to_numpy(),
            "beta_to_spy": beta_values,
            "max_drawdown_252d": drawdowns,
        }
    )
