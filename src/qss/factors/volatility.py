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
    subset = prices.loc[(prices["date"] <= as_of_date) & (prices["symbol"].isin([*symbols, benchmark_symbol]))].copy()
    returns = subset.pivot(index="date", columns="symbol", values="return_1d").sort_index()
    closes = subset.pivot(index="date", columns="symbol", values="adj_close").sort_index()
    benchmark = returns.get(benchmark_symbol, pd.Series(dtype="float64"))
    rows = []
    for symbol in symbols:
        asset_returns = returns.get(symbol, pd.Series(dtype="float64")).dropna()
        close_series = closes.get(symbol, pd.Series(dtype="float64")).dropna()
        rows.append(
            {
                "symbol": symbol,
                "realized_vol_60d": float(asset_returns.tail(60).std(ddof=0) * np.sqrt(252)) if len(asset_returns) >= 20 else np.nan,
                "realized_vol_252d": float(asset_returns.tail(252).std(ddof=0) * np.sqrt(252)) if len(asset_returns) >= 60 else np.nan,
                "beta_to_spy": _beta(asset_returns, benchmark),
                "max_drawdown_252d": _max_drawdown(close_series.tail(252)),
            }
        )
    return pd.DataFrame(rows)
