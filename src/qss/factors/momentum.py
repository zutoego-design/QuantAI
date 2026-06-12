from __future__ import annotations

import numpy as np
import pandas as pd


def _trailing_return(series: pd.Series, lookback: int, skip_recent: int = 0) -> float:
    clean = series.dropna()
    if len(clean) <= lookback + skip_recent:
        return np.nan
    end_idx = -1 - skip_recent if skip_recent else -1
    start_idx = end_idx - lookback
    start_value = clean.iloc[start_idx]
    end_value = clean.iloc[end_idx]
    if start_value == 0:
        return np.nan
    return float(end_value / start_value - 1)


def compute_momentum_factors(
    as_of_date: pd.Timestamp,
    prices: pd.DataFrame,
    symbols: list[str],
) -> pd.DataFrame:
    price_pivot = (
        prices.loc[(prices["date"] <= as_of_date) & (prices["symbol"].isin(symbols))]
        .sort_values(["symbol", "date"])
        .pivot(index="date", columns="symbol", values="adj_close")
    )
    data = {"symbol": symbols}
    for window, name, skip in [(252, "momentum_12_1", 21), (126, "momentum_6m", 0), (63, "momentum_3m", 0)]:
        data[name] = [ _trailing_return(price_pivot[symbol], window, skip_recent=skip) for symbol in symbols ]
    return pd.DataFrame(data)
