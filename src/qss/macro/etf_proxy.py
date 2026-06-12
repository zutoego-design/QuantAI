from __future__ import annotations

import numpy as np
import pandas as pd


def compute_etf_proxy_performance(prices: pd.DataFrame, tickers: dict[str, str], as_of_date: pd.Timestamp, lookback_days: int = 63) -> pd.DataFrame:
    subset = prices.loc[(prices["symbol"].isin(tickers.values())) & (prices["date"] <= as_of_date)]
    rows = []
    for label, symbol in tickers.items():
        series = subset.loc[subset["symbol"] == symbol].sort_values("date")["adj_close"]
        if len(series) <= lookback_days:
            value = np.nan
        else:
            value = float(series.iloc[-1] / series.iloc[-lookback_days] - 1)
        rows.append({"proxy": label, "symbol": symbol, "return_3m": value})
    return pd.DataFrame(rows)
