from __future__ import annotations

import numpy as np
import pandas as pd


def latest_series_value(macro: pd.DataFrame, series_id: str, as_of_date: pd.Timestamp) -> float:
    subset = macro.loc[(macro["series_id"] == series_id) & (macro["date"] <= as_of_date)].sort_values("date")
    return float(subset["value"].iloc[-1]) if not subset.empty else np.nan


def year_over_year_change(macro: pd.DataFrame, series_id: str, as_of_date: pd.Timestamp) -> float:
    subset = macro.loc[(macro["series_id"] == series_id) & (macro["date"] <= as_of_date)].sort_values("date")
    if len(subset) < 13:
        return np.nan
    latest = subset["value"].iloc[-1]
    prior = subset["value"].iloc[-13]
    if prior == 0:
        return np.nan
    return float(latest / prior - 1)


def zscore_recent(macro: pd.DataFrame, series_id: str, as_of_date: pd.Timestamp, window: int = 24) -> float:
    subset = macro.loc[(macro["series_id"] == series_id) & (macro["date"] <= as_of_date)].sort_values("date")["value"].tail(window)
    if len(subset) < 6 or subset.std(ddof=0) == 0:
        return 0.0
    return float((subset.iloc[-1] - subset.mean()) / subset.std(ddof=0))
