from __future__ import annotations

import numpy as np
import pandas as pd

from qss.data.fundamentals import latest_fundamentals_as_of


def compute_value_factors(
    as_of_date: pd.Timestamp,
    universe: pd.DataFrame,
    fundamentals: pd.DataFrame,
    latest_fundamentals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    latest = (
        latest_fundamentals
        if latest_fundamentals is not None
        else latest_fundamentals_as_of(fundamentals, as_of_date)
    )
    merged = universe.merge(latest, on="symbol", how="left", suffixes=("", "_fund"))
    market_cap = merged["market_cap"].replace({0: np.nan})
    frame = pd.DataFrame(
        {
            "symbol": merged["symbol"],
            "earnings_yield": merged["net_income"] / market_cap,
            "free_cash_flow_yield": merged["free_cash_flow"] / market_cap,
            "book_to_market": merged["shareholders_equity"] / market_cap,
            "sales_yield": merged["revenue"] / market_cap,
        }
    )
    return frame
