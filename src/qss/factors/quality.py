from __future__ import annotations

import numpy as np
import pandas as pd

from qss.data.fundamentals import latest_fundamentals_as_of


def compute_quality_factors(
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
    equity = merged["shareholders_equity"].replace({0: np.nan})
    revenue = merged["revenue"].replace({0: np.nan})
    assets = merged["total_assets"].replace({0: np.nan})
    return pd.DataFrame(
        {
            "symbol": merged["symbol"],
            "roe": merged["net_income"] / equity,
            "gross_margin": merged["gross_profit"] / revenue,
            "operating_margin": merged["operating_income"] / revenue,
            "debt_to_equity": merged["total_liabilities"] / equity,
            "accruals": (merged["net_income"] - merged["operating_cash_flow"]) / assets,
        }
    )
