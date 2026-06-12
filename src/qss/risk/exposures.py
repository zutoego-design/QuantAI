from __future__ import annotations

import numpy as np
import pandas as pd


def sector_exposure(weights: pd.DataFrame) -> pd.DataFrame:
    return weights.groupby("sector", as_index=False)["target_weight"].sum().rename(columns={"target_weight": "sector_weight"})


def single_name_concentration(weights: pd.DataFrame) -> float:
    return float(weights["target_weight"].max()) if not weights.empty else 0.0


def beta_to_benchmark(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    merged = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    if merged.empty or np.var(merged.iloc[:, 1], ddof=0) == 0:
        return 0.0
    return float(np.cov(merged.iloc[:, 0], merged.iloc[:, 1], ddof=0)[0, 1] / np.var(merged.iloc[:, 1], ddof=0))
