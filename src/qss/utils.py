from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def project_root() -> Path:
    return ROOT


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def to_timestamp(value: Any) -> pd.Timestamp:
    return pd.Timestamp(value).tz_localize(None) if pd.Timestamp(value).tzinfo else pd.Timestamp(value)


def latest_by(df: pd.DataFrame, group_col: str, sort_col: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return df.sort_values([group_col, sort_col]).groupby(group_col, as_index=False).tail(1)


def annualize_return(daily_returns: pd.Series) -> float:
    if daily_returns.empty:
        return 0.0
    total = (1 + daily_returns.fillna(0.0)).prod()
    years = len(daily_returns) / 252
    return float(total ** (1 / max(years, 1 / 252)) - 1)


def annualize_volatility(daily_returns: pd.Series) -> float:
    if daily_returns.std(ddof=0) == 0 or daily_returns.empty:
        return 0.0
    return float(daily_returns.std(ddof=0) * np.sqrt(252))
