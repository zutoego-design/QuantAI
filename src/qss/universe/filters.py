from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from qss.config.schema import UniverseConfig


@dataclass
class UniverseFilterInputs:
    market_cap: float | None
    price: float | None
    adv_20d: float | None
    history_days: int
    completeness: float
    sector: str
    security_type: str
    price_staleness_days: int | None = None


def evaluate_filters(inputs: UniverseFilterInputs, config: UniverseConfig) -> list[str]:
    reasons: list[str] = []
    if pd.isna(inputs.market_cap) or inputs.market_cap < config.filters.min_market_cap:
        reasons.append("market_cap")
    if pd.isna(inputs.price) or inputs.price < config.filters.min_price:
        reasons.append("price")
    if pd.isna(inputs.adv_20d) or inputs.adv_20d < config.filters.min_adv_20d:
        reasons.append("adv_20d")
    if inputs.history_days < config.filters.min_history_days:
        reasons.append("history")
    if pd.isna(inputs.completeness) or inputs.completeness < config.filters.min_price_data_completeness:
        reasons.append("completeness")
    if (
        inputs.price_staleness_days is None
        or inputs.price_staleness_days > config.filters.max_price_staleness_days
    ):
        reasons.append("price_stale")
    if inputs.sector in config.exclude.sectors:
        reasons.append("sector_excluded")
    if inputs.security_type in config.exclude.security_types:
        reasons.append("security_type_excluded")
    return reasons


def rolling_adv(prices: pd.DataFrame, window: int = 20) -> pd.Series:
    return (
        prices.sort_values("date")
        .assign(dollar_volume=lambda df: df["adj_close"] * df["volume"])
        .groupby("symbol")["dollar_volume"]
        .rolling(window=window, min_periods=min(window, 5))
        .mean()
        .reset_index(level=0, drop=True)
    )
