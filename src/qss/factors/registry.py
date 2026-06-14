from __future__ import annotations

from pathlib import Path

import pandas as pd

from qss.config.schema import AppConfig
from qss.data.storage import append_or_replace_parquet, read_parquet
from qss.factors.base import FACTOR_VALUE_COLUMNS
from qss.factors.momentum import compute_momentum_factors
from qss.factors.preprocessing import process_factor_values
from qss.factors.quality import compute_quality_factors
from qss.factors.text_event import compute_risk_disclosure_factor
from qss.factors.value import compute_value_factors
from qss.factors.volatility import compute_volatility_factors


def _melt_factor_frame(
    frame: pd.DataFrame,
    factor_group: str,
    date: pd.Timestamp,
    universe: pd.DataFrame,
    config: AppConfig,
) -> pd.DataFrame:
    direction_map = {
        factor_name: definition.direction
        for factor_name, definition in config.factor_groups[factor_group].factors.items()
    }
    configured_columns = [name for name in direction_map if name in frame.columns]
    frame = frame[["symbol", *configured_columns]]
    melted = frame.melt(id_vars="symbol", var_name="factor_name", value_name="raw_value")
    metadata = universe[["symbol", "sector", "market_cap"]].drop_duplicates("symbol")
    melted = melted.merge(metadata, on="symbol", how="left")
    melted["date"] = pd.Timestamp(date).normalize()
    melted["processed_value"] = pd.NA
    melted["factor_group"] = factor_group
    melted["direction"] = melted["factor_name"].map(direction_map)
    melted["source"] = factor_group
    return melted[FACTOR_VALUE_COLUMNS]


def compute_factor_values_for_date(
    as_of_date: pd.Timestamp,
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    universe: pd.DataFrame,
    config: AppConfig,
    filings: pd.DataFrame | None = None,
    latest_fundamentals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    eligible = universe.loc[universe["included"]].copy()
    if eligible.empty:
        return pd.DataFrame(columns=FACTOR_VALUE_COLUMNS)
    symbols = eligible["symbol"].tolist()
    groups: list[pd.DataFrame] = []
    if "value" in config.factor_groups:
        groups.append(
            _melt_factor_frame(
                compute_value_factors(
                    as_of_date,
                    eligible,
                    fundamentals,
                    latest_fundamentals=latest_fundamentals,
                ),
                "value",
                as_of_date,
                eligible,
                config,
            )
        )
    if "quality" in config.factor_groups:
        groups.append(
            _melt_factor_frame(
                compute_quality_factors(
                    as_of_date,
                    eligible,
                    fundamentals,
                    latest_fundamentals=latest_fundamentals,
                ),
                "quality",
                as_of_date,
                eligible,
                config,
            )
        )
    if "momentum" in config.factor_groups:
        groups.append(
            _melt_factor_frame(
                compute_momentum_factors(as_of_date, prices, symbols),
                "momentum",
                as_of_date,
                eligible,
                config,
            )
        )
    if "low_volatility" in config.factor_groups:
        groups.append(
            _melt_factor_frame(
                compute_volatility_factors(
                    as_of_date, prices, symbols, config.strategy.benchmark
                ),
                "low_volatility",
                as_of_date,
                eligible,
                config,
            )
        )
    if "text_event" in config.factor_groups:
        groups.append(
            _melt_factor_frame(
                compute_risk_disclosure_factor(
                    as_of_date,
                    symbols,
                    filings if filings is not None else pd.DataFrame(),
                    config.text_factors.cache_directory,
                    config.text_factors.risk_terms,
                    config.text_factors.lookback_days,
                ),
                "text_event",
                as_of_date,
                eligible,
                config,
            )
        )
    if not groups:
        return pd.DataFrame(columns=FACTOR_VALUE_COLUMNS)
    factor_values = pd.concat(groups, ignore_index=True)
    factor_values = process_factor_values(factor_values, config)
    return factor_values


def compute_and_store_factor_values(as_of_date: pd.Timestamp, config: AppConfig) -> pd.DataFrame:
    prices = read_parquet(Path(config.paths.silver_data) / "prices" / "prices_daily.parquet")
    fundamentals = read_parquet(
        Path(config.paths.silver_data) / "fundamentals" / "fundamental_observations.parquet"
    )
    if fundamentals.empty:
        fundamentals = read_parquet(
            Path(config.paths.silver_data)
            / "fundamentals"
            / "fundamentals_quarterly.parquet"
        )
    universe = read_parquet(
        Path(config.paths.silver_data) / "universe" / "eligible_universe.parquet"
    )
    filings = read_parquet(
        Path(config.paths.silver_data) / "events" / "sec_filings.parquet"
    )
    factor_values = compute_factor_values_for_date(
        as_of_date,
        prices,
        fundamentals,
        universe.loc[universe["date"] == pd.Timestamp(as_of_date)],
        config,
        filings=filings,
    )
    append_or_replace_parquet(
        factor_values,
        Path(config.paths.gold_data) / "factors" / "factor_values.parquet",
        ["date", "symbol", "factor_name"],
    )
    return factor_values
