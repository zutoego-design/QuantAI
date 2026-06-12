from __future__ import annotations

from pathlib import Path

import pandas as pd

from qss.config.schema import AppConfig, UniverseConfig
from qss.data.fundamentals import latest_fundamentals_as_of
from qss.data.storage import append_or_replace_parquet, read_parquet
from qss.universe.filters import UniverseFilterInputs, evaluate_filters, rolling_adv
from qss.universe.providers import default_universe_provider


def build_universe(
    as_of_date: pd.Timestamp,
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    security_master: pd.DataFrame,
    config: UniverseConfig,
) -> pd.DataFrame:
    as_of_date = pd.Timestamp(as_of_date).normalize()
    security_master = security_master.copy()
    if "sector" not in security_master:
        security_master["sector"] = "Unknown"
    if "security_type" not in security_master:
        security_master["security_type"] = "Common Stock"
    prices = prices.loc[prices["date"] <= as_of_date].copy()
    prices = prices.sort_values(["symbol", "date"])
    if prices.empty:
        return pd.DataFrame(
            columns=["date", "symbol", "included", "exclusion_reason", "market_cap", "price", "adv_20d", "sector"]
        )

    prices["adv_20d"] = rolling_adv(prices)
    latest_prices = prices.groupby("symbol", as_index=False).tail(1).rename(
        columns={"adj_close": "price", "date": "price_date"}
    )
    latest_prices["price_staleness_days"] = (
        as_of_date - pd.to_datetime(latest_prices["price_date"])
    ).dt.days
    history = prices.groupby("symbol").agg(
        history_days=("date", "count"),
        start_date=("date", "min"),
        end_date=("date", "max"),
    )
    trading_dates = pd.DatetimeIndex(sorted(prices["date"].dropna().unique()))
    required_dates = trading_dates[-config.filters.min_history_days :]
    recent_counts = (
        prices.loc[prices["date"].isin(required_dates)]
        .groupby("symbol")["date"]
        .nunique()
    )
    history["completeness"] = recent_counts.reindex(history.index).fillna(0) / max(
        len(required_dates), 1
    )

    latest_fund = latest_fundamentals_as_of(fundamentals, as_of_date)
    latest_fund["market_cap"] = latest_fund["shares_outstanding"] * latest_prices.set_index("symbol")["price"].reindex(latest_fund["symbol"]).values

    if "listing_date" in security_master:
        listing_date = pd.to_datetime(security_master["listing_date"], errors="coerce")
        security_master = security_master.loc[listing_date.isna() | (listing_date <= as_of_date)]
    if "delisting_date" in security_master:
        delisting_date = pd.to_datetime(security_master["delisting_date"], errors="coerce")
        security_master = security_master.loc[delisting_date.isna() | (delisting_date >= as_of_date)]
    universe = security_master.merge(
        latest_prices[
            ["symbol", "price", "adv_20d", "price_date", "price_staleness_days"]
        ],
        on="symbol",
        how="left",
    )
    universe = universe.merge(history[["history_days", "completeness"]], left_on="symbol", right_index=True, how="left")
    universe = universe.merge(latest_fund[["symbol", "market_cap"]], on="symbol", how="left")
    universe["date"] = as_of_date
    universe["history_days"] = universe["history_days"].fillna(0).astype(int)
    universe["completeness"] = universe["completeness"].fillna(0.0)

    reasons = []
    for row in universe.itertuples(index=False):
        result = evaluate_filters(
            UniverseFilterInputs(
                market_cap=row.market_cap,
                price=row.price,
                adv_20d=row.adv_20d,
                history_days=row.history_days,
                completeness=row.completeness,
                sector=row.sector,
                security_type=row.security_type,
                price_staleness_days=(
                    None
                    if pd.isna(row.price_staleness_days)
                    else int(row.price_staleness_days)
                ),
            ),
            config,
        )
        reasons.append(",".join(result))
    universe["exclusion_reason"] = reasons
    universe["included"] = universe["exclusion_reason"].eq("")
    columns = ["date", "symbol", "included", "exclusion_reason", "market_cap", "price", "adv_20d", "sector"]
    if "security_id" in universe:
        columns.insert(1, "security_id")
    return universe[columns]


def build_and_store_universe(as_of_date: pd.Timestamp, config: AppConfig) -> pd.DataFrame:
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
    security_master = default_universe_provider(config).snapshot(as_of_date)
    if security_master.empty:
        security_master = read_parquet(
            Path(config.paths.silver_data) / "universe" / "security_master.parquet"
        )
    universe = build_universe(as_of_date, prices, fundamentals, security_master, config.universe)
    append_or_replace_parquet(
        universe,
        Path(config.paths.silver_data) / "universe" / "eligible_universe.parquet",
        ["date", "symbol"],
    )
    return universe
