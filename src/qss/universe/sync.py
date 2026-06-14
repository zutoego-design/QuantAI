from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from qss.config.schema import AppConfig
from qss.data.storage import read_parquet, write_parquet
from qss.logging_utils import logger
from qss.universe.providers import (
    OPERATING_TYPES,
    AlphaVantageListingProvider,
    MassiveTickerProvider,
    NasdaqTraderProvider,
)
from qss.universe.sp500 import build_sp500_history


@dataclass
class UniverseSyncResult:
    security_master: pd.DataFrame
    symbol_history: pd.DataFrame
    listing_intervals: pd.DataFrame
    membership: pd.DataFrame
    validation: pd.DataFrame
    historical_months: int = 0
    requested_months: int = 0
    next_missing_date: str | None = None
    warning: str | None = None


def _month_ends(start: str, end: str) -> list[pd.Timestamp]:
    return list(pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="ME"))


def _eligible(frame: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    allowed = set(config.universe.allowed_security_types or OPERATING_TYPES)
    return frame.loc[
        frame["security_type"].isin(allowed)
        & frame["exchange"].astype(str).str.upper().isin(["NASDAQ", "XNAS"])
    ].copy()


def _tables_from_snapshots(snapshots: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = snapshots.sort_values(["security_id", "date", "symbol"])
    master = (
        ordered.groupby("security_id", as_index=False)
        .agg(
            name=("name", "last"),
            symbol=("symbol", "last"),
            exchange=("exchange", "last"),
            security_type=("security_type", "last"),
            first_seen=("date", "min"),
            last_seen=("date", "max"),
            source=("source", lambda values: ",".join(sorted(set(values)))),
        )
    )
    history = (
        ordered.groupby(["security_id", "symbol"], as_index=False)
        .agg(valid_from=("date", "min"), valid_to=("date", "max"), source=("source", "last"))
    )
    latest_snapshot = ordered["date"].max()
    open_symbols = set(
        ordered.loc[ordered["date"] == latest_snapshot, ["security_id", "symbol"]]
        .itertuples(index=False, name=None)
    )
    open_mask = [
        (security_id, symbol) in open_symbols
        for security_id, symbol in history[["security_id", "symbol"]].itertuples(
            index=False, name=None
        )
    ]
    history.loc[open_mask, "valid_to"] = pd.NaT
    intervals = (
        ordered.groupby(["security_id", "symbol"], as_index=False)
        .agg(valid_from=("date", "min"), valid_to=("date", "max"))
    )
    if "delisting_date" in ordered:
        explicit_delistings = (
            ordered.assign(
                delisting_date=pd.to_datetime(
                    ordered["delisting_date"], errors="coerce"
                )
            )
            .groupby(["security_id", "symbol"], as_index=False)["delisting_date"]
            .max()
        )
        intervals = intervals.merge(
            explicit_delistings, on=["security_id", "symbol"], how="left"
        )
    else:
        intervals["delisting_date"] = pd.NaT
    intervals["exchange"] = "XNAS"
    intervals["interval_quality"] = "snapshot_inferred"
    return master, history, intervals


def _validation_rows(
    internal: pd.DataFrame,
    external: pd.DataFrame,
    as_of: pd.Timestamp,
) -> dict:
    left = set(internal["symbol"])
    right = set(external["symbol"])
    union = left | right
    return {
        "date": as_of,
        "internal_count": len(left),
        "external_count": len(right),
        "intersection_count": len(left & right),
        "jaccard": len(left & right) / len(union) if union else 1.0,
        "missing_internal": len(right - left),
        "missing_external": len(left - right),
    }


def _sync_current_snapshot(
    config: AppConfig,
    start: str,
    end: str,
) -> UniverseSyncResult:
    requested_dates = _month_ends(start, end)
    if not requested_dates:
        requested_dates = [pd.Timestamp(end).normalize()]
    current = _eligible(NasdaqTraderProvider().fetch(), config)
    if current.empty:
        raise RuntimeError("Nasdaq Trader returned an empty current universe.")

    raw_root = Path(config.paths.raw_data) / "universe" / "nasdaq_trader"
    write_parquet(
        current,
        raw_root / f"{pd.Timestamp.today():%Y-%m-%d}.parquet",
    )
    snapshots = []
    for date in requested_dates:
        snapshot = current.copy()
        snapshot["date"] = date.to_period("M").start_time
        snapshot["source"] = "nasdaq_trader_current_backfill"
        snapshots.append(snapshot)
    combined = pd.concat(snapshots, ignore_index=True)
    membership = combined[
        ["date", "security_id", "symbol", "security_type", "source"]
    ].copy()
    membership["included"] = True
    membership["exclusion_reason"] = ""
    master, history, intervals = _tables_from_snapshots(combined)

    root = Path(config.paths.silver_data) / "universe"
    existing_master = read_parquet(root / "security_master.parquet")
    if not existing_master.empty and "symbol" in existing_master:
        enrichment_columns = [
            column
            for column in ["symbol", "sector", "sic", "sic_description"]
            if column in existing_master
        ]
        if len(enrichment_columns) > 1:
            enrichment = existing_master[enrichment_columns].drop_duplicates("symbol")
            master = master.merge(enrichment, on="symbol", how="left")

    validation = pd.DataFrame(
        columns=[
            "date",
            "internal_count",
            "external_count",
            "intersection_count",
            "jaccard",
            "missing_internal",
            "missing_external",
        ]
    )
    write_parquet(master, root / "security_master.parquet")
    write_parquet(history, root / "symbol_history.parquet")
    write_parquet(intervals, root / "listing_intervals.parquet")
    write_parquet(membership, root / "universe_membership.parquet")
    write_parquet(validation, root / "universe_validation.parquet")
    for year, group in membership.groupby(membership["date"].dt.year):
        write_parquet(
            group,
            root / "membership_by_year" / f"year={year}" / "part.parquet",
        )
    return UniverseSyncResult(
        master,
        history,
        intervals,
        membership,
        validation,
        historical_months=len(requested_dates),
        requested_months=len(requested_dates),
        warning=(
            "Current Nasdaq membership was backfilled across the research window. "
            "The resulting backtest has survivorship bias."
        ),
    )


def _sync_sp500_history(
    config: AppConfig,
    start: str,
    end: str,
) -> UniverseSyncResult:
    raw_root = Path(config.paths.raw_data) / "universe" / "sp500_wikipedia"
    history = build_sp500_history(start, end, raw_root=raw_root)
    root = Path(config.paths.silver_data) / "universe"

    existing_master = read_parquet(root / "security_master.parquet")
    master = history.security_master
    if not existing_master.empty and "symbol" in existing_master:
        enrichment_columns = [
            column
            for column in ["symbol", "sector", "sic", "sic_description"]
            if column in existing_master
        ]
        if len(enrichment_columns) > 1:
            enrichment = existing_master[enrichment_columns].drop_duplicates("symbol")
            master = master.merge(enrichment, on="symbol", how="left", suffixes=("", "_old"))
            for column in ["sector", "sic", "sic_description"]:
                old_column = f"{column}_old"
                if old_column in master:
                    if column in master:
                        master[column] = master[column].where(
                            ~master[column].astype(str).str.lower().isin(
                                ["", "unknown", "unclassified", "nan"]
                            ),
                            master[old_column],
                        )
                    else:
                        master[column] = master[old_column]
                    master = master.drop(columns=old_column)

    validation = pd.DataFrame(
        [
            {
                "date": pd.Timestamp(end).normalize(),
                "internal_count": int(
                    history.membership.loc[
                        history.membership["date"]
                        == history.membership["date"].max(),
                        "symbol",
                    ].nunique()
                ),
                "external_count": pd.NA,
                "intersection_count": pd.NA,
                "jaccard": pd.NA,
                "missing_internal": pd.NA,
                "missing_external": pd.NA,
                "source": "sp500_wikipedia_single_source_audit",
            }
        ]
    )
    write_parquet(master, root / "security_master.parquet")
    write_parquet(history.symbol_history, root / "symbol_history.parquet")
    write_parquet(history.listing_intervals, root / "listing_intervals.parquet")
    write_parquet(history.membership, root / "universe_membership.parquet")
    write_parquet(validation, root / "universe_validation.parquet")
    for year, group in history.membership.groupby(history.membership["date"].dt.year):
        write_parquet(
            group,
            root / "membership_by_year" / f"year={year}" / "part.parquet",
        )
    months = history.membership["date"].dt.to_period("M").nunique()
    return UniverseSyncResult(
        master,
        history.symbol_history,
        history.listing_intervals,
        history.membership,
        validation,
        historical_months=months,
        requested_months=months,
        warning=(
            "S&P 500 membership was reconstructed from Wikipedia current "
            "constituents and constituent-change history."
        ),
    )


def sync_universe(
    config: AppConfig,
    start_date: str | None = None,
    end_date: str | None = None,
    validate_recent: bool = True,
) -> UniverseSyncResult:
    start = start_date or config.universe.start_date
    end = end_date or str(pd.Timestamp.today().date())
    if config.universe.membership_mode == "current_snapshot":
        return _sync_current_snapshot(config, start, end)
    if config.universe.long_history_provider == "sp500_wikipedia":
        return _sync_sp500_history(config, start, end)
    alpha = AlphaVantageListingProvider()
    snapshots: list[pd.DataFrame] = []
    raw_root = Path(config.paths.raw_data) / "universe"
    alpha_cache = raw_root / "alpha_vantage"
    massive_cache = raw_root / "massive"
    request_budget = config.universe.max_remote_requests_per_sync
    alpha_requests = 0
    massive_requests = 0
    requested_dates = _month_ends(start, end)
    historical_dates: set[pd.Timestamp] = set()
    next_missing_date: str | None = None
    sync_warning: str | None = None
    for date in requested_dates:
        cache_path = alpha_cache / f"{date:%Y-%m-%d}.parquet"
        cached = read_parquet(cache_path)
        if not cached.empty:
            snapshots.append(_eligible(cached, config))
            historical_dates.add(date)
            continue
        if alpha_requests >= request_budget:
            next_missing_date = str(date.date())
            sync_warning = (
                f"Remote request budget reached; rerun to continue from {next_missing_date}."
            )
            break
        try:
            fetched = _eligible(alpha.fetch(date), config)
        except (RuntimeError, ValueError) as exc:
            next_missing_date = str(date.date())
            sync_warning = (
                f"Alpha Vantage paused at {next_missing_date}: {exc} "
                "Cached progress was preserved; rerun later to continue."
            )
            logger.warning(sync_warning)
            break
        write_parquet(fetched, cache_path)
        snapshots.append(fetched)
        historical_dates.add(date)
        alpha_requests += 1
        if alpha_requests < request_budget:
            time.sleep(config.universe.remote_request_interval_seconds)

    existing_membership = read_parquet(
        Path(config.paths.silver_data) / "universe" / "universe_membership.parquet"
    )
    if (
        not existing_membership.empty
        and {"security_id", "source"}.issubset(existing_membership.columns)
    ):
        historical = existing_membership.loc[
            existing_membership["source"].astype(str).str.contains(
                "alpha_vantage", regex=True
            )
        ].copy()
        if "date" in historical:
            historical["date"] = pd.to_datetime(historical["date"]).dt.normalize()
            historical = historical.loc[~historical["date"].isin(historical_dates)]
        if not historical.empty:
            existing_master = read_parquet(
                Path(config.paths.silver_data) / "universe" / "security_master.parquet"
            )
            historical = historical.merge(
                existing_master[
                    [
                        column
                        for column in [
                            "security_id",
                            "name",
                            "exchange",
                            "security_type",
                        ]
                        if column in existing_master
                    ]
                ].drop_duplicates("security_id"),
                on="security_id",
                how="left",
                suffixes=("", "_master"),
            )
            snapshots.append(historical)

    current = _eligible(NasdaqTraderProvider().fetch(), config)
    if not current.empty:
        snapshots.append(current)
    combined = pd.concat(snapshots, ignore_index=True).drop_duplicates(
        ["date", "security_id", "symbol", "source"],
        keep="last",
    )
    combined["date"] = pd.to_datetime(combined["date"]).dt.normalize()
    membership = combined[
        ["date", "security_id", "symbol", "security_type", "source"]
    ].copy()
    membership["included"] = True
    membership["exclusion_reason"] = ""
    master, history, intervals = _tables_from_snapshots(combined)

    validation_rows: list[dict] = []
    if validate_recent:
        massive = MassiveTickerProvider()
        validation_start = pd.Timestamp(end) - pd.DateOffset(years=config.universe.recent_validation_years)
        for date in _month_ends(str(validation_start.date()), end):
            internal = membership.loc[membership["date"] == date]
            if internal.empty:
                continue
            cache_path = massive_cache / f"{date:%Y-%m-%d}.parquet"
            external = read_parquet(cache_path)
            if external.empty:
                if massive_requests >= request_budget:
                    break
                external = _eligible(massive.fetch(date), config)
                write_parquet(external, cache_path)
                massive_requests += 1
            validation_rows.append(_validation_rows(internal, external, date))
    validation = pd.DataFrame(validation_rows)

    root = Path(config.paths.silver_data) / "universe"
    existing_master = read_parquet(root / "security_master.parquet")
    if not existing_master.empty and "symbol" in existing_master:
        enrichment_columns = [
            column
            for column in ["symbol", "sector", "sic", "sic_description"]
            if column in existing_master
        ]
        if len(enrichment_columns) > 1:
            enrichment = existing_master[enrichment_columns].drop_duplicates("symbol")
            master = master.merge(enrichment, on="symbol", how="left")
    write_parquet(master, root / "security_master.parquet")
    write_parquet(history, root / "symbol_history.parquet")
    write_parquet(intervals, root / "listing_intervals.parquet")
    write_parquet(membership, root / "universe_membership.parquet")
    write_parquet(validation, root / "universe_validation.parquet")

    for year, group in membership.groupby(membership["date"].dt.year):
        write_parquet(group, root / "membership_by_year" / f"year={year}" / "part.parquet")
    return UniverseSyncResult(
        master,
        history,
        intervals,
        membership,
        validation,
        historical_months=len(historical_dates),
        requested_months=len(requested_dates),
        next_missing_date=next_missing_date,
        warning=sync_warning,
    )


def universe_coverage_report(config: AppConfig, prices: pd.DataFrame | None = None) -> pd.DataFrame:
    root = Path(config.paths.silver_data) / "universe"
    membership = read_parquet(root / "universe_membership.parquet")
    validation = read_parquet(root / "universe_validation.parquet")
    if prices is None:
        prices = read_parquet(Path(config.paths.silver_data) / "prices" / "prices_daily.parquet")
    if membership.empty:
        return pd.DataFrame(
            [{"gate": "universe_membership", "value": 0.0, "threshold": 1.0, "passed": False}]
        )
    membership["date"] = pd.to_datetime(membership["date"]).dt.normalize()
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    rows: list[dict] = []
    for date, members in membership.loc[membership["included"]].groupby("date"):
        symbols = set(members["symbol"])
        available = set(
            prices.loc[
                (prices["date"] <= date)
                & (prices["date"] >= date - pd.Timedelta(days=7)),
                "symbol",
            ]
        )
        coverage = len(symbols & available) / len(symbols) if symbols else 0.0
        recent = date >= membership["date"].max() - pd.DateOffset(
            years=config.universe.recent_validation_years
        )
        threshold = (
            config.universe.min_recent_price_coverage
            if recent
            else config.universe.min_long_price_coverage
        )
        rows.append(
            {
                "date": date,
                "gate": "price_coverage",
                "value": coverage,
                "threshold": threshold,
                "passed": coverage >= threshold,
            }
        )
    if not validation.empty:
        for row in validation.itertuples(index=False):
            rows.append(
                {
                    "date": row.date,
                    "gate": "recent_jaccard",
                    "value": float(row.jaccard),
                    "threshold": config.universe.min_recent_jaccard,
                    "passed": float(row.jaccard) >= config.universe.min_recent_jaccard,
                }
            )
    return pd.DataFrame(rows)
