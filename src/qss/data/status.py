from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from qss.config.schema import AppConfig
from qss.data.storage import read_parquet
from qss.data.validation import (
    monthly_membership_coverage,
    required_research_credentials,
)


@dataclass
class ResearchDataStatus:
    checks: pd.DataFrame
    ready: bool


def membership_symbols(
    config: AppConfig,
    start_date: str | None = None,
    end_date: str | None = None,
    latest_only: bool = False,
) -> list[str]:
    membership = read_parquet(
        Path(config.paths.silver_data) / "universe" / "universe_membership.parquet"
    )
    if membership.empty or "symbol" not in membership:
        return []
    frame = membership.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    if "included" in frame:
        frame = frame.loc[frame["included"]]
    if start_date:
        frame = frame.loc[frame["date"] >= pd.Timestamp(start_date)]
    if end_date:
        frame = frame.loc[frame["date"] <= pd.Timestamp(end_date)]
    if latest_only and not frame.empty:
        frame = frame.loc[frame["date"] == frame["date"].max()]
    return sorted(frame["symbol"].dropna().astype(str).unique().tolist())


def _status_row(
    component: str,
    status: str,
    progress: str,
    detail: str,
) -> dict[str, str]:
    return {
        "component": component,
        "status": status,
        "progress": progress,
        "detail": detail,
    }


def research_data_status(
    config: AppConfig,
    start_date: str | None = None,
    end_date: str | None = None,
) -> ResearchDataStatus:
    start = pd.Timestamp(start_date or config.backtest.start_date).normalize()
    end = pd.Timestamp(end_date or pd.Timestamp.today()).normalize()
    current_membership_mode = (
        config.universe.membership_mode == "current_snapshot"
    )
    sp500_history_mode = (
        config.universe.membership_mode == "point_in_time"
        and config.universe.long_history_provider == "sp500_wikipedia"
    )
    root = Path(config.paths.silver_data)
    membership = read_parquet(root / "universe" / "universe_membership.parquet")
    master = read_parquet(root / "universe" / "security_master.parquet")
    prices = read_parquet(root / "prices" / "prices_daily.parquet")
    fundamentals = read_parquet(
        root / "fundamentals" / "fundamental_observations.parquet"
    )
    if fundamentals.empty:
        fundamentals = read_parquet(
            root / "fundamentals" / "fundamentals_quarterly.parquet"
        )
    macro = read_parquet(root / "macro" / "macro_observations.parquet")
    validation = read_parquet(root / "universe" / "universe_validation.parquet")

    rows: list[dict[str, str]] = []
    historical_months = 0
    expected_months = len(pd.date_range(start, end, freq="ME"))
    universe_detail = f"0/{expected_months} months; missing all requested months"
    research_symbols: set[str] = set()
    if not membership.empty and {"date", "symbol"}.issubset(membership):
        membership = membership.copy()
        membership["date"] = pd.to_datetime(membership["date"]).dt.normalize()
        included = (
            membership.loc[membership["included"]]
            if "included" in membership
            else membership
        )
        universe_rows = included
        if "source" in included:
            expected_source = (
                "nasdaq_trader_current_backfill"
                if current_membership_mode
                else "sp500_wikipedia"
                if sp500_history_mode
                else "alpha_vantage"
            )
            universe_rows = included.loc[
                included["source"].astype(str).str.contains(expected_source)
            ]
        (
            historical_months,
            expected_months,
            universe_ratio,
            universe_detail,
        ) = monthly_membership_coverage(
            universe_rows["date"],
            start,
            end,
        )
        research_symbols = set(
            included.loc[
                (included["date"] >= start)
                & (included["date"] <= end),
                "symbol",
            ]
            .dropna()
            .astype(str)
        )
    else:
        universe_ratio = 0.0
    rows.append(
        _status_row(
            (
                "Current universe baseline"
                if current_membership_mode
                else "Universe history"
            ),
            "ready" if universe_ratio >= config.universe.min_long_price_coverage else (
                "partial" if historical_months else "missing"
            ),
            f"{historical_months}/{expected_months} months",
            (
                "Current Nasdaq membership is intentionally backfilled; "
                f"historical constituent changes are not modeled. {universe_detail}"
                if current_membership_mode
                else (
                    "S&P 500 monthly membership reconstructed from constituent "
                    f"changes. {universe_detail}"
                )
                if sp500_history_mode
                else f"Alpha Vantage monthly point-in-time snapshots. {universe_detail}"
            ),
        )
    )

    price_symbols = (
        set(prices["symbol"].dropna().astype(str))
        if not prices.empty and "symbol" in prices
        else set()
    )
    required_price_symbols = {
        *research_symbols,
        config.backtest.primary_benchmark,
        config.backtest.secondary_benchmark,
        config.strategy.benchmark,
    }
    price_coverage = (
        len(required_price_symbols & price_symbols) / len(required_price_symbols)
        if required_price_symbols
        else 0.0
    )
    rows.append(
        _status_row(
            "Research prices",
            "ready" if price_coverage >= config.universe.min_long_price_coverage else (
                "partial" if price_symbols else "missing"
            ),
            f"{price_coverage:.1%} symbol coverage",
            f"{len(price_symbols)} stored symbols; {len(required_price_symbols)} required.",
        )
    )

    fundamental_symbols = (
        set(fundamentals["symbol"].dropna().astype(str))
        if not fundamentals.empty and "symbol" in fundamentals
        else set()
    )
    fundamental_coverage = (
        len(research_symbols & fundamental_symbols) / len(research_symbols)
        if research_symbols
        else 0.0
    )
    rows.append(
        _status_row(
            "SEC fundamentals",
            "ready" if fundamental_coverage >= config.strategy.min_factor_coverage else (
                "partial" if fundamental_symbols else "missing"
            ),
            f"{fundamental_coverage:.1%} symbol coverage",
            f"{len(fundamental_symbols)} stored symbols.",
        )
    )

    expected_series = set(config.macro.fred_series.values())
    macro_series = (
        set(macro["series_id"].dropna().astype(str))
        if not macro.empty and "series_id" in macro
        else set()
    )
    rows.append(
        _status_row(
            "Macro observations",
            "ready" if expected_series.issubset(macro_series) else (
                "partial" if macro_series else "missing"
            ),
            f"{len(expected_series & macro_series)}/{len(expected_series)} series",
            "FRED API key is preferred; public graph CSV remains the fallback.",
        )
    )

    known_sectors = 0
    if not master.empty:
        sector_frame = master
        if research_symbols and "symbol" in master:
            sector_frame = master.loc[
                master["symbol"].astype(str).isin(research_symbols)
            ].drop_duplicates("symbol")
        sectors = sector_frame.get(
            "sector",
            pd.Series("Unknown", index=sector_frame.index),
        ).fillna("Unknown")
        known_sectors = int(
            (
                ~sectors.astype(str)
                .str.lower()
                .isin(["", "unknown", "unclassified"])
            ).sum()
        )
    sector_total = len(sector_frame) if not master.empty else 0
    sector_coverage = known_sectors / sector_total if sector_total else 0.0
    rows.append(
        _status_row(
            "Sector metadata",
            "ready" if sector_coverage >= config.universe.min_sector_coverage else (
                "partial" if known_sectors else "missing"
            ),
            f"{sector_coverage:.1%} coverage",
            f"{known_sectors}/{sector_total} research symbols classified.",
        )
    )

    expected_validation_months = len(
        pd.date_range(end - pd.DateOffset(years=config.universe.recent_validation_years), end, freq="ME")
    )
    validation_months = (
        pd.to_datetime(validation["date"]).dt.to_period("M").nunique()
        if not validation.empty and "date" in validation
        else 0
    )
    if current_membership_mode:
        rows.append(
            _status_row(
                "Survivorship-bias disclosure",
                "ready",
                "explicit",
                "Reports are labeled as current-membership backtests.",
            )
        )
    elif config.universe.validation_provider == "disabled":
        source_rows = (
            validation["source"].astype(str).str.contains("sp500_wikipedia").any()
            if not validation.empty and "source" in validation
            else False
        )
        rows.append(
            _status_row(
                "Universe source audit",
                "ready" if source_rows else "missing",
                "single-source" if source_rows else "missing",
                (
                    "S&P 500 membership uses Wikipedia constituents plus change "
                    "history; Alpha/Massive universe APIs are not required."
                ),
            )
        )
    else:
        rows.append(
            _status_row(
                "Cross-source validation",
                "ready"
                if validation_months >= expected_validation_months * 0.95
                else ("partial" if validation_months else "missing"),
                f"{validation_months}/{expected_validation_months} months",
                "Massive recent-universe comparison.",
            )
        )

    credentials = required_research_credentials(config)
    present_credentials = sum(credentials.values())
    rows.append(
        _status_row(
            "Provider credentials",
            "ready" if all(credentials.values()) else "partial",
            f"{present_credentials}/{len(credentials)} configured",
            ", ".join(name for name, present in credentials.items() if not present)
            or "All required credentials are present.",
        )
    )

    checks = pd.DataFrame(rows)
    return ResearchDataStatus(
        checks=checks,
        ready=bool((checks["status"] == "ready").all()),
    )
