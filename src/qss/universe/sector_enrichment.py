from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from qss.config.schema import AppConfig
from qss.data.storage import append_or_replace_parquet, read_parquet, write_parquet
from qss.ingestion.sec_edgar import _sic_to_sector
from qss.logging_utils import logger
from qss.universe.providers import MassiveTickerProvider

UNKNOWN_SECTORS = {"", "unknown", "unclassified", "nan", "none"}


@dataclass
class SectorEnrichmentResult:
    requested: int
    fetched: int
    classified: int
    known: int
    total: int
    coverage: float


def _known_sector_mask(frame: pd.DataFrame) -> pd.Series:
    sectors = frame.get("sector", pd.Series("Unknown", index=frame.index))
    return ~sectors.fillna("").astype(str).str.strip().str.lower().isin(UNKNOWN_SECTORS)


def enrich_sector_metadata(
    config: AppConfig,
    start_date: str,
    end_date: str,
    tickers: list[str] | None = None,
    provider: MassiveTickerProvider | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> SectorEnrichmentResult:
    root = Path(config.paths.silver_data) / "universe"
    membership = read_parquet(root / "universe_membership.parquet")
    master = read_parquet(root / "security_master.parquet")
    if membership.empty or master.empty:
        return SectorEnrichmentResult(0, 0, 0, 0, 0, 0.0)

    membership["date"] = pd.to_datetime(membership["date"]).dt.normalize()
    research = membership.loc[
        membership["included"].astype(bool)
        & membership["date"].between(
            pd.Timestamp(start_date),
            pd.Timestamp(end_date),
        )
    ].copy()
    if tickers:
        research = research.loc[research["symbol"].astype(str).isin(set(tickers))]
    research_symbols = set(research["symbol"].dropna().astype(str))
    sector_frame = master.loc[
        master["symbol"].astype(str).isin(research_symbols)
    ].drop_duplicates("symbol", keep="last")
    known_mask = _known_sector_mask(sector_frame)
    known = int(known_mask.sum())
    total = len(sector_frame)
    coverage = known / total if total else 0.0
    target_known = math.ceil(config.universe.min_sector_coverage * total)
    needed = max(target_known - known, 0)
    if not needed:
        return SectorEnrichmentResult(0, 0, 0, known, total, coverage)

    provider = provider or MassiveTickerProvider()
    if not provider.api_key:
        logger.warning(
            "Sector metadata remains below threshold, but MASSIVE_API_KEY is not configured."
        )
        return SectorEnrichmentResult(0, 0, 0, known, total, coverage)

    unknown_symbols = set(sector_frame.loc[~known_mask, "symbol"].astype(str))
    ranked = (
        research.loc[research["symbol"].astype(str).isin(unknown_symbols)]
        .groupby("symbol")
        .agg(
            membership_months=("date", "size"),
            last_membership_date=("date", "max"),
        )
        .sort_values(["membership_months", "last_membership_date"], ascending=False)
    )
    request_limit = min(config.universe.max_remote_requests_per_sync, len(ranked))
    rows: list[dict[str, object]] = []
    requested = 0
    fetched = 0
    for symbol, record in ranked.head(request_limit).iterrows():
        requested += 1
        details = provider.fetch_details(symbol, record["last_membership_date"])
        if details:
            fetched += 1
            sector = _sic_to_sector(details.get("sic"))
            if sector.lower() not in UNKNOWN_SECTORS:
                details["sector"] = sector
                rows.append(details)
                if len(rows) >= needed:
                    break
        if requested < request_limit:
            sleep_fn(config.universe.remote_request_interval_seconds)

    if rows:
        metadata = pd.DataFrame(rows).drop_duplicates("symbol", keep="last")
        append_or_replace_parquet(
            metadata,
            root / "security_metadata.parquet",
            ["symbol"],
        )
        enrichment = metadata[
            [
                "symbol",
                "cik",
                "sic",
                "sic_description",
                "sector",
                "metadata_source",
                "metadata_ingestion_time",
            ]
        ]
        master = master.merge(
            enrichment,
            on="symbol",
            how="left",
            suffixes=("", "_massive"),
        )
        for column in [
            "cik",
            "sic",
            "sic_description",
            "sector",
            "metadata_source",
            "metadata_ingestion_time",
        ]:
            incoming_column = f"{column}_massive"
            if incoming_column not in master:
                continue
            incoming = master[incoming_column]
            invalid = incoming.isna() | incoming.astype(str).str.strip().str.lower().isin(
                UNKNOWN_SECTORS
            )
            incoming = incoming.mask(invalid)
            if column in master:
                existing = master[column]
                if column == "sector":
                    existing_invalid = (
                        existing.isna()
                        | existing.astype(str).str.strip().str.lower().isin(
                            UNKNOWN_SECTORS
                        )
                    )
                    replacement = incoming.combine_first(existing)
                    master[column] = existing.where(~existing_invalid, replacement)
                else:
                    master[column] = existing.combine_first(incoming)
            else:
                master[column] = incoming
            master = master.drop(columns=incoming_column)
        write_parquet(master, root / "security_master.parquet")

    classified = len(rows)
    known += classified
    coverage = known / total if total else 0.0
    logger.info(
        "Massive sector enrichment complete: requested={}, fetched={}, "
        "classified={}, coverage={:.1%}.",
        requested,
        fetched,
        classified,
        coverage,
    )
    return SectorEnrichmentResult(
        requested,
        fetched,
        classified,
        known,
        total,
        coverage,
    )
