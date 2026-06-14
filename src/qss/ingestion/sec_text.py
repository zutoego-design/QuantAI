from __future__ import annotations

import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable

import pandas as pd
import requests

from qss.config.schema import AppConfig
from qss.data.storage import append_or_replace_parquet, read_parquet, resolve_path


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        value = " ".join(data.split())
        if value:
            self.parts.append(value)


def normalize_filing_text(content: str) -> str:
    parser = _TextExtractor()
    parser.feed(content)
    return "\n".join(parser.parts)


def filing_document_url(row) -> str:
    cik = str(int(str(row.cik)))
    accession = str(row.accession).replace("-", "")
    return (
        "https://www.sec.gov/Archives/edgar/data/"
        f"{cik}/{accession}/{row.primary_document}"
    )


def cache_sec_filing_text(
    config: AppConfig,
    filings: pd.DataFrame | None = None,
    *,
    max_filings: int = 100,
    as_of_date: str | pd.Timestamp | None = None,
    fetcher: Callable[..., object] = requests.get,
) -> pd.DataFrame:
    if filings is None:
        filings = read_parquet(
            Path(config.paths.silver_data) / "events" / "sec_filings.parquet"
        )
    required = {
        "symbol",
        "cik",
        "filing_type",
        "filing_timestamp",
        "accession",
        "primary_document",
        "text_cache_key",
    }
    if filings.empty or not required.issubset(filings.columns):
        return pd.DataFrame()
    cache_root = resolve_path(config.text_factors.cache_directory)
    cache_root.mkdir(parents=True, exist_ok=True)
    user_agent = config.data_sources.fundamentals.user_agent
    rows = []
    pending = filings.loc[
        filings["filing_type"].astype(str).isin(config.text_factors.filing_types)
    ].copy()
    if as_of_date is not None:
        pending = pending.loc[
            pd.to_datetime(pending["filing_timestamp"]).dt.tz_localize(None)
            <= pd.Timestamp(as_of_date).tz_localize(None)
        ]
    pending = pending.sort_values("filing_timestamp", ascending=False).head(max_filings)
    for row in pending.itertuples(index=False):
        target = cache_root / f"{row.text_cache_key}.txt"
        error = ""
        if not target.exists():
            for attempt in range(3):
                try:
                    response = fetcher(
                        filing_document_url(row),
                        headers={
                            "User-Agent": user_agent,
                            "Accept-Encoding": "gzip, deflate",
                        },
                        timeout=60,
                    )
                    response.raise_for_status()
                    target.write_text(
                        normalize_filing_text(response.text),
                        encoding="utf-8",
                    )
                    error = ""
                    break
                except requests.RequestException as exc:
                    error = str(exc)
                    if attempt < 2:
                        time.sleep(0.5 * (attempt + 1))
        rows.append(
            {
                "symbol": row.symbol,
                "accession": row.accession,
                "text_cache_key": row.text_cache_key,
                "text_cached": target.exists(),
                "text_cache_path": str(target) if target.exists() else None,
                "text_cached_at": pd.Timestamp.now(tz="UTC").tz_localize(None),
                "text_cache_error": error,
            }
        )
    cached = pd.DataFrame(rows)
    if cached.empty:
        return cached
    updated = filings.merge(
        cached,
        on=["symbol", "accession", "text_cache_key"],
        how="left",
        suffixes=("", "_new"),
    )
    for column in [
        "text_cached",
        "text_cache_path",
        "text_cached_at",
        "text_cache_error",
    ]:
        new_column = f"{column}_new"
        if new_column in updated:
            if column in updated:
                updated[column] = updated[new_column].combine_first(updated[column])
            else:
                updated[column] = updated[new_column]
            updated = updated.drop(columns=new_column)
    append_or_replace_parquet(
        updated,
        Path(config.paths.silver_data) / "events" / "sec_filings.parquet",
        ["symbol", "accession"],
    )
    return cached
