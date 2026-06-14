from __future__ import annotations

import io
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from qss.config.schema import AppConfig
from qss.data.quality import check_data_quality, write_quality_report
from qss.data.storage import (
    append_with_source_precedence,
    read_parquet,
    write_parquet,
    write_text,
)
from qss.logging_utils import logger


@dataclass
class MacroIngestionResult:
    macro: pd.DataFrame
    quality_report: pd.DataFrame


def _fetch_series(
    series_id: str,
    api_key: str | None = None,
    retry_delays: tuple[float, ...] = (1.0, 3.0),
) -> pd.DataFrame:
    if api_key:
        url = "https://api.stlouisfed.org/fred/series/observations"
        request_kwargs = {
            "params": {
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "observation_start": "2000-01-01",
            }
        }
    else:
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
        request_kwargs = {
            "params": {
                "id": series_id,
                "cosd": "2000-01-01",
            }
        }
    response = None
    for attempt in range(len(retry_delays) + 1):
        try:
            response = requests.get(url, timeout=30, **request_kwargs)
            response.raise_for_status()
            break
        except requests.RequestException:
            if attempt >= len(retry_delays):
                raise
            logger.warning(
                "FRED request for {} failed; retrying in {:.1f}s.",
                series_id,
                retry_delays[attempt],
            )
            time.sleep(retry_delays[attempt])
    if response is None:
        raise RuntimeError(f"FRED returned no response for {series_id}.")
    if api_key:
        frame = pd.DataFrame(response.json().get("observations", []))
        frame = frame.rename(columns={"value": "value"})
    else:
        frame = pd.read_csv(io.StringIO(response.text))
        frame = frame.rename(
            columns={"DATE": "date", "observation_date": "date", series_id: "value"}
        )
    if not {"date", "value"}.issubset(frame):
        raise ValueError(f"FRED returned an invalid payload for {series_id}.")
    frame["series_id"] = series_id
    frame["date"] = pd.to_datetime(frame["date"])
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["value"])
    frame["available_date"] = frame["date"]
    frame["source"] = "fred"
    frame["quality_status"] = "live"
    frame["ingestion_time"] = pd.Timestamp.now(tz="UTC").tz_localize(None)
    return frame[
        [
            "series_id",
            "date",
            "value",
            "available_date",
            "source",
            "quality_status",
            "ingestion_time",
        ]
    ]


def _synthetic_series(series_id: str) -> pd.DataFrame:
    dates = pd.date_range("2000-01-01", pd.Timestamp.today().normalize(), freq="MS")
    trend = np.linspace(0, 1, len(dates))
    value = {
        "CPIAUCSL": 180 + 140 * trend,
        "UNRATE": 6 - 1.2 * trend + 0.5 * np.sin(trend * 18),
        "FEDFUNDS": 1.5 + 2.0 * np.sin(trend * 8),
        "DGS10": 2.5 + 1.5 * np.sin(trend * 9),
        "DGS2": 2.0 + 1.6 * np.sin(trend * 9 + 0.7),
        "BAA10Y": 2.0 + 0.4 * np.sin(trend * 12),
    }.get(series_id, 100 * trend)
    return pd.DataFrame(
        {
            "series_id": series_id,
            "date": dates,
            "value": value,
            "available_date": dates,
            "source": "synthetic_fallback",
            "quality_status": "invalid_for_research",
            "ingestion_time": pd.Timestamp.now(tz="UTC").tz_localize(None),
        }
    )


def generate_synthetic_macro(config: AppConfig) -> pd.DataFrame:
    return pd.concat(
        [_synthetic_series(series_id) for series_id in config.macro.fred_series.values()],
        ignore_index=True,
    ).drop_duplicates(["series_id", "date"], keep="last")


def ingest_macro(config: AppConfig) -> MacroIngestionResult:
    frames = []
    failures: list[str] = []
    api_key_env_var = config.data_sources.macro.api_key_env_var
    api_key = os.getenv(api_key_env_var) if api_key_env_var else None
    cache_root = Path(config.paths.raw_data) / "macro" / "fred"
    for series_id in config.macro.fred_series.values():
        cache_path = cache_root / f"{series_id}.parquet"
        try:
            frame = _fetch_series(series_id, api_key=api_key)
            write_parquet(frame, cache_path)
            frames.append(frame)
        except Exception:
            cached = read_parquet(cache_path)
            if not cached.empty:
                logger.warning(
                    "FRED fetch failed for {}; using the last live cache.",
                    series_id,
                )
                frames.append(cached)
                continue
            failures.append(series_id)
            if config.runtime.research_mode and not config.runtime.allow_synthetic:
                continue
            logger.warning(
                "FRED fetch failed for {}. Using synthetic fallback.",
                series_id,
            )
            frames.append(_synthetic_series(series_id))
    if failures:
        raise RuntimeError(
            "Research mode requires live FRED data; no cache is available for: "
            + ", ".join(failures)
        ) from None
    macro = pd.concat(frames, ignore_index=True).drop_duplicates(["series_id", "date"], keep="last")
    timestamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    write_text(macro.to_csv(index=False), Path(config.paths.raw_data) / "macro" / f"macro_raw_{timestamp}.csv")
    append_with_source_precedence(
        macro,
        Path(config.paths.silver_data) / "macro" / "macro_observations.parquet",
        ["series_id", "date"],
        {"fred": 100, "synthetic_fallback": 0},
    )
    quality = check_data_quality("macro_observations", macro, ["series_id", "date"])
    write_quality_report(quality, Path(config.paths.reports) / "data_quality" / f"data_quality_{pd.Timestamp.today():%Y%m%d}.csv")
    return MacroIngestionResult(macro=macro, quality_report=quality)
