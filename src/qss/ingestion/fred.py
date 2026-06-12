from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from qss.config.schema import AppConfig
from qss.data.quality import check_data_quality, write_quality_report
from qss.data.storage import append_with_source_precedence, write_text
from qss.logging_utils import logger


@dataclass
class MacroIngestionResult:
    macro: pd.DataFrame
    quality_report: pd.DataFrame


def _fetch_series(series_id: str) -> pd.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd=2000-01-01"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    frame = pd.read_csv(io.StringIO(response.text))
    frame = frame.rename(columns={"DATE": "date", "observation_date": "date", series_id: "value"})
    frame["series_id"] = series_id
    frame["date"] = pd.to_datetime(frame["date"])
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["value"])
    frame["available_date"] = frame["date"]
    frame["source"] = "fred"
    frame["quality_status"] = "live"
    frame["ingestion_time"] = pd.Timestamp.utcnow().tz_localize(None)
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
            "ingestion_time": pd.Timestamp.utcnow().tz_localize(None),
        }
    )


def ingest_macro(config: AppConfig) -> MacroIngestionResult:
    frames = []
    for series_id in config.macro.fred_series.values():
        try:
            frames.append(_fetch_series(series_id))
        except Exception as exc:
            if config.runtime.research_mode and not config.runtime.allow_synthetic:
                raise RuntimeError(
                    f"Research mode forbids synthetic macro data; FRED fetch failed for {series_id}."
                ) from exc
            logger.warning(
                "FRED fetch failed for {}: {}. Using synthetic fallback.", series_id, exc
            )
            frames.append(_synthetic_series(series_id))
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
