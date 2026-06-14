from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

from qss.data.storage import resolve_path, write_parquet

BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
FIVE_FACTOR_URL = f"{BASE_URL}/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
MOMENTUM_URL = f"{BASE_URL}/F-F_Momentum_Factor_daily_CSV.zip"


def _download_csv(url: str) -> str:
    last_error: requests.RequestException | None = None
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            break
        except requests.RequestException as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(attempt + 1)
    else:
        assert last_error is not None
        raise last_error
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError(f"No CSV file found in {url}.")
        return archive.read(csv_names[0]).decode("utf-8", errors="replace")


def _parse_daily_factors(text: str) -> pd.DataFrame:
    rows = []
    header: list[str] | None = None
    for raw_line in text.splitlines():
        parts = [part.strip() for part in raw_line.split(",")]
        if not parts:
            continue
        first = parts[0]
        if header is None and first == "" and len(parts) > 1:
            header = ["date", *parts[1:]]
            continue
        if header is None or not first.isdigit() or len(first) != 8:
            continue
        rows.append(parts[: len(header)])
    if header is None or not rows:
        raise ValueError("Fama-French daily factor file has no readable data table.")
    frame = pd.DataFrame(rows, columns=header)
    frame["date"] = pd.to_datetime(frame["date"], format="%Y%m%d")
    for column in frame.columns:
        if column != "date":
            frame[column] = pd.to_numeric(frame[column], errors="coerce") / 100.0
    return frame.dropna(subset=["date"]).reset_index(drop=True)


def load_fama_french_daily(
    cache_directory: str | Path,
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    root = resolve_path(cache_directory)
    target = root / "ff5_momentum_daily.parquet"
    if target.exists() and not refresh:
        return pd.read_parquet(target)
    five = _parse_daily_factors(_download_csv(FIVE_FACTOR_URL))
    momentum = _parse_daily_factors(_download_csv(MOMENTUM_URL))
    momentum_column = next(
        column for column in momentum.columns if column != "date"
    )
    momentum = momentum.rename(columns={momentum_column: "Mom"})
    frame = five.merge(momentum[["date", "Mom"]], on="date", how="inner")
    frame = frame.rename(columns={"Mkt-RF": "Mkt-RF"})
    required = ["date", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF", "Mom"]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Fama-French factor download is missing columns: {sorted(missing)}")
    write_parquet(frame[required], target)
    return frame[required]
