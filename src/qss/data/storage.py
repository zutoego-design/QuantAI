from __future__ import annotations

from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd

from qss.config.schema import AppConfig
from qss.utils import ensure_parent, project_root


def resolve_path(relative_path: str | Path) -> Path:
    path = Path(relative_path)
    return path if path.is_absolute() else (project_root() / path)


def ensure_data_directories(config: AppConfig) -> None:
    path_values = [
        config.paths.raw_data,
        config.paths.silver_data,
        config.paths.gold_data,
        config.paths.reports,
    ]
    for raw in path_values:
        resolve_path(raw).mkdir(parents=True, exist_ok=True)


def write_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    target = ensure_parent(resolve_path(path))
    df.to_parquet(target, index=False)
    return target


def append_or_replace_parquet(df: pd.DataFrame, path: str | Path, dedupe_keys: Iterable[str]) -> Path:
    target = resolve_path(path)
    if target.exists():
        existing = pd.read_parquet(target)
        df = pd.concat([existing, df], ignore_index=True)
    df = df.drop_duplicates(list(dedupe_keys), keep="last")
    return write_parquet(df, target)


def append_with_source_precedence(
    df: pd.DataFrame,
    path: str | Path,
    dedupe_keys: Iterable[str],
    source_priority: dict[str, int],
) -> Path:
    target = resolve_path(path)
    if target.exists():
        df = pd.concat([pd.read_parquet(target), df], ignore_index=True)
    if "source" not in df:
        raise ValueError("Source precedence requires a source column.")
    df = df.copy()
    df["_source_priority"] = df["source"].map(source_priority).fillna(0)
    sort_columns = [*dedupe_keys, "_source_priority"]
    if "ingestion_time" in df:
        sort_columns.append("ingestion_time")
    df = (
        df.sort_values(sort_columns)
        .drop_duplicates(list(dedupe_keys), keep="last")
        .drop(columns="_source_priority")
    )
    return write_parquet(df, target)


def read_parquet(path: str | Path) -> pd.DataFrame:
    target = resolve_path(path)
    if not target.exists():
        return pd.DataFrame()
    return pd.read_parquet(target)


def write_csv(df: pd.DataFrame, path: str | Path) -> Path:
    target = ensure_parent(resolve_path(path))
    df.to_csv(target, index=False)
    return target


def write_text(content: str, path: str | Path) -> Path:
    target = ensure_parent(resolve_path(path))
    target.write_text(content, encoding="utf-8")
    return target


def query_parquet(sql: str, **bindings: str | Path) -> pd.DataFrame:
    conn = duckdb.connect()
    try:
        for name, path in bindings.items():
            conn.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{resolve_path(path)}')")
        return conn.execute(sql).fetch_df()
    finally:
        conn.close()
