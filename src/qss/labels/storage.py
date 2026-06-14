from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from qss.config.schema import AppConfig
from qss.data.storage import resolve_path, write_parquet
from qss.labels.schema import LabelDefinition


def persist_labels(
    labels: pd.DataFrame,
    definition: LabelDefinition,
    config: AppConfig,
) -> dict[str, Path]:
    root = resolve_path(config.paths.gold_data) / "labels"
    versioned = root / definition.name / definition.version / "labels.parquet"
    latest = root / f"{definition.name}.parquet"
    write_parquet(labels, versioned)
    write_parquet(labels, latest)
    metadata = root / definition.name / definition.version / "label_config.json"
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(definition.model_dump_json(indent=2), encoding="utf-8")
    return {"versioned": versioned, "latest": latest, "config": metadata}


def write_run_label_config(
    definitions: list[LabelDefinition],
    run_path: str | Path,
) -> Path:
    target = Path(run_path) / "label_config.json"
    target.write_text(
        json.dumps([item.model_dump(mode="json") for item in definitions], indent=2),
        encoding="utf-8",
    )
    return target
