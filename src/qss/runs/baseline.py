from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from qss.config.schema import AppConfig
from qss.data.storage import resolve_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_legacy_baseline(
    config: AppConfig,
    label: str = "legacy-demo-20260612",
) -> Path:
    candidates = [
        resolve_path(config.paths.silver_data) / "prices" / "prices_daily.parquet",
        resolve_path(config.paths.silver_data)
        / "fundamentals"
        / "fundamentals_quarterly.parquet",
        resolve_path(config.paths.silver_data)
        / "universe"
        / "universe_membership.parquet",
        resolve_path(config.paths.reports) / "backtest" / "backtest_metrics.csv",
        resolve_path(config.paths.reports) / "backtest" / "daily_returns.csv",
    ]
    artifacts = [
        {
            "path": str(path),
            "size": path.stat().st_size,
            "modified": pd.Timestamp(path.stat().st_mtime, unit="s").isoformat(),
            "sha256": _sha256(path),
        }
        for path in candidates
        if path.exists()
    ]
    payload = {
        "label": label,
        "status": "legacy-demo",
        "trusted_for_strategy_decisions": False,
        "reason": (
            "Current-membership seed universe, synthetic fallback rows, simplified accounting, "
            "and incomplete point-in-time coverage."
        ),
        "captured_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "artifacts": artifacts,
    }
    target = resolve_path(config.paths.reports) / "baselines" / f"{label}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target
