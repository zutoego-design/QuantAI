from __future__ import annotations

import hashlib
import json
from pathlib import Path

from qss.config.schema import AppConfig
from qss.data.storage import resolve_path


def research_input_paths(config: AppConfig) -> list[Path]:
    silver = resolve_path(config.paths.silver_data)
    candidates = [
        silver / "prices" / "prices_daily.parquet",
        silver / "universe" / "security_master.parquet",
        silver / "universe" / "universe_membership.parquet",
        silver / "events" / "sec_filings.parquet",
    ]
    observations = (
        silver / "fundamentals" / "fundamental_observations.parquet"
    )
    quarterly = silver / "fundamentals" / "fundamentals_quarterly.parquet"
    candidates.append(observations if observations.exists() else quarterly)
    return [path for path in candidates if path.exists()]


def _file_digest(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def build_data_snapshot(
    config: AppConfig,
    paths: list[Path] | None = None,
) -> dict:
    root = resolve_path(".").resolve()
    entries = []
    for path in sorted(paths or research_input_paths(config)):
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(root).as_posix()
        except ValueError:
            relative = str(resolved)
        entries.append(
            {
                "path": relative,
                "size": resolved.stat().st_size,
                "sha256": _file_digest(resolved),
            }
        )
    identity_payload = json.dumps(entries, sort_keys=True, separators=(",", ":"))
    return {
        "snapshot_id": hashlib.sha256(identity_payload.encode("utf-8")).hexdigest(),
        "algorithm": "sha256",
        "files": entries,
    }


def write_data_snapshot(snapshot: dict, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    return target
