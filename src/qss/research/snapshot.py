from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
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
        silver / "macro" / "macro_observations.parquet",
    ]
    observations = (
        silver / "fundamentals" / "fundamental_observations.parquet"
    )
    quarterly = silver / "fundamentals" / "fundamentals_quarterly.parquet"
    candidates.append(observations if observations.exists() else quarterly)
    style_cache = resolve_path(config.research_validation.style_factor_cache)
    if style_cache.is_dir():
        candidates.extend(path for path in style_cache.rglob("*") if path.is_file())
    elif style_cache.exists():
        candidates.append(style_cache)
    return sorted({path.resolve() for path in candidates if path.exists()})


def _file_digest(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_input(path: Path, digest: str, config: AppConfig) -> Path:
    archive_root = (
        resolve_path(config.paths.raw_data).parent
        / "archive"
        / "research_inputs"
    )
    target = archive_root / digest[:2] / f"{digest}{path.suffix.lower()}"
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    try:
        os.link(path, temporary)
    except OSError:
        shutil.copy2(path, temporary)
    temporary.replace(target)
    return target


def dependency_environment() -> dict:
    packages = sorted(
        (
            {
                "name": distribution.metadata.get("Name", distribution.name),
                "version": distribution.version,
            }
            for distribution in importlib.metadata.distributions()
        ),
        key=lambda item: (str(item["name"]).lower(), str(item["version"])),
    )
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
    }


def snapshot_identity_payload(snapshot: dict) -> str:
    identity = {
        "files": snapshot.get("files", []),
        "environment": snapshot.get("environment", {}),
    }
    return json.dumps(identity, sort_keys=True, separators=(",", ":"))


def build_data_snapshot(
    config: AppConfig,
    paths: list[Path] | None = None,
) -> dict:
    root = resolve_path(".").resolve()
    entries = []
    for path in sorted(paths or research_input_paths(config)):
        resolved = path.resolve()
        digest = _file_digest(resolved)
        archived = _archive_input(resolved, digest, config).resolve()
        try:
            relative = resolved.relative_to(root).as_posix()
        except ValueError:
            relative = str(resolved)
        try:
            archive_relative = archived.relative_to(root).as_posix()
        except ValueError:
            archive_relative = str(archived)
        entries.append(
            {
                "path": relative,
                "archive_path": archive_relative,
                "size": resolved.stat().st_size,
                "sha256": digest,
            }
        )
    snapshot = {
        "schema_version": "2.0",
        "algorithm": "sha256",
        "files": entries,
        "environment": dependency_environment(),
    }
    snapshot["snapshot_id"] = hashlib.sha256(
        snapshot_identity_payload(snapshot).encode("utf-8")
    ).hexdigest()
    return snapshot


def write_data_snapshot(snapshot: dict, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    return target
