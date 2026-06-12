from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from qss.config.schema import AppConfig
from qss.config.validation import validate_config
from qss.utils import project_root


def _deep_merge(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in incoming.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _resolve(path: str | Path, base_dir: Path | None = None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if base_dir is not None:
        candidate = (base_dir / candidate).resolve()
        if candidate.exists():
            return candidate
    return (project_root() / path).resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config file {path} must contain a mapping at the root.")
    return payload


def load_config(paths: list[str]) -> dict:
    merged: dict[str, Any] = {}
    seen: set[Path] = set()

    def _load_recursive(raw_path: str, parent_dir: Path | None = None) -> None:
        path = _resolve(raw_path, parent_dir)
        if path in seen:
            return
        seen.add(path)
        payload = _load_yaml(path)
        for include in payload.get("includes", []):
            _load_recursive(include, path.parent)
        non_include_payload = {k: v for k, v in payload.items() if k != "includes"}
        merged.update(_deep_merge(merged, non_include_payload))

    for raw in paths:
        _load_recursive(raw)
    return merged


def get_config(config_paths: list[str]) -> AppConfig:
    return validate_config(load_config(config_paths))
