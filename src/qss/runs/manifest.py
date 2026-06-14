from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from qss.config.schema import AppConfig
from qss.data.storage import resolve_path

REPORT_SCHEMA_VERSION = "1.0"


def _json_default(value: Any) -> str:
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def config_hash(config: AppConfig) -> str:
    payload = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def code_version() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unversioned-workspace"


@dataclass
class RunManifest:
    run_id: str
    run_type: str
    status: str
    created_at: str
    data_cutoff: str | None
    config_hash: str
    config: dict[str, Any]
    code_version: str
    python_version: str
    report_schema_version: str = REPORT_SCHEMA_VERSION
    data_sources: dict[str, Any] = field(default_factory=dict)
    quality_gates: dict[str, Any] = field(default_factory=dict)
    bias_flags: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    research_protocol: dict[str, Any] | None = None
    spec_hash: str | None = None
    data_snapshot_id: str | None = None
    trial_number: int | None = None
    evidence_status: str | None = None


@dataclass
class RunContext:
    root: Path
    manifest: RunManifest

    def path(self, *parts: str) -> Path:
        target = self.root.joinpath(*parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def update(
        self,
        *,
        status: str | None = None,
        quality_gates: dict[str, Any] | None = None,
        bias_flags: list[str] | None = None,
        notes: list[str] | None = None,
        research_protocol: dict[str, Any] | None = None,
        spec_hash: str | None = None,
        data_snapshot_id: str | None = None,
        trial_number: int | None = None,
        evidence_status: str | None = None,
    ) -> None:
        if status is not None:
            self.manifest.status = status
        if quality_gates:
            self.manifest.quality_gates.update(quality_gates)
        if bias_flags:
            self.manifest.bias_flags = sorted(set([*self.manifest.bias_flags, *bias_flags]))
        if notes:
            self.manifest.notes.extend(notes)
        if research_protocol is not None:
            self.manifest.research_protocol = research_protocol
        if spec_hash is not None:
            self.manifest.spec_hash = spec_hash
        if data_snapshot_id is not None:
            self.manifest.data_snapshot_id = data_snapshot_id
        if trial_number is not None:
            self.manifest.trial_number = trial_number
        if evidence_status is not None:
            self.manifest.evidence_status = evidence_status
        self.write_manifest()

    def write_manifest(self) -> Path:
        target = self.path("manifest.json")
        target.write_text(
            json.dumps(asdict(self.manifest), indent=2, sort_keys=True, default=_json_default),
            encoding="utf-8",
        )
        return target


def create_run_context(
    config: AppConfig,
    run_type: str,
    data_cutoff: str | pd.Timestamp | None = None,
    run_id: str | None = None,
) -> RunContext:
    now = pd.Timestamp.now(tz="UTC")
    timestamp = now.strftime("%Y%m%dT%H%M%SZ")
    identifier = run_id or f"{timestamp}-{run_type}-{uuid4().hex[:8]}"
    root = resolve_path(config.paths.reports) / "runs" / identifier
    root.mkdir(parents=True, exist_ok=False)
    manifest = RunManifest(
        run_id=identifier,
        run_type=run_type,
        status="running",
        created_at=now.isoformat(),
        data_cutoff=str(pd.Timestamp(data_cutoff).date()) if data_cutoff is not None else None,
        config_hash=config_hash(config),
        config=config.model_dump(mode="json"),
        code_version=code_version(),
        python_version=platform.python_version(),
        data_sources=config.data_sources.model_dump(mode="json"),
    )
    context = RunContext(root=root, manifest=manifest)
    context.write_manifest()
    return context
