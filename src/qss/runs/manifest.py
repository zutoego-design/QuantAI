from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from qss.config.schema import AppConfig
from qss.data.storage import resolve_path

REPORT_SCHEMA_VERSION = "1.1"
RESEARCH_IDENTITY_PATHS = [
    "src",
    "qss",
    "tests",
    "configs",
    "experiments",
    "docs",
    "notebooks",
    ".github",
    "pyproject.toml",
    "uv.lock",
    "README.md",
    "sitecustomize.py",
]


def _json_default(value: Any) -> str:
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def config_hash(config: AppConfig) -> str:
    payload = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _git_command(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=resolve_path("."),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=10,
    )


def _git_bytes_command(*args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args],
        cwd=resolve_path("."),
        capture_output=True,
        text=False,
        check=False,
        timeout=10,
    )


def workspace_identity() -> dict[str, Any]:
    try:
        head = _git_command("rev-parse", "HEAD")
        status = _git_command(
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            *RESEARCH_IDENTITY_PATHS,
        )
        if head.returncode == 0 and status.returncode == 0:
            patch = _git_bytes_command(
                "diff",
                "--binary",
                "HEAD",
                "--",
                *RESEARCH_IDENTITY_PATHS,
            )
            untracked = _git_command(
                "ls-files",
                "--others",
                "--exclude-standard",
                "--",
                *RESEARCH_IDENTITY_PATHS,
            )
            untracked_paths = [
                line.strip()
                for line in untracked.stdout.splitlines()
                if line.strip()
            ]
            digest = hashlib.sha256()
            digest.update(patch.stdout)
            root = resolve_path(".").resolve()
            untracked_files = []
            for relative in sorted(untracked_paths):
                source = (root / relative).resolve()
                if not source.is_file():
                    continue
                content_digest = hashlib.sha256(source.read_bytes()).hexdigest()
                digest.update(relative.encode("utf-8"))
                digest.update(content_digest.encode("ascii"))
                untracked_files.append(
                    {
                        "path": relative.replace("\\", "/"),
                        "sha256": content_digest,
                        "size": source.stat().st_size,
                    }
                )
            dirty = bool(status.stdout.strip())
            workspace_hash = digest.hexdigest() if dirty else None
            commit = head.stdout.strip()
            return {
                "version": (
                    f"git:{commit}:dirty:{workspace_hash}"
                    if dirty
                    else f"git:{commit}:clean"
                ),
                "commit": commit,
                "dirty": dirty,
                "patch": patch.stdout,
                "patch_sha256": (
                    hashlib.sha256(patch.stdout).hexdigest()
                    if patch.stdout
                    else None
                ),
                "workspace_sha256": workspace_hash,
                "untracked_files": untracked_files,
            }
    except (OSError, subprocess.SubprocessError):
        pass
    return {
        "version": "unversioned-workspace",
        "commit": None,
        "dirty": True,
        "patch": "",
        "patch_sha256": None,
        "workspace_sha256": None,
        "untracked_files": [],
    }


def code_version() -> str:
    return str(workspace_identity()["version"])


def _environment_snapshot() -> dict[str, Any]:
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
    code_commit: str | None
    code_dirty: bool
    code_patch_sha256: str | None
    workspace_sha256: str | None
    python_version: str
    environment_sha256: str
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
    identity = workspace_identity()
    environment = _environment_snapshot()
    environment_payload = json.dumps(
        environment,
        sort_keys=True,
        separators=(",", ":"),
    )
    (root / "environment.json").write_text(
        json.dumps(environment, indent=2),
        encoding="utf-8",
    )
    workspace_payload = {
        key: value
        for key, value in identity.items()
        if key != "patch"
    }
    (root / "workspace_identity.json").write_text(
        json.dumps(workspace_payload, indent=2),
        encoding="utf-8",
    )
    if identity["patch"]:
        (root / "code.patch").write_bytes(identity["patch"])
    repository_root = resolve_path(".").resolve()
    for item in identity["untracked_files"]:
        source = (repository_root / item["path"]).resolve()
        target = root / "untracked_files" / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    manifest = RunManifest(
        run_id=identifier,
        run_type=run_type,
        status="running",
        created_at=now.isoformat(),
        data_cutoff=str(pd.Timestamp(data_cutoff).date()) if data_cutoff is not None else None,
        config_hash=config_hash(config),
        config=config.model_dump(mode="json"),
        code_version=str(identity["version"]),
        code_commit=identity["commit"],
        code_dirty=bool(identity["dirty"]),
        code_patch_sha256=identity["patch_sha256"],
        workspace_sha256=identity["workspace_sha256"],
        python_version=platform.python_version(),
        environment_sha256=hashlib.sha256(
            environment_payload.encode("utf-8")
        ).hexdigest(),
        data_sources=config.data_sources.model_dump(mode="json"),
    )
    context = RunContext(root=root, manifest=manifest)
    context.write_manifest()
    return context
