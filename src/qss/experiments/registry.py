from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from qss.config.schema import AppConfig
from qss.data.storage import resolve_path

REGISTRY_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    run_id VARCHAR PRIMARY KEY,
    parent_run_id VARCHAR,
    run_type VARCHAR NOT NULL,
    strategy_id VARCHAR,
    universe VARCHAR,
    factor_set_json VARCHAR,
    label_type VARCHAR,
    model_type VARCHAR,
    start_date DATE,
    end_date DATE,
    validation_method VARCHAR,
    net_cagr DOUBLE,
    net_sharpe DOUBLE,
    max_drawdown DOUBLE,
    turnover DOUBLE,
    approval_status VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    config_hash VARCHAR,
    study_id VARCHAR,
    research_stage VARCHAR,
    trial_family VARCHAR,
    protocol_json VARCHAR,
    spec_hash VARCHAR,
    data_snapshot_id VARCHAR,
    trial_number INTEGER,
    evidence_status VARCHAR,
    evaluation_scope VARCHAR,
    run_path VARCHAR NOT NULL,
    created_at TIMESTAMP NOT NULL
)
"""

REGISTRY_ADDITIONS = {
    "study_id": "VARCHAR",
    "research_stage": "VARCHAR",
    "trial_family": "VARCHAR",
    "protocol_json": "VARCHAR",
    "spec_hash": "VARCHAR",
    "data_snapshot_id": "VARCHAR",
    "trial_number": "INTEGER",
    "evidence_status": "VARCHAR",
    "evaluation_scope": "VARCHAR",
}


def _connect_with_retry(
    path: Path,
    *,
    read_only: bool = False,
    attempts: int = 10,
    delay_seconds: float = 0.25,
):
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return duckdb.connect(str(path), read_only=read_only)
        except duckdb.IOException as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(delay_seconds * (attempt + 1))
    assert last_error is not None
    raise last_error


class ExperimentRegistry:
    def __init__(self, path: str | Path):
        self.path = resolve_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _connect_with_retry(self.path) as connection:
            connection.execute(REGISTRY_SCHEMA)
            columns = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info('experiments')"
                ).fetchall()
            }
            for name, data_type in REGISTRY_ADDITIONS.items():
                if name not in columns:
                    connection.execute(
                        f"ALTER TABLE experiments ADD COLUMN {name} {data_type}"
                    )

    @classmethod
    def from_config(cls, config: AppConfig) -> "ExperimentRegistry":
        return cls(config.registry.path)

    def upsert(self, record: dict[str, Any]) -> None:
        columns = [
            "run_id",
            "parent_run_id",
            "run_type",
            "strategy_id",
            "universe",
            "factor_set_json",
            "label_type",
            "model_type",
            "start_date",
            "end_date",
            "validation_method",
            "net_cagr",
            "net_sharpe",
            "max_drawdown",
            "turnover",
            "approval_status",
            "status",
            "config_hash",
            "study_id",
            "research_stage",
            "trial_family",
            "protocol_json",
            "spec_hash",
            "data_snapshot_id",
            "trial_number",
            "evidence_status",
            "evaluation_scope",
            "run_path",
            "created_at",
        ]
        values = [record.get(column) for column in columns]
        placeholders = ", ".join(["?"] * len(columns))
        updates = ", ".join(
            f"{column} = excluded.{column}" for column in columns if column != "run_id"
        )
        with _connect_with_retry(self.path) as connection:
            connection.execute(
                f"""
                INSERT INTO experiments ({", ".join(columns)})
                VALUES ({placeholders})
                ON CONFLICT (run_id) DO UPDATE SET {updates}
                """,
                values,
            )

    def update_approval_status(self, run_id: str, approval_status: str) -> None:
        with _connect_with_retry(self.path) as connection:
            connection.execute(
                "UPDATE experiments SET approval_status = ? WHERE run_id = ?",
                [approval_status, run_id],
            )

    def next_trial_number(self, trial_family: str) -> int:
        with _connect_with_retry(self.path, read_only=True) as connection:
            value = connection.execute(
                """
                SELECT COUNT(*)
                FROM experiments
                WHERE trial_family = ? AND run_type = 'experiment'
                """,
                [trial_family],
            ).fetchone()[0]
        return int(value or 0) + 1

    def trial_count(self, trial_family: str) -> int:
        with _connect_with_retry(self.path, read_only=True) as connection:
            value = connection.execute(
                """
                SELECT COUNT(*)
                FROM experiments
                WHERE trial_family = ? AND run_type = 'experiment'
                """,
                [trial_family],
            ).fetchone()[0]
        return int(value or 0)

    def data_snapshot_for_spec(self, spec_hash: str) -> str | None:
        with _connect_with_retry(self.path, read_only=True) as connection:
            row = connection.execute(
                """
                SELECT data_snapshot_id
                FROM experiments
                WHERE spec_hash = ?
                  AND run_type = 'experiment'
                  AND data_snapshot_id IS NOT NULL
                ORDER BY created_at DESC
                LIMIT 1
                """,
                [spec_hash],
            ).fetchone()
        return str(row[0]) if row and row[0] else None

    def query(
        self,
        *,
        strategy_id: str | None = None,
        model_type: str | None = None,
        approval_status: str | None = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        clauses = []
        parameters: list[Any] = []
        for column, value in [
            ("strategy_id", strategy_id),
            ("model_type", model_type),
            ("approval_status", approval_status),
        ]:
            if value is not None:
                clauses.append(f"{column} = ?")
                parameters.append(value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        parameters.append(max(1, min(limit, 1000)))
        with _connect_with_retry(self.path, read_only=True) as connection:
            return connection.execute(
                f"SELECT * FROM experiments {where} ORDER BY created_at DESC LIMIT ?",
                parameters,
            ).fetch_df()


def registry_record_from_run(
    config: AppConfig,
    run_id: str,
    run_type: str,
    run_path: str | Path,
    *,
    status: str,
    created_at: str,
    config_hash: str,
    start_date: str | None = None,
    end_date: str | None = None,
    metrics: pd.DataFrame | None = None,
    parent_run_id: str | None = None,
    approval_status: str = "draft",
    research_protocol: dict[str, Any] | None = None,
    spec_hash: str | None = None,
    data_snapshot_id: str | None = None,
    trial_number: int | None = None,
    evidence_status: str | None = None,
    evaluation_scope: str | None = None,
) -> dict[str, Any]:
    metric_values: dict[str, float] = {}
    if metrics is not None and not metrics.empty:
        if {"metric", "value"}.issubset(metrics.columns):
            metric_values = metrics.set_index("metric")["value"].to_dict()
        elif len(metrics) == 1:
            metric_values = {
                str(column): float(value)
                for column, value in metrics.iloc[0].items()
                if pd.notna(value) and isinstance(value, (int, float))
            }
    factor_names = sorted(
        name for group in config.factor_groups.values() for name in group.factors
    )
    return {
        "run_id": run_id,
        "parent_run_id": parent_run_id,
        "run_type": run_type,
        "strategy_id": config.strategy.name,
        "universe": config.universe.name,
        "factor_set_json": json.dumps(factor_names),
        "label_type": config.ml.target if config.ml.enabled else "forward_return",
        "model_type": config.ml.model_type if config.ml.enabled else "rule_score",
        "start_date": start_date,
        "end_date": end_date,
        "validation_method": "purged_walk_forward" if config.ml.enabled else "backtest",
        "net_cagr": metric_values.get("cagr"),
        "net_sharpe": metric_values.get("sharpe_ratio"),
        "max_drawdown": metric_values.get("max_drawdown"),
        "turnover": metric_values.get("average_turnover"),
        "approval_status": approval_status,
        "status": status,
        "config_hash": config_hash,
        "study_id": (research_protocol or {}).get("study_id"),
        "research_stage": (research_protocol or {}).get("stage"),
        "trial_family": (research_protocol or {}).get("trial_family"),
        "protocol_json": (
            json.dumps(research_protocol, sort_keys=True)
            if research_protocol is not None
            else None
        ),
        "spec_hash": spec_hash,
        "data_snapshot_id": data_snapshot_id,
        "trial_number": trial_number,
        "evidence_status": evidence_status,
        "evaluation_scope": evaluation_scope,
        "run_path": str(Path(run_path).resolve()),
        "created_at": created_at,
    }


def register_run_path(config: AppConfig, run_path: str | Path) -> bool:
    run_root = resolve_path(run_path)
    manifest_path = run_root / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    metrics_path = run_root / "metrics.csv"
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else None
    approval_path = run_root / "approval_packet.json"
    approval_status = "draft"
    if approval_path.exists():
        try:
            approval_status = json.loads(
                approval_path.read_text(encoding="utf-8")
            ).get("status", "draft")
        except (OSError, ValueError, json.JSONDecodeError):
            approval_status = "draft"
    ExperimentRegistry.from_config(config).upsert(
        registry_record_from_run(
            config,
            manifest["run_id"],
            manifest.get("run_type", "unknown"),
            run_root,
            status=manifest.get("status", "unknown"),
            created_at=manifest.get(
                "created_at",
                pd.Timestamp.now(tz="UTC").isoformat(),
            ),
            config_hash=manifest.get("config_hash", ""),
            start_date=manifest.get("config", {})
            .get("backtest", {})
            .get("start_date"),
            end_date=manifest.get("data_cutoff"),
            metrics=metrics,
            approval_status=approval_status,
            research_protocol=manifest.get("research_protocol"),
            spec_hash=manifest.get("spec_hash"),
            data_snapshot_id=manifest.get("data_snapshot_id"),
            trial_number=manifest.get("trial_number"),
            evidence_status=manifest.get("evidence_status"),
            evaluation_scope=(
                "holdout"
                if manifest.get("research_protocol", {}).get("stage")
                == "confirmatory"
                else "full_sample"
            ),
        )
    )
    return True


def refresh_registry(config: AppConfig) -> int:
    runs_root = resolve_path(config.paths.reports) / "runs"
    refreshed = 0
    for manifest_path in runs_root.glob("*/manifest.json"):
        if register_run_path(config, manifest_path.parent):
            refreshed += 1
    return refreshed
