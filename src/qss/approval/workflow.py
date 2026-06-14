from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from qss.config.schema import AppConfig
from qss.data.storage import resolve_path, write_csv
from qss.experiments.registry import ExperimentRegistry

ApprovalState = Literal[
    "draft",
    "review_required",
    "approved_for_candidate",
    "rejected",
]


class ApprovalEvent(BaseModel):
    from_state: ApprovalState
    to_state: ApprovalState
    reviewer: str
    note: str = ""
    changed_at: str


class ApprovalPacket(BaseModel):
    run_id: str
    strategy_id: str
    as_of_date: str
    status: ApprovalState = "review_required"
    candidate_weights: str
    internal_orders: str
    approved_export: str | None = None
    risk_checks: dict[str, bool] = Field(default_factory=dict)
    history: list[ApprovalEvent] = Field(default_factory=list)


def create_approval_packet(
    config: AppConfig,
    run_id: str,
    as_of_date: pd.Timestamp,
    portfolio: pd.DataFrame,
    orders: pd.DataFrame,
    risk_checks: dict[str, bool],
) -> tuple[ApprovalPacket, Path]:
    root = resolve_path(config.approval.directory) / run_id
    root.mkdir(parents=True, exist_ok=False)
    candidate = write_csv(portfolio, root / "candidate_target_weights.csv")
    internal_orders = write_csv(orders, root / "internal_orders.csv")
    packet = ApprovalPacket(
        run_id=run_id,
        strategy_id=config.strategy.name,
        as_of_date=str(pd.Timestamp(as_of_date).date()),
        status="review_required",
        candidate_weights=str(candidate),
        internal_orders=str(internal_orders),
        risk_checks=risk_checks,
    )
    packet_path = root / "approval_packet.json"
    packet_path.write_text(packet.model_dump_json(indent=2), encoding="utf-8")
    return packet, packet_path


def load_approval_packet(path: str | Path) -> tuple[ApprovalPacket, Path]:
    target = Path(path)
    if target.is_dir():
        target = target / "approval_packet.json"
    return ApprovalPacket.model_validate_json(target.read_text(encoding="utf-8")), target


def transition_approval(
    config: AppConfig,
    packet_path: str | Path,
    new_state: Literal["approved_for_candidate", "rejected"],
    reviewer: str,
    note: str = "",
) -> ApprovalPacket:
    if not reviewer.strip():
        raise ValueError("A human reviewer identity is required.")
    packet, target = load_approval_packet(packet_path)
    if packet.status != "review_required":
        raise ValueError(f"Approval packet is already in terminal state {packet.status!r}.")
    if new_state == "approved_for_candidate" and not all(packet.risk_checks.values()):
        raise ValueError("Cannot approve a packet with failed risk checks.")
    event = ApprovalEvent(
        from_state=packet.status,
        to_state=new_state,
        reviewer=reviewer.strip(),
        note=note,
        changed_at=pd.Timestamp.now(tz="UTC").isoformat(),
    )
    packet.history.append(event)
    packet.status = new_state
    if new_state == "approved_for_candidate":
        weights = pd.read_csv(packet.candidate_weights)
        approved = write_csv(weights, target.parent / "approved_target_weights.csv")
        packet.approved_export = str(approved)
    target.write_text(packet.model_dump_json(indent=2), encoding="utf-8")
    run_root = resolve_path(config.paths.reports) / "runs" / packet.run_id
    if run_root.exists():
        (run_root / "approval_packet.json").write_text(
            packet.model_dump_json(indent=2),
            encoding="utf-8",
        )
        if packet.approved_export:
            write_csv(
                pd.read_csv(packet.approved_export),
                run_root / "approved_target_weights.csv",
            )
    if config.registry.enabled:
        ExperimentRegistry.from_config(config).update_approval_status(
            packet.run_id, packet.status
        )
    return packet
