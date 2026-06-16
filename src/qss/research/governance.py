from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qss.data.storage import resolve_path
from qss.research.protocol import ResearchProtocol
from qss.runs.manifest import workspace_identity

CLOSED_STUDY_STATUSES = {"closed", "superseded", "rejected_final"}
DEFAULT_STUDY_CLOSURES_PATH = Path("experiments") / "study_closures.json"


def load_study_closures(path: str | Path = DEFAULT_STUDY_CLOSURES_PATH) -> list[dict[str, Any]]:
    target = resolve_path(path)
    if not target.exists():
        return []
    payload = json.loads(target.read_text(encoding="utf-8"))
    closures = payload.get("closures", payload) if isinstance(payload, dict) else payload
    if not isinstance(closures, list):
        raise ValueError(f"Study closure registry must be a list: {target}")
    return [item for item in closures if isinstance(item, dict)]


def holdout_reuse_detector(
    protocol: ResearchProtocol,
    closures: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if protocol.stage != "confirmatory":
        return []
    closed = closures if closures is not None else load_study_closures()
    violations: list[dict[str, Any]] = []
    for closure in closed:
        status = str(closure.get("study_status") or closure.get("status") or "")
        if status not in CLOSED_STUDY_STATUSES:
            continue
        same_study = closure.get("study_id") == protocol.study_id
        same_family_holdout = (
            closure.get("trial_family") == protocol.trial_family
            and closure.get("holdout_start") == protocol.holdout_start
            and closure.get("holdout_end") == protocol.holdout_end
        )
        if not same_study and not same_family_holdout:
            continue
        violations.append(
            {
                "study_id": closure.get("study_id"),
                "study_status": status,
                "trial_family": closure.get("trial_family"),
                "holdout_start": closure.get("holdout_start"),
                "holdout_end": closure.get("holdout_end"),
                "final_run_id": closure.get("final_run_id"),
                "reason": (
                    "closed_study" if same_study else "closed_holdout_reuse"
                ),
            }
        )
    return violations


def confirmatory_rerun_guard(
    protocol: ResearchProtocol,
    *,
    trial_number: int,
    trial_budget: int | None = None,
    registry_enabled: bool = True,
    closures: list[dict[str, Any]] | None = None,
    require_clean_git: bool | None = None,
    identity: dict[str, Any] | None = None,
) -> None:
    if protocol.stage != "confirmatory":
        return
    if protocol.study_status in CLOSED_STUDY_STATUSES:
        raise ValueError(
            f"Study {protocol.study_id} is {protocol.study_status}; "
            "create a new preregistered study before running confirmatory evidence."
        )
    violations = holdout_reuse_detector(protocol, closures)
    if violations:
        violation = violations[0]
        raise ValueError(
            "Confirmatory holdout reuse is blocked: "
            f"{violation['study_id']} is {violation['study_status']} for "
            f"{violation['holdout_start']}..{violation['holdout_end']} "
            f"(final run {violation.get('final_run_id') or 'unknown'})."
        )
    clean_required = (
        protocol.clean_git_required
        if require_clean_git is None
        else require_clean_git
    )
    if clean_required:
        current_identity = identity if identity is not None else workspace_identity()
        if current_identity.get("dirty"):
            raise ValueError(
                "Confirmatory studies require a clean git workspace. "
                f"Current identity is {current_identity.get('version', 'unknown')}."
            )
    budget = trial_budget if trial_budget is not None else protocol.trial_budget
    if budget is None:
        return
    if not registry_enabled:
        raise ValueError("Confirmatory trial_budget enforcement requires the registry.")
    if trial_number > budget:
        raise ValueError(
            f"Trial budget exceeded for {protocol.trial_family}: "
            f"next trial {trial_number} > budget {budget}."
        )
