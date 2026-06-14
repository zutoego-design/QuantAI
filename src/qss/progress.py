from __future__ import annotations

import json
from typing import Any

PROGRESS_PREFIX = "QSS_PROGRESS "


def emit_progress(stage: str, progress: float, message: str) -> None:
    payload = {
        "stage": stage,
        "progress": min(max(float(progress), 0.0), 1.0),
        "message": message,
    }
    print(
        f"{PROGRESS_PREFIX}{json.dumps(payload, ensure_ascii=True)}",
        flush=True,
    )


def parse_progress_line(line: str) -> dict[str, Any] | None:
    if not line.startswith(PROGRESS_PREFIX):
        return None
    try:
        payload = json.loads(line[len(PROGRESS_PREFIX) :])
        return {
            "stage": str(payload["stage"]),
            "progress": min(max(float(payload["progress"]), 0.0), 1.0),
            "message": str(payload["message"]),
        }
    except (KeyError, TypeError, ValueError):
        return None
