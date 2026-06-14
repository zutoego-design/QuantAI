from __future__ import annotations

import json

from qss.progress import PROGRESS_PREFIX, emit_progress, parse_progress_line


def test_progress_event_round_trip(capsys):
    emit_progress("targets", 0.42, "Generating rebalance 4/12")

    line = capsys.readouterr().out.strip()

    assert line.startswith(PROGRESS_PREFIX)
    assert parse_progress_line(line) == {
        "stage": "targets",
        "progress": 0.42,
        "message": "Generating rebalance 4/12",
    }


def test_progress_is_clamped_and_invalid_lines_are_ignored():
    payload = json.dumps(
        {"stage": "complete", "progress": 2.0, "message": "Done"}
    )

    assert parse_progress_line(f"{PROGRESS_PREFIX}{payload}")["progress"] == 1.0
    assert parse_progress_line("ordinary log line") is None
    assert parse_progress_line(f"{PROGRESS_PREFIX}not-json") is None
