from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_report_payload(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if target.is_dir():
        target = target / "report.json"
    payload = _read_json(target)
    if "structured_report" in payload:
        return _read_json(Path(str(payload["structured_report"])))
    return payload


def _numeric(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def compare_report_payloads(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_metrics = left.get("metrics") or {}
    right_metrics = right.get("metrics") or {}
    metric_rows = []
    for metric in sorted(set(left_metrics) | set(right_metrics)):
        left_value = _numeric(left_metrics.get(metric))
        right_value = _numeric(right_metrics.get(metric))
        metric_rows.append(
            {
                "metric": metric,
                "left": left_value,
                "right": right_value,
                "delta": (
                    right_value - left_value
                    if left_value is not None and right_value is not None
                    else None
                ),
            }
        )
    identity_keys = [
        "source_run_id",
        "data_snapshot_id",
        "spec_hash",
        "evidence_status",
        "trial_number",
        "trial_budget",
        "holdout_inspection_count",
    ]
    identity = {
        key: {
            "left": left.get(key),
            "right": right.get(key),
            "changed": left.get(key) != right.get(key),
        }
        for key in identity_keys
    }
    protocol_keys = [
        "study_id",
        "stage",
        "study_status",
        "holdout_start",
        "holdout_end",
        "trial_family",
        "trial_budget",
    ]
    left_protocol = left.get("protocol") or {}
    right_protocol = right.get("protocol") or {}
    protocol = {
        key: {
            "left": left_protocol.get(key, left.get(key)),
            "right": right_protocol.get(key, right.get(key)),
            "changed": left_protocol.get(key, left.get(key))
            != right_protocol.get(key, right.get(key)),
        }
        for key in protocol_keys
    }
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "left_report": left.get("source_run_id"),
        "right_report": right.get("source_run_id"),
        "identity": identity,
        "protocol": protocol,
        "metrics": metric_rows,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def render_report_diff_markdown(diff: dict[str, Any]) -> str:
    lines = [
        "# Report Diff",
        "",
        f"- Left: `{diff.get('left_report') or 'N/A'}`",
        f"- Right: `{diff.get('right_report') or 'N/A'}`",
        f"- Generated at: `{diff.get('generated_at')}`",
        "",
        "## Identity",
        "| Field | Left | Right | Changed |",
        "| --- | --- | --- | --- |",
    ]
    for field, row in (diff.get("identity") or {}).items():
        lines.append(
            "| "
            f"{field} | {_fmt(row.get('left'))} | {_fmt(row.get('right'))} | "
            f"{row.get('changed')} |"
        )
    lines.extend(
        [
            "",
            "## Protocol",
            "| Field | Left | Right | Changed |",
            "| --- | --- | --- | --- |",
        ]
    )
    for field, row in (diff.get("protocol") or {}).items():
        lines.append(
            "| "
            f"{field} | {_fmt(row.get('left'))} | {_fmt(row.get('right'))} | "
            f"{row.get('changed')} |"
        )
    lines.extend(
        [
            "",
            "## Metrics",
            "| Metric | Left | Right | Delta |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in diff.get("metrics") or []:
        lines.append(
            "| "
            f"{row['metric']} | {_fmt(row.get('left'))} | "
            f"{_fmt(row.get('right'))} | {_fmt(row.get('delta'))} |"
        )
    return "\n".join(lines) + "\n"


def write_report_diff(
    left: str | Path,
    right: str | Path,
    output: str | Path,
) -> tuple[Path, Path]:
    left_payload = load_report_payload(left)
    right_payload = load_report_payload(right)
    diff = compare_report_payloads(left_payload, right_payload)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path = output_path.with_suffix(".md")
    json_path = output_path.with_suffix(".json")
    markdown_path.write_text(render_report_diff_markdown(diff), encoding="utf-8")
    json_path.write_text(json.dumps(diff, indent=2, ensure_ascii=False), encoding="utf-8")
    return markdown_path, json_path
