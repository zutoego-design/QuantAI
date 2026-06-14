from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def research_evidence_decision(
    *,
    stage: str,
    primary_metric: str,
    threshold: float,
    bootstrap_summary: pd.DataFrame,
    deflated_sharpe: dict[str, float],
    net_total_return: float,
    required_probability: float,
    blockers: list[str] | None = None,
) -> dict:
    blocking_issues = sorted(set(blockers or []))
    if stage != "confirmatory":
        return {
            "status": "inconclusive",
            "stage": stage,
            "primary_metric": primary_metric,
            "threshold": threshold,
            "blocking_issues": blocking_issues,
            "reasons": [
                "Exploratory experiments cannot support confirmatory research claims."
            ],
        }
    primary = bootstrap_summary.loc[
        bootstrap_summary["metric"] == primary_metric
    ]
    lower_bound = (
        float(primary.iloc[0]["one_sided_lower_95"]) if not primary.empty else None
    )
    probability = deflated_sharpe.get("probability")
    reasons = []
    if lower_bound is None or lower_bound <= threshold:
        reasons.append(
            "The primary metric one-sided 95% lower bound does not exceed "
            "the preregistered threshold."
        )
    if probability is None or probability < required_probability:
        reasons.append(
            "The Deflated Sharpe probability is below the required threshold."
        )
    if net_total_return <= 0:
        reasons.append("Out-of-sample net total return is not positive.")
    if blocking_issues:
        reasons.append("Methodology blockers remain unresolved.")
    if net_total_return <= 0 or blocking_issues:
        status = "rejected"
    elif reasons:
        status = "inconclusive"
    else:
        status = "supported"
    return {
        "status": status,
        "stage": stage,
        "primary_metric": primary_metric,
        "threshold": threshold,
        "primary_metric_one_sided_lower_95": lower_bound,
        "deflated_sharpe_probability": probability,
        "required_deflated_sharpe_probability": required_probability,
        "net_total_return": net_total_return,
        "blocking_issues": blocking_issues,
        "reasons": reasons,
    }


def write_research_decision(decision: dict, root: str | Path) -> tuple[Path, Path]:
    directory = Path(root)
    json_path = directory / "research_decision.json"
    markdown_path = directory / "research_decision.md"
    json_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    lines = [
        "# Research Evidence Decision",
        "",
        f"- Status: `{decision['status']}`",
        f"- Stage: `{decision['stage']}`",
        f"- Primary metric: `{decision['primary_metric']}`",
        f"- Preregistered threshold: `{decision['threshold']}`",
        "",
        "## Blocking Issues",
        *(
            [f"- {item}" for item in decision.get("blocking_issues", [])]
            or ["- None."]
        ),
        "",
        "## Reasons",
        *([f"- {item}" for item in decision.get("reasons", [])] or ["- All gates passed."]),
    ]
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return markdown_path, json_path
