from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def build_bias_review(
    *,
    manifest: dict,
    factor_diagnostics: pd.DataFrame,
    sector_exposure: pd.DataFrame,
    sector_attribution: pd.DataFrame,
    concentration: pd.DataFrame,
    cost_sensitivity: pd.DataFrame,
    data_diagnostics: pd.DataFrame,
    delisting_sensitivity: pd.DataFrame | None = None,
) -> dict:
    blocking: list[str] = []
    concerns: list[str] = []
    follow_ups: list[str] = []
    quality = manifest.get("quality_gates", {})
    if quality.get("synthetic_rows", 0):
        blocking.append("Synthetic input rows were used in a research-mode run.")
    if quality.get("missing_return_fill_count", 0):
        concerns.append("Missing security returns were filled during the ledger simulation.")
    bias_flags = manifest.get("bias_flags", [])
    if any("survivorship" in flag or "approximate" in flag for flag in bias_flags):
        concerns.append("Universe construction carries survivorship or free-data history risk.")
    exposure_column = (
        "portfolio_weight"
        if "portfolio_weight" in sector_exposure
        else "weight"
        if "weight" in sector_exposure
        else None
    )
    if not sector_exposure.empty and exposure_column:
        peak_sector = float(
            sector_exposure.groupby("sector")[exposure_column].max().max()
        )
        if peak_sector > 0.30:
            concerns.append(f"Peak sector exposure reached {peak_sector:.1%}.")
    if sector_attribution.empty or "portfolio_linked_contribution" not in sector_attribution:
        concerns.append("Sector return attribution is missing.")
        follow_ups.append("Confirm sector exposure is not the primary return driver.")
    else:
        investable = sector_attribution.loc[
            ~sector_attribution["sector"].isin(["Transaction Costs", "Unknown"])
        ]
        absolute_total = float(
            investable["portfolio_linked_contribution"].abs().sum()
        )
        if absolute_total > 0:
            dominant = investable.loc[
                investable["portfolio_linked_contribution"].abs().idxmax()
            ]
            dominant_share = (
                abs(float(dominant["portfolio_linked_contribution"]))
                / absolute_total
            )
            if dominant_share > 0.50:
                concerns.append(
                    f"Sector contribution is dominated by {dominant['sector']} "
                    f"at {dominant_share:.1%} of absolute linked contribution."
                )
    if not concentration.empty and "hhi" in concentration:
        peak_hhi = float(concentration["hhi"].max())
        if peak_hhi > 0.10:
            concerns.append(f"Portfolio concentration HHI reached {peak_hhi:.3f}.")
    if cost_sensitivity.empty or len(cost_sensitivity) < 3:
        concerns.append("Cost sensitivity coverage is incomplete.")
        follow_ups.append("Run at least three transaction-cost scenarios.")
    elif "total_return" in cost_sensitivity:
        worst = float(cost_sensitivity["total_return"].min())
        best = float(cost_sensitivity["total_return"].max())
        if best > 0 and worst <= 0:
            concerns.append("The strategy loses profitability under the tested cost range.")
    if not factor_diagnostics.empty and "coverage" in factor_diagnostics:
        weak = factor_diagnostics.loc[factor_diagnostics["coverage"] < 0.80, "factor_name"]
        if not weak.empty:
            concerns.append(f"Weak factor sample coverage: {', '.join(sorted(weak.astype(str)))}.")
    if not factor_diagnostics.empty and "fdr_significant" in factor_diagnostics:
        supported = factor_diagnostics["fdr_significant"].fillna(False).astype(bool)
        if not supported.any():
            concerns.append(
                "No configured factor survives the multiple-testing-adjusted IC check."
            )
    if (
        delisting_sensitivity is not None
        and not delisting_sensitivity.empty
        and "delisting_liquidations" in delisting_sensitivity
        and int(delisting_sensitivity["delisting_liquidations"].max()) == 0
    ):
        concerns.append(
            "No delisting liquidation was observed; real delisting handling remains unverified."
        )
    if not data_diagnostics.empty and "value" in data_diagnostics:
        low_coverage = data_diagnostics.loc[
            data_diagnostics["check"].astype(str).str.contains("coverage", case=False)
            & (pd.to_numeric(data_diagnostics["value"], errors="coerce") < 0.90)
        ]
        if not low_coverage.empty:
            concerns.append("At least one data coverage diagnostic fell below 90%.")
    follow_ups.extend(
        [
            "Review point-in-time universe and delisting coverage.",
            "Compare rebalance-day and portfolio-size sensitivity.",
        ]
    )
    recommendation = (
        "reject"
        if blocking
        else "hold_for_further_testing"
        if concerns
        else "eligible_for_human_review"
    )
    return {
        "blocking_issues": blocking,
        "major_concerns": concerns,
        "required_follow_up_tests": sorted(set(follow_ups)),
        "recommendation": recommendation,
    }


def write_bias_review(review: dict, run_path: str | Path) -> tuple[Path, Path]:
    root = Path(run_path)
    json_path = root / "bias_review.json"
    json_path.write_text(json.dumps(review, indent=2), encoding="utf-8")
    sections = [
        "# Bias and Risk Review",
        "",
        "## Blocking Issues",
        *([f"- {item}" for item in review["blocking_issues"]] or ["- None."]),
        "",
        "## Major Concerns",
        *([f"- {item}" for item in review["major_concerns"]] or ["- None."]),
        "",
        "## Required Follow-up Tests",
        *([f"- {item}" for item in review["required_follow_up_tests"]] or ["- None."]),
        "",
        "## Recommendation",
        f"- {review['recommendation']}",
    ]
    markdown_path = root / "bias_review.md"
    markdown_path.write_text("\n".join(sections) + "\n", encoding="utf-8")
    return markdown_path, json_path
