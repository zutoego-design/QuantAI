from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from qss.data.storage import resolve_path


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = [
        "model_type",
        "evaluation_scope",
        "evidence_status",
        "acceptance_status",
        "net_total_return",
        "net_sharpe",
        "average_turnover",
        "mean_rank_ic",
        "text_coverage",
        "bias_recommendation",
        "recommendation",
    ]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for row in frame[columns].itertuples(index=False, name=None):
        values = [
            ""
            if pd.isna(value)
            else f"{float(value):.4f}"
            if isinstance(value, float)
            else str(value)
            for value in row
        ]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def _metric_map(path: Path) -> dict[str, float]:
    frame = pd.read_csv(path)
    return {
        str(row.metric): float(row.value)
        for row in frame.itertuples(index=False)
    }


def _full_child_run(experiment_path: Path) -> Path:
    children = json.loads(
        (experiment_path / "child_runs.json").read_text(encoding="utf-8")
    )
    full = next(
        (item for item in children if item.get("period") == "full"),
        None,
    )
    if full is None:
        raise ValueError(f"No full child run in {experiment_path}")
    return experiment_path.parent / str(full["run_id"])


def _acceptance_status(reports_root: Path, *source_run_ids: str) -> str:
    manifests = sorted(
        reports_root.glob("*-acceptance-*/manifest.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for manifest_path in manifests:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if any(
            source_run_id in str(note)
            for note in manifest.get("notes", [])
            for source_run_id in source_run_ids
        ):
            return str(manifest.get("status", "unknown"))
    return "not_run"


def _robustness_matrix_complete(robustness: pd.DataFrame) -> bool:
    if robustness.empty or not {"test", "setting", "run_id"}.issubset(
        robustness.columns
    ):
        return False
    if (
        "status" in robustness
        and robustness["status"].fillna("").isin(["skipped", "invalid"]).any()
    ):
        return False
    completed = {
        (str(row.test), str(row.setting))
        for row in robustness.itertuples(index=False)
        if str(row.run_id).strip()
    }
    required = {
        ("base", "configured"),
        ("subperiod", "first_half"),
        ("subperiod", "second_half"),
        ("top_n_sensitivity", "30"),
        ("top_n_sensitivity", "50"),
        ("top_n_sensitivity", "100"),
        ("rebalance_day_shift", "-5"),
        ("rebalance_day_shift", "5"),
    }
    return required.issubset(completed)


def experiment_comparison_row(
    experiment_path: str | Path,
    reports_root: str | Path,
) -> dict:
    root = Path(experiment_path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    child = _full_child_run(root)
    child_manifest = json.loads(
        (child / "manifest.json").read_text(encoding="utf-8")
    )
    metrics = _metric_map(child / "metrics.csv")
    config = manifest.get("config", {})
    ml_enabled = bool(config.get("ml", {}).get("enabled"))
    text_enabled = bool(config.get("text_factors", {}).get("enabled"))
    model_type = (
        str(config.get("ml", {}).get("model_type"))
        if ml_enabled
        else "text_rule_score"
        if text_enabled
        else "rule_score"
    )
    net_total_return = metrics.get("net_total_return")
    net_sharpe = metrics.get("sharpe_ratio")
    average_turnover = metrics.get("average_turnover")
    mean_rank_ic = None
    evaluation_scope = "legacy_reference"
    holdout_metrics_path = root / "holdout_evaluation" / "portfolio_metrics.csv"
    if holdout_metrics_path.exists():
        holdout = pd.read_csv(holdout_metrics_path).iloc[0]
        net_total_return = float(holdout["net_total_return"])
        net_sharpe = float(holdout["net_sharpe"])
        average_turnover = float(holdout["average_turnover"])
        metrics["cagr"] = float(holdout["cagr"])
        metrics["max_drawdown"] = float(holdout["max_drawdown"])
        evaluation_scope = "holdout"
        selected_model = str(holdout["model_type"])
        model_metadata = (
            root
            / "holdout_evaluation"
            / selected_model
            / "model_evaluation.json"
        )
        if model_metadata.exists():
            mean_rank_ic = float(
                json.loads(model_metadata.read_text(encoding="utf-8")).get(
                    "mean_rank_ic",
                    float("nan"),
                )
            )
    elif ml_enabled:
        ml_root = root / "ml_evaluation"
        portfolio = pd.read_csv(ml_root / "portfolio_metrics.csv").iloc[0]
        aggregate = pd.read_csv(ml_root / "aggregate_metrics.csv").iloc[0]
        net_total_return = float(portfolio["net_total_return"])
        net_sharpe = float(portfolio["net_sharpe"])
        average_turnover = float(portfolio["average_turnover"])
        mean_rank_ic = float(aggregate["mean_rank_ic"])
    diagnostics = pd.read_csv(child / "factor_diagnostics.csv")
    text_row = diagnostics.loc[
        diagnostics["factor_name"] == "risk_disclosure_score"
    ]
    text_coverage = (
        float(1.0 - text_row.iloc[0]["missing_rate"])
        if not text_row.empty
        else None
    )
    bias = json.loads((child / "bias_review.json").read_text(encoding="utf-8"))
    return {
        "experiment_run_id": manifest["run_id"],
        "child_run_id": child_manifest["run_id"],
        "model_type": model_type,
        "evaluation_scope": evaluation_scope,
        "evidence_status": manifest.get("evidence_status", "legacy_reference"),
        "status": child_manifest.get("status"),
        "acceptance_status": _acceptance_status(
            resolve_path(reports_root),
            manifest["run_id"],
            child_manifest["run_id"],
        ),
        "net_total_return": net_total_return,
        "net_sharpe": net_sharpe,
        "average_turnover": average_turnover,
        "cagr": metrics.get("cagr"),
        "max_drawdown": metrics.get("max_drawdown"),
        "mean_rank_ic": mean_rank_ic,
        "text_coverage": text_coverage,
        "bias_recommendation": bias.get("recommendation"),
        "recommendation": "",
    }


def generate_baseline_comparison(
    experiment_paths: list[str | Path],
    output_path: str | Path,
    reports_root: str | Path,
) -> tuple[pd.DataFrame, Path, Path]:
    rows = [
        experiment_comparison_row(path, reports_root)
        for path in experiment_paths
    ]
    frame = pd.DataFrame(rows)
    holdout_rows = frame.loc[frame["evaluation_scope"] == "holdout"]
    rule_rows = holdout_rows.loc[holdout_rows["model_type"] == "rule_score"]
    rule_sharpe = (
        float(rule_rows.iloc[0]["net_sharpe"])
        if not rule_rows.empty
        else float("nan")
    )
    for index, row in frame.iterrows():
        if row["evaluation_scope"] != "holdout":
            recommendation = "legacy_reference"
        elif row["status"] != "valid":
            recommendation = "rejected"
        elif row["evidence_status"] == "supported":
            recommendation = "supported"
        elif row["evidence_status"] == "rejected":
            recommendation = "rejected"
        elif row["model_type"] == "text_rule_score" and (
            pd.isna(row["text_coverage"]) or float(row["text_coverage"]) < 0.80
        ):
            recommendation = "needs_more_data"
        elif (
            row["acceptance_status"] == "valid"
            and (pd.isna(rule_sharpe) or float(row["net_sharpe"]) > rule_sharpe)
            and float(row["net_total_return"]) > 0
        ):
            recommendation = "inconclusive"
        else:
            recommendation = "inconclusive"
        frame.loc[index, "recommendation"] = recommendation

    output = resolve_path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output.with_suffix(".csv")
    frame.to_csv(csv_path, index=False)

    all_rule_rows = frame.loc[frame["model_type"] == "rule_score"]
    rule_experiment = (
        Path(all_rule_rows.iloc[0]["experiment_run_id"])
        if not all_rule_rows.empty
        else Path()
    )
    if rule_experiment and not rule_experiment.is_absolute():
        rule_experiment = resolve_path(reports_root) / str(rule_experiment)
    robustness_path = rule_experiment / "robustness_matrix.csv"
    robustness = (
        pd.read_csv(robustness_path)
        if robustness_path.exists()
        else pd.DataFrame()
    )
    robustness_complete = _robustness_matrix_complete(robustness)
    best = frame.sort_values("net_sharpe", ascending=False).iloc[0]
    lightgbm = frame.loc[frame["model_type"] == "lightgbm"]
    ridge = frame.loc[frame["model_type"] == "ridge"]
    text = frame.loc[frame["model_type"] == "text_rule_score"]
    lines = [
        "# Holdout Baseline Comparison",
        "",
        f"Generated from {len(frame)} canonical experiment runs.",
        "",
        _markdown_table(frame),
        "",
        "## Decisions",
        "",
        f"- Highest net Sharpe: `{best['model_type']}` at "
        f"`{float(best['net_sharpe']):.4f}`.",
        (
            f"- Rule-score holdout net Sharpe: `{rule_sharpe:.4f}`."
            if not pd.isna(rule_sharpe)
            else "- No confirmatory rule-score holdout reference is available."
        ),
        f"- Rule robustness matrix complete: `{str(robustness_complete).lower()}`.",
    ]
    if not lightgbm.empty and not ridge.empty:
        lines.append(
            "- LightGBM vs Ridge net Sharpe difference: "
            f"`{float(lightgbm.iloc[0]['net_sharpe'] - ridge.iloc[0]['net_sharpe']):.4f}`."
        )
    if not text.empty:
        lines.append(
            "- SEC text input coverage: "
            f"`{float(text.iloc[0]['text_coverage']):.2%}`; "
            f"decision `{text.iloc[0]['recommendation']}`."
        )
    lines.extend(
        [
            "",
            "Recommendations are evidence labels, not trading approvals.",
            "",
            "## Sources",
            "",
            *[
                f"- `{row.experiment_run_id}` -> `{row.child_run_id}`"
                for row in frame.itertuples(index=False)
            ],
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return frame, output, csv_path
