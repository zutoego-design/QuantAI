import json
from pathlib import Path

import pandas as pd

from qss.reporting.comprehensive_report import (
    ensure_comprehensive_report,
    find_latest_research_run,
    generate_comprehensive_report,
)
from qss.reporting.report_diff import compare_report_payloads, write_report_diff


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_experiment(reports: Path, run_id: str, created_at: str, status: str) -> Path:
    root = reports / "runs" / run_id
    root.mkdir(parents=True)
    protocol = {
        "study_id": "test-study",
        "stage": "confirmatory",
        "development_start": "2023-01-01",
        "development_end": "2023-06-30",
        "holdout_start": "2023-08-01",
        "holdout_end": "2023-12-31",
        "primary_metric": "sharpe_ratio",
        "primary_metric_threshold": 0.0,
    }
    _write_json(
        root / "manifest.json",
        {
            "run_id": run_id,
            "run_type": "experiment",
            "status": status,
            "created_at": created_at,
            "data_cutoff": "2023-12-31",
            "data_snapshot_id": "snapshot-id",
            "spec_hash": "spec-hash",
            "trial_number": 2,
            "code_dirty": True,
            "code_version": "git:abc:dirty:def",
            "research_protocol": protocol,
        },
    )
    _write_json(root / "research_protocol.json", protocol)
    _write_json(
        root / "research_decision.json",
        {
            "status": "rejected",
            "selected_model": "rule_score",
            "required_deflated_sharpe_probability": 0.95,
            "reasons": [
                "The Deflated Sharpe probability is below the required threshold."
            ],
            "blocking_issues": [],
        },
    )
    if status != "valid":
        return root

    evaluation = root / "holdout_evaluation" / "rule_score"
    evaluation.mkdir(parents=True)
    pd.DataFrame(
        [
            {"category": "performance", "metric": "net_total_return", "value": 0.12},
            {"category": "performance", "metric": "benchmark_total_return", "value": 0.08},
            {"category": "performance", "metric": "cagr", "value": 0.11},
            {"category": "performance", "metric": "sharpe_ratio", "value": 1.1},
            {"category": "performance", "metric": "max_drawdown", "value": -0.09},
            {"category": "portfolio", "metric": "average_turnover", "value": 0.1},
            {"category": "portfolio", "metric": "cost_drag", "value": 0.002},
        ]
    ).to_csv(evaluation / "metrics.csv", index=False)
    pd.DataFrame(
        {
            "date": ["2023-08-01", "2023-08-02", "2023-08-03"],
            "portfolio_value": [1_000_000, 1_010_000, 1_020_000],
            "benchmark_value": [1_000_000, 1_005_000, 1_008_000],
            "drawdown": [0.0, 0.0, -0.001],
        }
    ).to_csv(evaluation / "daily_returns.csv", index=False)
    pd.DataFrame(
        [
            {
                "metric": "sharpe_ratio",
                "estimate": 1.1,
                "lower_95": -0.1,
                "upper_95": 2.0,
                "one_sided_lower_95": 0.1,
            }
        ]
    ).to_csv(root / "bootstrap_summary.csv", index=False)
    _write_json(
        root / "deflated_sharpe.json",
        {
            "observed_sharpe": 1.1,
            "expected_max_sharpe": 0.7,
            "probability": 0.8,
            "trial_count": 2,
        },
    )
    _write_json(
        root / "style_factor_summary.json",
        {"coverage": 1.0, "r_squared": 0.7},
    )
    pd.DataFrame(
        [
            {
                "factor": "alpha",
                "annualized_coefficient": 0.04,
                "t_stat": 0.8,
                "p_value": 0.4,
            }
        ]
    ).to_csv(root / "style_factor_exposures.csv", index=False)
    pd.DataFrame(
        [
            {
                "factor_name": "value",
                "ic": 0.02,
                "fdr_q_value": 0.2,
                "fdr_significant": False,
                "direction_matches": True,
            }
        ]
    ).to_csv(root / "factor_evidence.csv", index=False)
    pd.DataFrame([{"test": "base", "status": "valid"}]).to_csv(
        root / "robustness_matrix.csv",
        index=False,
    )
    return root


def _make_backtest(
    reports: Path,
    run_id: str,
    created_at: str,
    *,
    artifact_level: str | None = None,
) -> Path:
    root = reports / "runs" / run_id
    root.mkdir(parents=True)
    manifest = {
        "run_id": run_id,
        "run_type": "backtest",
        "status": "valid",
        "created_at": created_at,
    }
    if artifact_level is not None:
        manifest["quality_gates"] = {"artifact_level": artifact_level}
    _write_json(root / "manifest.json", manifest)
    pd.DataFrame(
        [
            {"metric": "net_total_return", "value": 0.2},
            {"metric": "sharpe_ratio", "value": 1.0},
        ]
    ).to_csv(root / "metrics.csv", index=False)
    pd.DataFrame(
        {
            "date": ["2023-01-01", "2023-01-02"],
            "portfolio_value": [1_000_000, 1_010_000],
            "benchmark_value": [1_000_000, 1_005_000],
            "drawdown": [0.0, 0.0],
        }
    ).to_csv(root / "daily_returns.csv", index=False)
    return root


def test_comprehensive_report_selects_latest_valid_research_result(tmp_path):
    reports = tmp_path / "reports"
    valid = _make_experiment(
        reports,
        "20260101T000000Z-experiment-valid",
        "2026-01-01T00:00:00+00:00",
        "valid",
    )
    _make_experiment(
        reports,
        "20260102T000000Z-experiment-invalid",
        "2026-01-02T00:00:00+00:00",
        "invalid",
    )
    _make_backtest(
        reports,
        "20260104T000000Z-backtest-robustness",
        "2026-01-04T00:00:00+00:00",
        artifact_level="metrics",
    )
    acceptance = reports / "runs" / "20260103T000000Z-acceptance-test"
    _write_json(
        acceptance / "manifest.json",
        {
            "run_id": acceptance.name,
            "run_type": "acceptance",
            "status": "valid",
            "created_at": "2026-01-03T00:00:00+00:00",
            "notes": [f"Validated source run {valid.name}."],
        },
    )
    pd.DataFrame(
        [{"check": "source_run_valid", "passed": True, "details": "valid"}]
    ).to_csv(acceptance / "acceptance_checks.csv", index=False)

    assert find_latest_research_run(reports) == valid
    bundle = generate_comprehensive_report(
        reports,
        output_root=reports / "comprehensive",
    )
    payload = json.loads(bundle.structured_report.read_text(encoding="utf-8"))
    report = bundle.html_report.read_text(encoding="utf-8")

    assert payload["source_run_id"] == valid.name
    assert payload["metrics"]["net_total_return"] == 0.12
    assert payload["evidence_status"] == "rejected"
    assert payload["code_dirty"] is True
    assert "Dirty git workspace" in report
    assert (
        "../../runs/20260101T000000Z-experiment-valid/"
        "holdout_evaluation/rule_score/metrics.csv"
    ) in report
    assert bundle.pointer.exists()

    reused = ensure_comprehensive_report(reports)
    assert reused.html_report == bundle.html_report


def test_comprehensive_report_uses_full_backtest_when_no_experiment_exists(tmp_path):
    reports = tmp_path / "reports"
    full = _make_backtest(
        reports,
        "20260101T000000Z-backtest-full",
        "2026-01-01T00:00:00+00:00",
    )
    _make_backtest(
        reports,
        "20260102T000000Z-backtest-robustness",
        "2026-01-02T00:00:00+00:00",
        artifact_level="metrics",
    )

    assert find_latest_research_run(reports) == full

    bundle = generate_comprehensive_report(
        reports,
        output_root=reports / "comprehensive",
    )
    report = bundle.html_report.read_text(encoding="utf-8")

    assert report.count('<span class="gate-status">未评估</span>') == 3
    assert "该运行未生成确认性 Bootstrap、Deflated Sharpe 和预注册因子证据。" in report


def test_report_diff_compares_identity_and_metric_delta(tmp_path):
    left = {
        "source_run_id": "run-a",
        "evidence_status": "rejected",
        "data_snapshot_id": "snapshot-a",
        "spec_hash": "spec-a",
        "trial_number": 1,
        "trial_budget": 2,
        "protocol": {"study_id": "study-a", "holdout_start": "2024-01-01"},
        "metrics": {"net_total_return": 0.10, "sharpe_ratio": 0.8},
    }
    right = {
        **left,
        "source_run_id": "run-b",
        "evidence_status": "rejected_final",
        "holdout_inspection_count": 2,
        "metrics": {"net_total_return": 0.13, "sharpe_ratio": 0.7},
    }

    diff = compare_report_payloads(left, right)
    returns = next(
        row for row in diff["metrics"] if row["metric"] == "net_total_return"
    )
    assert round(returns["delta"], 6) == 0.03
    assert diff["identity"]["evidence_status"]["changed"]

    left_path = tmp_path / "left.json"
    right_path = tmp_path / "right.json"
    _write_json(left_path, left)
    _write_json(right_path, right)
    markdown_path, json_path = write_report_diff(
        left_path,
        right_path,
        tmp_path / "diff",
    )
    assert markdown_path.exists()
    assert json.loads(json_path.read_text(encoding="utf-8"))["right_report"] == "run-b"
