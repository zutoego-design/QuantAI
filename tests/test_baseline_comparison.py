import json

import pandas as pd

from qss.research.comparison import generate_baseline_comparison


def _experiment(root, experiment_id, child_id, model_type, ml=False, text=False):
    experiment = root / experiment_id
    child = root / child_id
    experiment.mkdir(parents=True)
    child.mkdir(parents=True)
    manifest = {
        "run_id": experiment_id,
        "config": {
            "ml": {"enabled": ml, "model_type": model_type},
            "text_factors": {"enabled": text},
        },
    }
    (experiment / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (experiment / "child_runs.json").write_text(
        json.dumps([{"period": "full", "run_id": child_id}]),
        encoding="utf-8",
    )
    (child / "manifest.json").write_text(
        json.dumps({"run_id": child_id, "status": "valid"}),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"metric": "net_total_return", "value": 1.0},
            {"metric": "sharpe_ratio", "value": 1.0},
            {"metric": "average_turnover", "value": 0.1},
            {"metric": "cagr", "value": 0.1},
            {"metric": "max_drawdown", "value": -0.2},
        ]
    ).to_csv(child / "metrics.csv", index=False)
    pd.DataFrame(
        {
            "factor_name": ["risk_disclosure_score"] if text else ["roe"],
            "missing_rate": [0.5 if text else 0.0],
        }
    ).to_csv(child / "factor_diagnostics.csv", index=False)
    (child / "bias_review.json").write_text(
        json.dumps({"recommendation": "eligible_for_human_review"}),
        encoding="utf-8",
    )
    if ml:
        ml_root = experiment / "ml_evaluation"
        ml_root.mkdir()
        pd.DataFrame(
            [
                {
                    "net_total_return": 1.2,
                    "net_sharpe": 1.1,
                    "average_turnover": 0.2,
                }
            ]
        ).to_csv(ml_root / "portfolio_metrics.csv", index=False)
        pd.DataFrame([{"mean_rank_ic": 0.02}]).to_csv(
            ml_root / "aggregate_metrics.csv",
            index=False,
        )
    return experiment


def test_baseline_comparison_uses_ml_metrics_and_marks_sparse_text(tmp_path):
    rule = _experiment(
        tmp_path,
        "rule-experiment",
        "rule-child",
        "ridge",
    )
    ridge = _experiment(
        tmp_path,
        "ridge-experiment",
        "ridge-child",
        "ridge",
        ml=True,
    )
    text = _experiment(
        tmp_path,
        "text-experiment",
        "text-child",
        "ridge",
        text=True,
    )
    fixed_rows = [
        {"test": "base", "setting": "configured", "run_id": "base"},
        {
            "test": "subperiod",
            "setting": "first_half",
            "run_id": "first",
        },
        {
            "test": "subperiod",
            "setting": "second_half",
            "run_id": "second",
        },
    ]
    top_n_rows = [
        {
            "test": "top_n_sensitivity",
            "setting": setting,
            "run_id": f"top-{setting}",
        }
        for setting in ["30", "50", "100"]
    ]
    shift_rows = [
        {
            "test": "rebalance_day_shift",
            "setting": setting,
            "run_id": f"shift-{setting}",
        }
        for setting in ["-5", "5"]
    ]
    pd.DataFrame(fixed_rows + top_n_rows + shift_rows).to_csv(
        rule / "robustness_matrix.csv",
        index=False,
    )
    acceptance = tmp_path / "20260614T000000Z-acceptance-test"
    acceptance.mkdir()
    (acceptance / "manifest.json").write_text(
        json.dumps(
            {
                "status": "valid",
                "notes": ["Validated source run rule-experiment."],
            }
        ),
        encoding="utf-8",
    )

    frame, markdown, csv_path = generate_baseline_comparison(
        [rule, ridge, text],
        tmp_path / "comparison.md",
        tmp_path,
    )

    decisions = frame.set_index("model_type")["recommendation"].to_dict()
    assert decisions["rule_score"] == "legacy_reference"
    assert decisions["ridge"] == "legacy_reference"
    assert decisions["text_rule_score"] == "legacy_reference"
    assert frame.loc[
        frame["experiment_run_id"] == "rule-experiment",
        "acceptance_status",
    ].iloc[0] == "valid"
    assert markdown.exists()
    assert csv_path.exists()
    content = markdown.read_text(
        encoding="utf-8"
    )
    assert "Rule robustness matrix complete: `true`." in content
    assert "No confirmatory rule-score holdout reference is available." in content
