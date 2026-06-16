import json

import pandas as pd
import pytest

from qss.approval.workflow import create_approval_packet, transition_approval
from qss.config.loader import get_config
from qss.experiments.registry import ExperimentRegistry, register_run_path
from qss.factors.metadata import configured_factor_metadata
from qss.research.audit import build_bias_review, write_bias_review


def test_all_configured_factors_have_metadata():
    config = get_config(["configs/default.yaml"])
    metadata = configured_factor_metadata(config)
    configured = {
        name for group in config.factor_groups.values() for name in group.factors
    }
    assert {item.name for item in metadata} == configured
    assert all(item.description and item.inputs and item.leakage_checks for item in metadata)


def test_registry_is_queryable_without_folder_scanning(tmp_path):
    registry = ExperimentRegistry(tmp_path / "registry.duckdb")
    registry.upsert(
        {
            "run_id": "run-1",
            "run_type": "experiment",
            "strategy_id": "strategy-a",
            "universe": "sp500",
            "factor_set_json": '["roe"]',
            "label_type": "forward_return",
            "model_type": "ridge",
            "validation_method": "purged_walk_forward",
            "approval_status": "review_required",
            "status": "valid",
            "config_hash": "abc",
            "run_path": str(tmp_path / "run-1"),
            "created_at": "2025-01-01T00:00:00",
        }
    )
    result = registry.query(model_type="ridge")
    assert list(result["run_id"]) == ["run-1"]


def test_register_run_path_handles_null_research_protocol(tmp_path):
    config = get_config(["configs/default.yaml"])
    config.registry.path = str(tmp_path / "registry.duckdb")
    run = tmp_path / "runs" / "run-null-protocol"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "run-null-protocol",
                "run_type": "backtest",
                "status": "valid",
                "created_at": "2026-01-01T00:00:00+00:00",
                "config_hash": "abc",
                "research_protocol": None,
            }
        ),
        encoding="utf-8",
    )

    assert register_run_path(config, run)
    assert (
        ExperimentRegistry(config.registry.path).query().iloc[0]["run_id"]
        == "run-null-protocol"
    )


def test_approval_requires_human_transition_and_exports_only_after_approval(tmp_path):
    config = get_config(["configs/default.yaml"])
    config.approval.directory = str(tmp_path / "approvals")
    config.paths.reports = str(tmp_path / "reports")
    config.registry.path = str(tmp_path / "registry.duckdb")
    portfolio = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "target_weight": [1.0],
        }
    )
    orders = pd.DataFrame({"symbol": ["AAA"], "action": ["BUY"]})
    packet, path = create_approval_packet(
        config,
        "rebalance-1",
        pd.Timestamp("2025-01-31"),
        portfolio,
        orders,
        {"risk": True},
    )
    assert packet.status == "review_required"
    assert packet.approved_export is None
    with pytest.raises(ValueError):
        transition_approval(config, path, "approved_for_candidate", "", "")
    approved = transition_approval(
        config,
        path,
        "approved_for_candidate",
        "human@example.com",
        "Reviewed",
    )
    assert approved.status == "approved_for_candidate"
    assert approved.approved_export is not None
    assert pd.read_csv(approved.approved_export).iloc[0]["symbol"] == "AAA"


def test_bias_review_is_deterministic_and_structured(tmp_path):
    review = build_bias_review(
        manifest={"quality_gates": {"synthetic_rows": 0}, "bias_flags": []},
        factor_diagnostics=pd.DataFrame(
            {"factor_name": ["roe"], "coverage": [0.95]}
        ),
        sector_exposure=pd.DataFrame({"sector": ["Tech"], "weight": [0.20]}),
        sector_attribution=pd.DataFrame(
            {
                "sector": ["Tech", "Health"],
                "portfolio_linked_contribution": [0.1, 0.1],
            }
        ),
        concentration=pd.DataFrame({"hhi": [0.05]}),
        cost_sensitivity=pd.DataFrame(
            {"cost_bps": [5, 10, 25], "total_return": [0.2, 0.18, 0.1]}
        ),
        data_diagnostics=pd.DataFrame(),
    )
    markdown, structured = write_bias_review(review, tmp_path)
    assert review["recommendation"] == "eligible_for_human_review"
    assert "Blocking Issues" in markdown.read_text(encoding="utf-8")
    assert json.loads(structured.read_text(encoding="utf-8")) == review
