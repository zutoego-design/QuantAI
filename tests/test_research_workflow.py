import json

import pandas as pd
import pytest

from qss.backtest.metrics import comprehensive_factor_diagnostics
from qss.config.loader import get_config
from qss.data.validation import failed_check_summary
from qss.research.orchestrator import ExperimentSpec
from qss.runs.manifest import create_run_context


def test_run_contexts_are_isolated_and_manifested(tmp_path):
    config = get_config(["configs/default.yaml"])
    config.paths.reports = str(tmp_path)
    first = create_run_context(config, "test", "2025-01-01")
    second = create_run_context(config, "test", "2025-01-01")
    assert first.root != second.root
    manifest = json.loads(first.path("manifest.json").read_text(encoding="utf-8"))
    assert manifest["config_hash"]
    assert manifest["status"] == "running"


def test_experiment_spec_rejects_reversed_dates():
    with pytest.raises(ValueError):
        ExperimentSpec(
            hypothesis="test",
            start_date="2025-02-01",
            end_date="2025-01-01",
        )


def test_experiment_overrides_are_bounded():
    config = get_config(["configs/default.yaml"])
    spec = ExperimentSpec(
        hypothesis="single factor",
        factors=["roe"],
        portfolio={"target_num_holdings": 25},
        costs={"commission_bps": 2.0},
        start_date="2020-01-01",
        end_date="2025-01-01",
    )
    from qss.research.orchestrator import ResearchOrchestrator

    updated = ResearchOrchestrator(config)._configured_experiment(spec)
    assert list(updated.factor_groups) == ["quality"]
    assert list(updated.factor_groups["quality"].factors) == ["roe"]
    assert updated.optimizer.constraints.target_num_holdings == 25
    assert updated.backtest.transaction_cost.commission_bps == 2.0


def test_factor_diagnostics_include_ic_quantiles_decay_and_correlation():
    dates = pd.to_datetime(["2025-01-01", "2025-02-01"])
    factor_rows = []
    price_rows = []
    price_dates = pd.date_range("2024-12-31", periods=90, freq="D")
    for index, symbol in enumerate(["A", "B", "C", "D", "E"]):
        for date in dates:
            factor_rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "factor_name": "quality",
                    "processed_value": float(index),
                }
            )
        for offset, date in enumerate(price_dates):
            price_rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "adj_close": 100 + offset * (index + 1),
                }
            )
    reports = comprehensive_factor_diagnostics(
        pd.DataFrame(factor_rows), pd.DataFrame(price_rows)
    )
    assert {"ic", "rank_ic", "ic_ir", "t_stat", "coverage"}.issubset(
        reports["summary"].columns
    )
    assert not reports["quantiles"].empty
    assert set(reports["decay"]["horizon_days"]) >= {1, 5, 21}
    assert not reports["correlation"].empty


def test_failed_check_summary_is_actionable_and_bounded():
    checks = pd.DataFrame(
        [
            {"check": "prices_nonempty", "passed": False, "value": 0},
            {"check": "macro_nonempty", "passed": True, "value": 20},
            {"check": "membership_nonempty", "passed": False, "value": 0},
        ]
    )
    assert failed_check_summary(checks, limit=1) == "prices_nonempty=0; +1 more"


def test_two_calendar_years_include_subperiod_robustness(monkeypatch, tmp_path):
    config = get_config(["configs/default.yaml"])
    config.paths.reports = str(tmp_path / "reports")
    config.registry.enabled = False
    spec = ExperimentSpec(
        hypothesis="calendar boundary",
        robustness_tests=["subperiod"],
        start_date="2024-01-01",
        end_date="2025-12-31",
    )

    class Result:
        def __init__(self, run_id):
            self.run_id = run_id
            self.run_path = tmp_path / run_id
            self.run_path.mkdir()
            self.metrics = pd.DataFrame(
                [
                    {"metric": "cagr", "value": 0.1},
                    {"metric": "sharpe_ratio", "value": 1.0},
                    {"metric": "max_drawdown", "value": -0.1},
                ]
            )

    monkeypatch.setattr(
        "qss.research.orchestrator.validate_research_data",
        lambda *args, **kwargs: type("Validation", (), {"status": "valid"})(),
    )
    monkeypatch.setattr(
        "qss.research.orchestrator.load_backtest_data",
        lambda *args, **kwargs: object(),
    )
    calls = []

    def fake_run_backtest(start_date, end_date, *args, **kwargs):
        calls.append((start_date, end_date, kwargs.get("artifact_level", "full")))
        return Result(f"run-{len(calls)}")

    monkeypatch.setattr(
        "qss.research.orchestrator.run_backtest",
        fake_run_backtest,
    )

    context = __import__(
        "qss.research.orchestrator",
        fromlist=["ResearchOrchestrator"],
    ).ResearchOrchestrator(config).run(spec)

    assert context.manifest.status == "valid"
    assert [call[2] for call in calls] == ["full", "metrics", "metrics"]
