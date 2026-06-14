from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qss.config.loader import get_config
from qss.experiments.registry import ExperimentRegistry
from qss.ingestion.fama_french import _parse_daily_factors
from qss.research.decision import research_evidence_decision
from qss.research.orchestrator import (
    ExperimentSpec,
    _config_diff,
    _holdout_factor_diagnostics,
)
from qss.research.portfolio_evaluation import simulate_score_portfolio
from qss.research.protocol import ResearchProtocol, validate_label_gap
from qss.research.snapshot import build_data_snapshot
from qss.research.statistics import (
    benjamini_hochberg,
    block_bootstrap_summary,
    deflated_sharpe_probability,
    fama_french_style_regression,
    newey_west_mean_test,
)


def test_legacy_experiment_is_exploratory():
    spec = ExperimentSpec(
        hypothesis="legacy",
        start_date="2020-01-01",
        end_date="2025-01-01",
    )
    protocol = spec.research_protocol()
    assert protocol.stage == "exploratory"
    assert protocol.study_id.startswith("legacy-")


def test_confirmatory_protocol_requires_label_gap():
    protocol = ResearchProtocol(
        study_id="study-a",
        stage="confirmatory",
        development_start="2024-01-01",
        development_end="2024-03-29",
        holdout_start="2024-04-15",
        holdout_end="2024-12-31",
        trial_family="family-a",
    )
    calendar = pd.bdate_range("2024-01-01", "2024-12-31")
    with pytest.raises(ValueError, match="separated"):
        validate_label_gap(protocol, calendar, 21)
    protocol.holdout_start = "2024-05-06"
    validate_label_gap(protocol, calendar, 21)


def test_data_snapshot_is_stable_and_changes_with_input(tmp_path):
    config = get_config(["configs/default.yaml"])
    config.paths.silver_data = str(tmp_path / "silver")
    price = tmp_path / "silver" / "prices" / "prices_daily.parquet"
    price.parent.mkdir(parents=True)
    price.write_bytes(b"first")
    first = build_data_snapshot(config)
    second = build_data_snapshot(config)
    assert first["snapshot_id"] == second["snapshot_id"]
    archived = Path(first["files"][0]["archive_path"])
    assert archived.exists()
    assert archived.read_bytes() == b"first"
    price.write_bytes(b"second")
    changed = build_data_snapshot(config)
    assert changed["snapshot_id"] != first["snapshot_id"]


def test_data_snapshot_includes_macro_style_factors_and_environment(tmp_path):
    config = get_config(["configs/default.yaml"])
    config.paths.silver_data = str(tmp_path / "silver")
    config.research_validation.style_factor_cache = str(tmp_path / "fama_french")
    macro = tmp_path / "silver" / "macro" / "macro_observations.parquet"
    macro.parent.mkdir(parents=True)
    macro.write_bytes(b"macro")
    style = tmp_path / "fama_french" / "ff5_momentum_daily.parquet"
    style.parent.mkdir(parents=True)
    style.write_bytes(b"style")

    snapshot = build_data_snapshot(config)
    paths = {item["path"].replace("\\", "/") for item in snapshot["files"]}

    assert any(path.endswith("macro/macro_observations.parquet") for path in paths)
    assert any("fama_french" in path for path in paths)
    assert snapshot["environment"]["packages"]


def test_confirmatory_spec_requires_full_robustness_matrix():
    with pytest.raises(ValueError, match="robustness tests"):
        ExperimentSpec(
            hypothesis="confirmatory",
            study_id="study-a",
            research_stage="confirmatory",
            development_start="2023-01-01",
            development_end="2023-06-30",
            holdout_start="2023-08-01",
            holdout_end="2024-12-31",
            trial_family="family-a",
            robustness_tests=[],
            start_date="2023-01-01",
            end_date="2024-12-31",
        )


def test_confirmatory_factor_diagnostics_only_receive_holdout_rows(monkeypatch):
    captured = {}

    def fake_diagnostics(factors, prices):
        captured["factor_dates"] = pd.to_datetime(factors["date"])
        captured["price_dates"] = pd.to_datetime(prices["date"])
        return {
            "summary": pd.DataFrame(),
            "quantiles": pd.DataFrame(),
            "decay": pd.DataFrame(),
            "correlation": pd.DataFrame(),
        }

    monkeypatch.setattr(
        "qss.research.orchestrator.comprehensive_factor_diagnostics",
        fake_diagnostics,
    )
    factors = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2023-06-01", "2023-08-01", "2023-09-01"]
            ),
            "symbol": ["A", "A", "A"],
        }
    )
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2023-06-01", "2023-08-01", "2023-12-29", "2024-01-02"]
            ),
            "symbol": ["A"] * 4,
            "adj_close": [1.0, 1.1, 1.2, 1.3],
        }
    )

    _, scope = _holdout_factor_diagnostics(
        factors,
        prices,
        "2023-08-01",
        "2023-12-31",
    )

    assert captured["factor_dates"].min() == pd.Timestamp("2023-08-01")
    assert captured["factor_dates"].max() == pd.Timestamp("2023-09-01")
    assert captured["price_dates"].max() == pd.Timestamp("2023-12-29")
    assert scope["evaluation_scope"] == "holdout"


def test_registry_tracks_trials_and_snapshot_identity(tmp_path):
    registry = ExperimentRegistry(tmp_path / "registry.duckdb")
    registry.upsert(
        {
            "run_id": "experiment-1",
            "run_type": "experiment",
            "approval_status": "draft",
            "status": "valid",
            "trial_family": "family-a",
            "trial_number": 1,
            "spec_hash": "spec-a",
            "data_snapshot_id": "snapshot-a",
            "run_path": str(tmp_path),
            "created_at": "2026-01-01T00:00:00",
        }
    )
    assert registry.next_trial_number("family-a") == 2
    assert registry.trial_count("family-a") == 1
    assert registry.data_snapshot_for_spec("spec-a") == "snapshot-a"
    registry.upsert(
        {
            "run_id": "experiment-2",
            "run_type": "experiment",
            "approval_status": "draft",
            "status": "invalid",
            "trial_family": "family-a",
            "trial_number": 1,
            "spec_hash": "spec-b",
            "data_snapshot_id": "snapshot-a",
            "run_path": str(tmp_path),
            "created_at": "2026-01-02T00:00:00",
        }
    )
    assert registry.next_trial_number("family-a") == 3
    assert registry.trial_count("family-a") == 2


def test_hac_fdr_bootstrap_and_deflated_sharpe_are_deterministic():
    rng = np.random.default_rng(7)
    values = pd.Series(rng.normal(0.002, 0.01, size=240))
    mean_test = newey_west_mean_test(values)
    assert mean_test.t_stat > 0
    adjusted = benjamini_hochberg(pd.Series([0.01, 0.03, 0.20]))
    assert adjusted.tolist() == pytest.approx([0.03, 0.045, 0.20])
    daily = pd.DataFrame(
        {
            "portfolio_return": values,
            "benchmark_return": rng.normal(0.0005, 0.009, size=240),
        }
    )
    first = block_bootstrap_summary(
        daily,
        primary_metric="sharpe_ratio",
        samples=200,
        seed=11,
    )
    second = block_bootstrap_summary(
        daily,
        primary_metric="sharpe_ratio",
        samples=200,
        seed=11,
    )
    pd.testing.assert_frame_equal(first, second)
    one_trial = deflated_sharpe_probability(values, 1)
    many_trials = deflated_sharpe_probability(values, 50)
    assert many_trials["probability"] < one_trial["probability"]


def test_fama_french_regression_recovers_positive_market_loading():
    rng = np.random.default_rng(9)
    dates = pd.bdate_range("2024-01-01", periods=260)
    factors = pd.DataFrame(
        {
            "date": dates,
            "Mkt-RF": rng.normal(0.0004, 0.01, len(dates)),
            "SMB": rng.normal(0.0, 0.004, len(dates)),
            "HML": rng.normal(0.0, 0.004, len(dates)),
            "RMW": rng.normal(0.0, 0.003, len(dates)),
            "CMA": rng.normal(0.0, 0.003, len(dates)),
            "Mom": rng.normal(0.0, 0.005, len(dates)),
            "RF": 0.0001,
        }
    )
    portfolio = (
        factors["RF"]
        + 0.0002
        + 0.8 * factors["Mkt-RF"]
        + rng.normal(0.0, 0.002, len(dates))
    )
    exposures, summary = fama_french_style_regression(
        pd.DataFrame({"date": dates, "portfolio_return": portfolio}),
        factors,
    )
    market = exposures.set_index("factor").loc["Mkt-RF", "coefficient"]
    assert market == pytest.approx(0.8, abs=0.05)
    assert summary["r_squared"] > 0.8


def test_fama_french_parser_ignores_prose_blank_lines():
    frame = _parse_daily_factors(
        "description\n\n, Mom\n20240102, 1.25\n20240103, -0.50\n"
    )
    assert list(frame.columns) == ["date", "Mom"]
    assert frame["Mom"].tolist() == pytest.approx([0.0125, -0.005])


def _portfolio_fixture():
    dates = pd.bdate_range("2024-01-01", periods=100)
    rows = []
    returns = {
        "A": 0.001,
        "B": 0.0005,
        "C": -0.0002,
        "SPY": 0.0004,
        "^GSPC": 0.0004,
    }
    for symbol, daily_return in returns.items():
        prices = 100 * np.cumprod(np.repeat(1 + daily_return, len(dates)))
        for index, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "adj_close": prices[index],
                    "volume": 10_000_000,
                    "return_1d": 0.0 if index == 0 else daily_return,
                }
            )
    scores = pd.DataFrame(
        [
            {
                "date": dates[30],
                "symbol": symbol,
                "total_score": score,
                "sector": sector,
                "market_cap": 1e10,
            }
            for symbol, score, sector in [
                ("A", 3.0, "Tech"),
                ("B", 2.0, "Health"),
                ("C", 1.0, "Energy"),
            ]
        ]
        + [
            {
                "date": dates[60],
                "symbol": symbol,
                "total_score": score,
                "sector": sector,
                "market_cap": 1e10,
            }
            for symbol, score, sector in [
                ("A", 3.0, "Tech"),
                ("B", 2.0, "Health"),
                ("C", 1.0, "Energy"),
            ]
        ]
    )
    return pd.DataFrame(rows), scores, dates


def test_common_score_simulator_is_deterministic(tmp_path):
    prices, scores, dates = _portfolio_fixture()
    config = get_config(["configs/default.yaml"])
    config.runtime.research_mode = False
    config.optimizer.constraints.target_num_holdings = 2
    config.optimizer.constraints.max_weight = 0.75
    config.optimizer.constraints.max_sector_weight = 1.0
    config.optimizer.constraints.max_turnover_per_rebalance = 2.0
    config.optimizer.constraints.tracking_error_limit = None
    config.backtest.transaction_cost.commission_bps = 0.0
    config.backtest.transaction_cost.slippage_bps = 0.0
    config.backtest.transaction_cost.market_impact_coefficient = 0.0
    first = simulate_score_portfolio(
        scores,
        prices,
        config,
        start_date=str(dates[30].date()),
        end_date=str(dates[-1].date()),
        output_path=tmp_path / "first",
    )
    second = simulate_score_portfolio(
        scores.rename(columns={"total_score": "prediction"}).rename(
            columns={"prediction": "total_score"}
        ),
        prices,
        config,
        start_date=str(dates[30].date()),
        end_date=str(dates[-1].date()),
        output_path=tmp_path / "second",
    )
    pd.testing.assert_frame_equal(first.daily_returns, second.daily_returns)
    assert (tmp_path / "first" / "metrics.csv").exists()


def test_top_n_robustness_diff_only_changes_target_count():
    base = get_config(["configs/default.yaml"])
    variant = base.model_copy(deep=True)
    variant.optimizer.constraints.target_num_holdings = 30
    assert set(_config_diff(base, variant)) == {
        "optimizer.constraints.target_num_holdings"
    }


def test_research_decision_separates_artifacts_from_evidence():
    bootstrap = pd.DataFrame(
        [
            {
                "metric": "sharpe_ratio",
                "one_sided_lower_95": 0.2,
            }
        ]
    )
    supported = research_evidence_decision(
        stage="confirmatory",
        primary_metric="sharpe_ratio",
        threshold=0.0,
        bootstrap_summary=bootstrap,
        deflated_sharpe={"probability": 0.97},
        net_total_return=0.1,
        required_probability=0.95,
    )
    blocked = research_evidence_decision(
        stage="confirmatory",
        primary_metric="sharpe_ratio",
        threshold=0.0,
        bootstrap_summary=bootstrap,
        deflated_sharpe={"probability": 0.97},
        net_total_return=0.1,
        required_probability=0.95,
        blockers=["style regression missing"],
    )
    assert supported["status"] == "supported"
    assert blocked["status"] == "rejected"
