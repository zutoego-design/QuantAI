import json

import pandas as pd

from qss.config.loader import get_config
from qss.research.historical_replay import (
    CONTROL_ID,
    SelectionRules,
    _selection_status,
    load_candidate_definition,
    load_replay_suite,
)


def test_replay_suite_has_seven_non_overlapping_annual_folds():
    suite = load_replay_suite("configs/historical_replay/suite.yaml")
    assert len(suite.folds) == 7
    assert [fold.fold_id for fold in suite.folds] == [
        "2019",
        "2020",
        "2021",
        "2022",
        "2023",
        "2024",
        "2025",
    ]
    test_dates = []
    for fold in suite.folds:
        assert pd.Timestamp(fold.development_end) < pd.Timestamp(fold.test_start)
        test_dates.extend(pd.date_range(fold.test_start, fold.test_end))
    assert len(test_dates) == len(set(test_dates))


def test_candidate_configs_freeze_expected_factor_semantics():
    base = get_config(["configs/default.yaml"])
    control = load_candidate_definition(
        base,
        "configs/historical_replay/v1_control.yaml",
    )
    fixed = load_candidate_definition(
        base,
        "configs/historical_replay/v1_drawdown_fixed.yaml",
    )
    core = load_candidate_definition(
        base,
        "configs/historical_replay/v2_core.yaml",
    )
    defensive = load_candidate_definition(
        base,
        "configs/historical_replay/v2_core_defensive.yaml",
    )

    assert control.candidate_id == CONTROL_ID
    assert (
        control.config.factor_groups["low_volatility"]
        .factors["max_drawdown_252d"]
        .direction
        == -1
    )
    assert (
        fixed.config.factor_groups["low_volatility"]
        .factors["max_drawdown_252d"]
        .direction
        == 1
    )
    assert set(core.config.factor_groups) == {"value", "momentum"}
    assert set(core.config.factor_groups["value"].factors) == {
        "sales_yield",
        "earnings_yield",
    }
    assert set(core.config.factor_groups["momentum"].factors) == {
        "momentum_3m",
        "momentum_6m",
    }
    assert core.config.factor_groups["value"].weight == 0.55
    assert core.config.factor_groups["momentum"].weight == 0.45
    assert defensive.config.optimizer.constraints.max_weight == 0.04
    assert defensive.config.optimizer.constraints.max_sector_weight == 0.20
    assert defensive.config.optimizer.constraints.max_turnover_per_rebalance == 0.20
    assert defensive.config.optimizer.constraints.tracking_error_limit == 0.08


def test_selection_rules_require_performance_cost_and_robustness():
    rules = SelectionRules()
    control = {
        "combined_sharpe": 1.0,
        "combined_max_drawdown": -0.15,
    }
    summary = {
        "all_folds_valid": True,
        "positive_years": 6,
        "spy_outperformance_years": 5,
        "combined_sharpe": 1.2,
        "combined_max_drawdown": -0.16,
    }
    robustness = pd.DataFrame(
        [
            {
                "test": "cost_sensitivity",
                "setting": "25.0",
                "status": "valid",
                "net_total_return": 0.25,
                "sharpe_decline": 0.10,
            },
            {
                "test": "top_n_sensitivity",
                "setting": "30",
                "status": "valid",
                "net_total_return": 0.20,
                "sharpe_decline": 0.20,
            },
            {
                "test": "rebalance_day_shift",
                "setting": "-5",
                "status": "valid",
                "net_total_return": 0.18,
                "sharpe_decline": 0.25,
            },
        ]
    )
    passed, failures = _selection_status(summary, control, robustness, rules)
    assert passed
    assert failures == []

    failed_robustness = robustness.copy()
    failed_robustness.loc[
        failed_robustness["test"] == "rebalance_day_shift",
        "sharpe_decline",
    ] = 0.31
    passed, failures = _selection_status(
        summary,
        control,
        failed_robustness,
        rules,
    )
    assert not passed
    assert json.loads(json.dumps(failures)) == ["robustness_decline_or_invalid"]
