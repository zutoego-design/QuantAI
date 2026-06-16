import numpy as np
import pandas as pd

from qss.config.loader import get_config
from qss.portfolio.optimizer import (
    optimize_portfolio_to_target_count,
    optimize_portfolio_with_status,
)


def _sample_scores(count: int = 25) -> pd.DataFrame:
    sectors = ["Tech", "Health", "Energy", "Consumer", "Industrial"]
    rows = []
    for idx in range(count):
        rows.append(
            {
                "date": pd.Timestamp("2025-12-31"),
                "symbol": f"S{idx:02d}",
                "total_score": 1 - idx * 0.01,
                "sector": sectors[idx % len(sectors)],
                "market_cap": 1_000_000_000 + idx * 10_000_000,
            }
        )
    return pd.DataFrame(rows)


def test_optimizer_constraints_respected():
    config = get_config(["configs/default.yaml"]).optimizer
    scores = _sample_scores()
    covariance = pd.DataFrame(np.eye(len(scores)) * 0.0001, index=scores["symbol"], columns=scores["symbol"])
    result = optimize_portfolio_with_status(
        scores=scores,
        covariance=covariance,
        previous_weights=pd.Series(0.0, index=scores["symbol"]),
        sector_map=scores.set_index("symbol")["sector"],
        config=config,
    )
    weights = result.weights
    assert abs(weights["target_weight"].sum() - 1.0) < 1e-6
    assert (weights["target_weight"] >= -1e-8).all()
    assert weights["target_weight"].max() <= config.constraints.max_weight + 1e-6
    assert weights.groupby("sector")["target_weight"].sum().max() <= config.constraints.max_sector_weight + 1e-6
    assert len(weights) == min(config.constraints.target_num_holdings, len(scores))


def test_optimizer_fallback_when_turnover_constraint_is_impossible():
    config = get_config(["configs/default.yaml"]).optimizer
    config.constraints.max_turnover_per_rebalance = 0.01
    scores = _sample_scores()
    covariance = pd.DataFrame(np.eye(len(scores)) * 0.0001, index=scores["symbol"], columns=scores["symbol"])
    previous = pd.Series(0.0, index=scores["symbol"])
    previous.iloc[0] = 1.0
    result = optimize_portfolio_with_status(
        scores=scores,
        covariance=covariance,
        previous_weights=previous,
        sector_map=scores.set_index("symbol")["sector"],
        config=config,
    )
    assert result.status == "fallback"
    assert abs(result.weights["target_weight"].sum() - 1.0) < 1e-6


def test_optimizer_counts_positions_outside_new_candidate_set_as_turnover():
    config = get_config(["configs/default.yaml"]).optimizer
    config.constraints.max_turnover_per_rebalance = 0.30
    scores = _sample_scores()
    covariance = pd.DataFrame(
        np.eye(len(scores)) * 0.0001,
        index=scores["symbol"],
        columns=scores["symbol"],
    )
    previous = pd.Series({"OUT": 0.50, "S00": 0.50})
    result = optimize_portfolio_with_status(
        scores=scores,
        covariance=covariance,
        previous_weights=previous,
        sector_map=scores.set_index("symbol")["sector"],
        config=config,
    )
    assert result.status == "fallback"


def test_operational_optimizer_enforces_exact_target_count():
    config = get_config(["configs/default.yaml"]).optimizer
    scores = _sample_scores(100)
    covariance = pd.DataFrame(
        np.eye(len(scores)) * 0.0001,
        index=scores["symbol"],
        columns=scores["symbol"],
    )

    result = optimize_portfolio_to_target_count(
        scores=scores,
        covariance=covariance,
        previous_weights=pd.Series(dtype=float),
        sector_map=scores.set_index("symbol")["sector"],
        config=config,
    )

    assert result.status != "fallback"
    assert len(result.weights) == config.constraints.target_num_holdings
    assert result.weights["target_weight"].min() > 0
    assert (
        result.weights.groupby("sector")["target_weight"].sum().max()
        <= config.constraints.max_sector_weight + 1e-6
    )


def test_exact_target_count_runs_after_sparse_broad_fallback():
    config = get_config(["configs/default.yaml"]).optimizer
    config.constraints.target_num_holdings = 100
    scores = _sample_scores(100)
    covariance = pd.DataFrame(
        np.eye(len(scores)) * 0.0001,
        index=scores["symbol"],
        columns=scores["symbol"],
    )

    broad = optimize_portfolio_with_status(
        scores=scores,
        covariance=covariance,
        previous_weights=pd.Series(dtype=float),
        sector_map=scores.set_index("symbol")["sector"],
        config=config,
    )
    result = optimize_portfolio_to_target_count(
        scores=scores,
        covariance=covariance,
        previous_weights=pd.Series(dtype=float),
        sector_map=scores.set_index("symbol")["sector"],
        config=config,
    )

    assert broad.status == "fallback"
    assert len(broad.weights) < config.constraints.target_num_holdings
    assert result.status != "fallback"
    assert len(result.weights) == config.constraints.target_num_holdings
    assert result.weights["target_weight"].min() > 0
