from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd

from qss.config.schema import OptimizerConfig
from qss.logging_utils import logger
from qss.portfolio.constraints import build_sector_masks, validate_weights


@dataclass
class OptimizationResult:
    weights: pd.DataFrame
    status: str
    warning: str | None = None


def build_equal_weight_portfolio(scores: pd.DataFrame, top_n: int) -> pd.DataFrame:
    candidates = scores.sort_values("total_score", ascending=False).head(top_n).copy()
    if candidates.empty:
        return pd.DataFrame(columns=["symbol", "target_weight", "previous_weight", "trade_weight", "sector", "alpha_score"])
    weight = 1.0 / len(candidates)
    candidates["target_weight"] = weight
    candidates["previous_weight"] = 0.0
    candidates["trade_weight"] = candidates["target_weight"]
    candidates["alpha_score"] = candidates["total_score"]
    return candidates[["symbol", "target_weight", "previous_weight", "trade_weight", "sector", "alpha_score"]]


def optimize_portfolio(
    scores: pd.DataFrame,
    covariance: pd.DataFrame,
    previous_weights: pd.Series,
    sector_map: pd.Series,
    config: OptimizerConfig,
) -> pd.DataFrame:
    result = optimize_portfolio_with_status(scores, covariance, previous_weights, sector_map, config)
    return result.weights


def optimize_portfolio_with_status(
    scores: pd.DataFrame,
    covariance: pd.DataFrame,
    previous_weights: pd.Series,
    sector_map: pd.Series,
    config: OptimizerConfig,
) -> OptimizationResult:
    target_holdings = min(config.constraints.target_num_holdings, len(scores))
    candidates = scores.sort_values("total_score", ascending=False).head(target_holdings).copy()
    if candidates.empty:
        return OptimizationResult(weights=build_equal_weight_portfolio(scores, config.fallback.top_n), status="fallback", warning="No candidates available.")

    symbols = candidates["symbol"].tolist()
    sigma = covariance.reindex(index=symbols, columns=symbols).fillna(0.0).values
    sigma = sigma + np.eye(len(symbols)) * 1e-6
    alpha = candidates["total_score"].values.astype(float)
    previous = previous_weights.reindex(symbols).fillna(0.0).values.astype(float)
    outside_turnover = float(
        previous_weights.loc[~previous_weights.index.isin(symbols)].abs().sum()
    )
    sector_masks = build_sector_masks(symbols, sector_map)
    benchmark = np.repeat(1.0 / len(symbols), len(symbols))

    w = cp.Variable(len(symbols))
    risk = cp.quad_form(w, sigma)
    turnover = cp.norm1(w - previous) + outside_turnover
    objective = cp.Maximize(alpha @ w - config.objective.risk_aversion * risk - config.objective.turnover_penalty * turnover)

    constraints = [cp.sum(w) == 1]
    if config.constraints.long_only:
        minimum_position = max(config.constraints.min_weight, 1e-8)
        constraints.append(w >= minimum_position)
    constraints.append(w <= config.constraints.max_weight)
    if config.constraints.max_turnover_per_rebalance is not None and previous.sum() > 1e-8:
        constraints.append(turnover <= config.constraints.max_turnover_per_rebalance)
    for mask in sector_masks.values():
        constraints.append(mask @ w <= config.constraints.max_sector_weight)
    if config.constraints.tracking_error_limit is not None:
        tracking_error = cp.quad_form(w - benchmark, sigma)
        constraints.append(tracking_error <= (config.constraints.tracking_error_limit**2) / 252)

    problem = cp.Problem(objective, constraints)
    try:
        solved = False
        for solver in (cp.CLARABEL, cp.SCS):
            try:
                problem.solve(solver=solver, verbose=False)
                if problem.status in {"optimal", "optimal_inaccurate"} and w.value is not None:
                    solved = True
                    break
            except Exception:
                continue
        if not solved or w.value is None:
            raise ValueError(problem.status)
        weights = np.asarray(w.value).reshape(-1)
        portfolio = pd.DataFrame(
            {
                "symbol": symbols,
                "target_weight": weights,
                "previous_weight": previous,
                "trade_weight": weights - previous,
                "sector": sector_map.reindex(symbols).values,
                "alpha_score": alpha,
            }
        )
        portfolio["target_weight"] = portfolio["target_weight"].clip(lower=0.0)
        portfolio["target_weight"] = portfolio["target_weight"] / portfolio["target_weight"].sum()
        portfolio = portfolio.loc[portfolio["target_weight"] > 1e-8].copy()
        if len(portfolio) != target_holdings:
            raise ValueError(
                f"Optimizer produced {len(portfolio)} holdings; expected {target_holdings}."
            )
        portfolio["trade_weight"] = portfolio["target_weight"] - portfolio["previous_weight"]
        validate_weights(portfolio, config.constraints.max_weight, config.constraints.max_sector_weight)
        return OptimizationResult(weights=portfolio, status=str(problem.status))
    except Exception as exc:
        logger.warning("Optimizer failed: {}", exc)
        fallback = build_equal_weight_portfolio(scores, min(config.fallback.top_n, len(scores)))
        fallback["previous_weight"] = previous_weights.reindex(fallback["symbol"]).fillna(0.0).values
        fallback["trade_weight"] = fallback["target_weight"] - fallback["previous_weight"]
        return OptimizationResult(weights=fallback, status="fallback", warning=str(exc))
