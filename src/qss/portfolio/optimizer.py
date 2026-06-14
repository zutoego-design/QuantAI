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


def build_equal_weight_portfolio(
    scores: pd.DataFrame,
    top_n: int,
    *,
    max_sector_weight: float | None = None,
) -> pd.DataFrame:
    ranked = scores.sort_values("total_score", ascending=False)
    if max_sector_weight is None:
        candidates = ranked.head(top_n).copy()
    else:
        max_names_per_sector = max(
            1,
            int(np.floor(max_sector_weight * top_n + 1e-12)),
        )
        selected: list[int] = []
        sector_counts: dict[str, int] = {}
        for index, row in ranked.iterrows():
            sector = str(row.get("sector", "")).strip()
            constrained = sector.lower() not in {
                "",
                "unknown",
                "unclassified",
                "nan",
            }
            if (
                constrained
                and sector_counts.get(sector, 0) >= max_names_per_sector
            ):
                continue
            selected.append(index)
            if constrained:
                sector_counts[sector] = sector_counts.get(sector, 0) + 1
            if len(selected) == top_n:
                break
        candidates = ranked.loc[selected].copy()
        if len(candidates) < min(top_n, len(ranked)):
            raise ValueError(
                f"Only {len(candidates)} of {top_n} requested holdings can satisfy "
                f"the {max_sector_weight:.2%} sector limit."
            )
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


def optimize_portfolio_to_target_count(
    scores: pd.DataFrame,
    covariance: pd.DataFrame,
    previous_weights: pd.Series,
    sector_map: pd.Series,
    config: OptimizerConfig,
) -> OptimizationResult:
    broad = optimize_portfolio_with_status(
        scores,
        covariance,
        previous_weights,
        sector_map,
        config,
    )
    target = min(config.constraints.target_num_holdings, len(scores))
    if broad.status == "fallback" or len(broad.weights) == target:
        return broad

    selected_order = broad.weights.sort_values(
        "target_weight",
        ascending=False,
    )["symbol"].astype(str).tolist()
    selected_order.extend(
        scores.sort_values("total_score", ascending=False)["symbol"]
        .astype(str)
        .tolist()
    )
    selected = set(list(dict.fromkeys(selected_order))[:target])
    restricted_scores = scores.loc[
        scores["symbol"].astype(str).isin(selected)
    ].copy()
    exact_config = config.model_copy(deep=True)
    exact_config.candidate_count = target
    exact_config.fallback.top_n = target
    exact_config.constraints.min_weight = max(
        exact_config.constraints.min_weight,
        1e-4,
    )
    exact = optimize_portfolio_with_status(
        restricted_scores,
        covariance,
        previous_weights,
        sector_map,
        exact_config,
    )
    if exact.status == "fallback" or len(exact.weights) != target:
        return OptimizationResult(
            weights=exact.weights,
            status="fallback",
            warning=(
                f"Exact-cardinality optimization produced {len(exact.weights)} "
                f"holdings; {target} are required. {exact.warning or ''}"
            ).strip(),
        )
    exact.status = f"{exact.status}_target_cardinality"
    return exact


def optimize_portfolio_with_status(
    scores: pd.DataFrame,
    covariance: pd.DataFrame,
    previous_weights: pd.Series,
    sector_map: pd.Series,
    config: OptimizerConfig,
) -> OptimizationResult:
    target_holdings = min(config.constraints.target_num_holdings, len(scores))
    candidate_count = min(
        max(int(config.candidate_count), target_holdings),
        len(scores),
    )
    ranked = scores.sort_values("total_score", ascending=False)
    candidates = ranked.head(candidate_count).copy()
    existing_symbols = set(
        previous_weights.loc[previous_weights.abs() > 1e-8].index.astype(str)
    )
    if existing_symbols:
        held_candidates = ranked.loc[ranked["symbol"].astype(str).isin(existing_symbols)]
        candidates = (
            pd.concat([candidates, held_candidates], ignore_index=True)
            .drop_duplicates("symbol", keep="first")
            .copy()
        )
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
        minimum_holdings = min(target_holdings, max(1, int(target_holdings * 0.5)))
        if len(portfolio) < minimum_holdings:
            raise ValueError(
                f"Optimizer produced {len(portfolio)} holdings; "
                f"minimum acceptable is {minimum_holdings}."
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
