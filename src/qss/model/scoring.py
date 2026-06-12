from __future__ import annotations

from pathlib import Path

import pandas as pd

from qss.config.schema import AppConfig, StrategyConfig
from qss.data.storage import append_or_replace_parquet, read_parquet
from qss.model.calibration import clip_score_extremes


def compute_alpha_scores(factor_values: pd.DataFrame, config: StrategyConfig | AppConfig) -> pd.DataFrame:
    app_config = config if isinstance(config, AppConfig) else None
    factor_groups = app_config.factor_groups if app_config is not None else None
    if factor_groups is None:
        raise TypeError("compute_alpha_scores requires AppConfig in this implementation.")
    if factor_values.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "symbol",
                "value_score",
                "quality_score",
                "momentum_score",
                "low_volatility_score",
                "total_score",
                "rank",
                "sector",
                "market_cap",
                "factor_coverage",
            ]
        )

    factor_values = factor_values.copy()
    pivot = factor_values.pivot_table(
        index=["date", "symbol", "sector", "market_cap"],
        columns="factor_name",
        values="processed_value",
        aggfunc="last",
    ).reset_index()

    group_scores = {
        "value_score": [],
        "quality_score": [],
        "momentum_score": [],
        "low_volatility_score": [],
    }
    group_weight_map: dict[str, float] = {}
    factor_weight_total = 0.0
    weighted_coverage = pd.Series(0.0, index=pivot.index)

    for group_name, group_config in factor_groups.items():
        column_name = f"{group_name}_score"
        factor_weights = {name: definition.weight for name, definition in group_config.factors.items()}
        available = [name for name in factor_weights if name in pivot.columns]
        factor_weight_total += sum(factor_weights.values()) * group_config.weight
        if available:
            numerator = sum(
                pivot[name].fillna(0.0) * factor_weights[name] for name in available
            )
            denominator = sum(
                pivot[name].notna().astype(float) * factor_weights[name] for name in available
            )
            pivot[column_name] = numerator / denominator.replace(0.0, pd.NA)
            weighted_coverage += denominator * group_config.weight
        else:
            pivot[column_name] = pd.NA
        group_weight_map[column_name] = group_config.weight

    score_cols = list(group_weight_map)
    pivot = clip_score_extremes(pivot, score_cols)
    group_numerator = sum(
        pivot[col].fillna(0.0) * group_weight_map[col] for col in score_cols
    )
    group_denominator = sum(
        pivot[col].notna().astype(float) * group_weight_map[col] for col in score_cols
    )
    pivot["total_score"] = group_numerator / group_denominator.replace(0.0, pd.NA)
    pivot["factor_coverage"] = (
        weighted_coverage / factor_weight_total if factor_weight_total else 0.0
    )
    pivot = pivot.loc[
        pivot["factor_coverage"] >= app_config.strategy.min_factor_coverage
    ].copy()
    pivot["rank"] = pivot.groupby("date")["total_score"].rank(method="first", ascending=False).astype(int)
    for column in group_scores:
        if column not in pivot:
            pivot[column] = pd.NA
    return pivot[
        [
            "date",
            "symbol",
            "value_score",
            "quality_score",
            "momentum_score",
            "low_volatility_score",
            "total_score",
            "rank",
            "sector",
            "market_cap",
            "factor_coverage",
        ]
    ].sort_values(["date", "rank"])


def compute_and_store_scores(as_of_date: pd.Timestamp, config: AppConfig) -> pd.DataFrame:
    factor_values = read_parquet(Path(config.paths.gold_data) / "factors" / "factor_values.parquet")
    scores = compute_alpha_scores(factor_values.loc[factor_values["date"] == pd.Timestamp(as_of_date)], config)
    append_or_replace_parquet(scores, Path(config.paths.gold_data) / "scores" / "alpha_scores.parquet", ["date", "symbol"])
    return scores
