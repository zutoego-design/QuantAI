from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from qss.config.schema import MLConfig
from qss.model.baselines import build_baseline_model
from qss.research.walk_forward import walk_forward_splits


def build_model_dataset(
    factor_values: pd.DataFrame,
    labels: pd.DataFrame,
    label_name: str,
) -> tuple[pd.DataFrame, list[str]]:
    selected = labels.loc[labels["label_name"] == label_name].copy()
    feature_frame = factor_values.pivot_table(
        index=["date", "symbol"],
        columns="factor_name",
        values="processed_value",
        aggfunc="last",
    ).reset_index()
    feature_columns = [
        column for column in feature_frame.columns if column not in {"date", "symbol"}
    ]
    dataset = feature_frame.merge(
        selected[
            [
                "date",
                "symbol",
                "label_value",
                "label_start_time",
                "label_end_time",
                "overlap",
                "purge_required",
            ]
        ],
        on=["date", "symbol"],
        how="inner",
    ).dropna(subset=["label_value"])
    return dataset.sort_values(["date", "symbol"]).reset_index(drop=True), feature_columns


def fit_holdout_predictions(
    factor_values: pd.DataFrame,
    labels: pd.DataFrame,
    config: MLConfig,
    *,
    development_end: str,
    holdout_start: str,
    holdout_end: str,
) -> tuple[pd.DataFrame, dict]:
    dataset, feature_columns = build_model_dataset(
        factor_values,
        labels,
        config.target,
    )
    development_end_time = pd.Timestamp(development_end)
    holdout_start_time = pd.Timestamp(holdout_start)
    holdout_end_time = pd.Timestamp(holdout_end)
    embargo_cutoff = holdout_start_time - pd.Timedelta(
        days=config.walk_forward.embargo_days
    )
    train = dataset.loc[
        (pd.to_datetime(dataset["date"]) <= development_end_time)
        & (pd.to_datetime(dataset["label_end_time"]) < embargo_cutoff)
    ].copy()
    test = dataset.loc[
        pd.to_datetime(dataset["date"]).between(
            holdout_start_time,
            holdout_end_time,
        )
    ].copy()
    if train.empty or test.empty:
        raise ValueError("Holdout model evaluation requires non-empty train and test data.")
    model = build_baseline_model(config.model_type, config.parameters)
    model.fit(train[feature_columns], train["label_value"])
    predictions = model.predict(test[feature_columns])
    result = test[["date", "symbol", "label_value"]].copy()
    result["prediction"] = predictions
    rank_ic = result.groupby("date").apply(
        lambda cross: cross["prediction"].corr(
            cross["label_value"],
            method="spearman",
        ),
        include_groups=False,
    )
    metadata = {
        "model_type": config.model_type,
        "train_start": str(pd.to_datetime(train["date"]).min().date()),
        "train_end": str(pd.to_datetime(train["date"]).max().date()),
        "test_start": str(pd.to_datetime(test["date"]).min().date()),
        "test_end": str(pd.to_datetime(test["date"]).max().date()),
        "train_rows": len(train),
        "test_rows": len(test),
        "purged_rows": int(
            (
                (pd.to_datetime(dataset["date"]) <= development_end_time)
                & (pd.to_datetime(dataset["label_end_time"]) >= embargo_cutoff)
            ).sum()
        ),
        "mean_rank_ic": float(rank_ic.mean()),
        "feature_columns": feature_columns,
    }
    return result, metadata


def evaluate_walk_forward(
    factor_values: pd.DataFrame,
    labels: pd.DataFrame,
    config: MLConfig,
    output_path: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    dataset, feature_columns = build_model_dataset(factor_values, labels, config.target)
    folds = walk_forward_splits(dataset, config.walk_forward)
    fold_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    split_rows: list[dict] = []
    for fold, train_index, test_index in folds:
        train = dataset.loc[train_index]
        test = dataset.loc[test_index]
        model = build_baseline_model(config.model_type, config.parameters)
        model.fit(train[feature_columns], train["label_value"])
        predictions = model.predict(test[feature_columns])
        rank_ic = pd.Series(predictions).corr(
            test["label_value"].reset_index(drop=True),
            method="spearman",
        )
        fold_rows.append(
            {
                "fold": fold.fold,
                "model_type": config.model_type,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "train_rows": len(train),
                "test_rows": len(test),
                "purged_rows": fold.purged_rows,
                "mse": mean_squared_error(test["label_value"], predictions),
                "mae": mean_absolute_error(test["label_value"], predictions),
                "rank_ic": rank_ic,
            }
        )
        prediction = test[["date", "symbol", "label_value"]].copy()
        prediction["prediction"] = predictions
        prediction["fold"] = fold.fold
        prediction_frames.append(prediction)
        for role, dates in [("train", fold.train_dates), ("test", fold.test_dates)]:
            split_rows.extend(
                {"fold": fold.fold, "role": role, "date": date} for date in dates
            )
    fold_metrics = pd.DataFrame(fold_rows)
    aggregate = pd.DataFrame(
        [
            {
                "model_type": config.model_type,
                "folds": len(fold_metrics),
                "mean_mse": float(fold_metrics["mse"].mean()) if not fold_metrics.empty else np.nan,
                "mean_mae": float(fold_metrics["mae"].mean()) if not fold_metrics.empty else np.nan,
                "mean_rank_ic": (
                    float(fold_metrics["rank_ic"].mean()) if not fold_metrics.empty else np.nan
                ),
            }
        ]
    )
    predictions = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame(columns=["date", "symbol", "label_value", "prediction", "fold"])
    )
    forward = labels.loc[
        labels["label_name"] == "forward_return",
        ["date", "symbol", "label_value", "horizon"],
    ].rename(columns={"label_value": "realized_forward_return"})
    mapped = predictions.merge(forward, on=["date", "symbol"], how="left")
    portfolio_rows: list[dict] = []
    previous_weights = pd.Series(dtype=float)
    for date, cross_section in mapped.groupby("date"):
        selected = cross_section.dropna(subset=["realized_forward_return"]).nlargest(
            config.portfolio_top_n,
            "prediction",
        )
        if selected.empty:
            continue
        weights = pd.Series(
            1.0 / len(selected),
            index=selected["symbol"],
            dtype=float,
        )
        union = previous_weights.index.union(weights.index)
        turnover = float(
            (
                weights.reindex(union, fill_value=0.0)
                - previous_weights.reindex(union, fill_value=0.0)
            ).abs().sum()
            / 2.0
        )
        gross_return = float(selected["realized_forward_return"].mean())
        cost = turnover * config.transaction_cost_bps / 10_000.0
        portfolio_rows.append(
            {
                "date": date,
                "holding_count": len(selected),
                "turnover": turnover,
                "gross_return": gross_return,
                "transaction_cost": cost,
                "net_return": gross_return - cost,
            }
        )
        previous_weights = weights
    portfolio_returns = pd.DataFrame(portfolio_rows)
    if portfolio_returns.empty:
        portfolio_metrics = pd.DataFrame(
            [
                {
                    "model_type": config.model_type,
                    "periods": 0,
                    "gross_total_return": np.nan,
                    "net_total_return": np.nan,
                    "net_sharpe": np.nan,
                    "average_turnover": np.nan,
                    "transaction_cost_bps": config.transaction_cost_bps,
                }
            ]
        )
    else:
        horizon = (
            float(forward["horizon"].dropna().median())
            if "horizon" in forward and not forward["horizon"].dropna().empty
            else 21.0
        )
        annualization = np.sqrt(252.0 / max(horizon, 1.0))
        net_std = float(portfolio_returns["net_return"].std(ddof=0))
        portfolio_metrics = pd.DataFrame(
            [
                {
                    "model_type": config.model_type,
                    "periods": len(portfolio_returns),
                    "gross_total_return": float(
                        (1.0 + portfolio_returns["gross_return"]).prod() - 1.0
                    ),
                    "net_total_return": float(
                        (1.0 + portfolio_returns["net_return"]).prod() - 1.0
                    ),
                    "net_sharpe": (
                        float(portfolio_returns["net_return"].mean() / net_std * annualization)
                        if net_std > 0
                        else np.nan
                    ),
                    "average_turnover": float(portfolio_returns["turnover"].mean()),
                    "transaction_cost_bps": config.transaction_cost_bps,
                }
            ]
        )
    result = {
        "fold_metrics": fold_metrics,
        "aggregate_metrics": aggregate,
        "predictions": predictions,
        "split_manifest": pd.DataFrame(split_rows),
        "portfolio_period_returns": portfolio_returns,
        "portfolio_metrics": portfolio_metrics,
    }
    if output_path is not None:
        root = Path(output_path)
        root.mkdir(parents=True, exist_ok=True)
        for name, frame in result.items():
            frame.to_csv(root / f"{name}.csv", index=False)
        (root / "model_config.json").write_text(
            json.dumps(config.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )
    return result
