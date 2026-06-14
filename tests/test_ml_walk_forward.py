import numpy as np
import pandas as pd

from qss.config.loader import get_config
from qss.model.baselines import build_baseline_model
from qss.model.evaluation import evaluate_walk_forward
from qss.research.walk_forward import walk_forward_splits


def _dataset():
    dates = pd.date_range("2022-01-31", periods=24, freq="ME")
    factor_rows = []
    label_rows = []
    for date_index, date in enumerate(dates):
        for symbol_index, symbol in enumerate(["A", "B", "C", "D"]):
            value = float(date_index + symbol_index)
            factor_rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "factor_name": "test_factor",
                    "processed_value": value,
                }
            )
            common = {
                "date": date,
                "symbol": symbol,
                "label_start_time": date,
                "label_end_time": date + pd.Timedelta(days=40),
                "overlap": False,
                "purge_required": False,
                "horizon": 21,
            }
            label_rows.extend(
                [
                    {
                        **common,
                        "label_name": "cross_sectional_rank",
                        "label_value": (symbol_index + 1) / 4,
                    },
                    {
                        **common,
                        "label_name": "forward_return",
                        "label_value": 0.01 * (symbol_index + 1),
                    },
                ]
            )
    return pd.DataFrame(factor_rows), pd.DataFrame(label_rows)


def test_walk_forward_purges_overlapping_training_labels():
    config = get_config(["configs/default.yaml"]).ml.walk_forward
    config.train_periods = 12
    config.min_train_periods = 6
    config.test_periods = 2
    config.step_periods = 2
    config.embargo_days = 5
    _, labels = _dataset()
    folds = walk_forward_splits(labels, config)
    assert folds
    assert any(fold.purged_rows > 0 for fold, _, _ in folds)
    for fold, train_index, _ in folds:
        train = labels.loc[train_index]
        assert (
            pd.to_datetime(train["label_end_time"])
            < fold.test_start - pd.Timedelta(days=config.embargo_days)
        ).all()


def test_ridge_baseline_reports_fold_and_aggregate_metrics(tmp_path):
    factors, labels = _dataset()
    config = get_config(["configs/default.yaml"]).ml
    config.enabled = True
    config.model_type = "ridge"
    config.walk_forward.train_periods = 12
    config.walk_forward.min_train_periods = 6
    config.walk_forward.test_periods = 2
    config.walk_forward.step_periods = 2
    result = evaluate_walk_forward(factors, labels, config, tmp_path)
    assert not result["fold_metrics"].empty
    assert result["aggregate_metrics"].iloc[0]["folds"] > 0
    assert np.isfinite(result["fold_metrics"]["mae"]).all()
    assert result["portfolio_metrics"].iloc[0]["periods"] > 0
    assert {
        "gross_return",
        "transaction_cost",
        "net_return",
    }.issubset(result["portfolio_period_returns"].columns)
    assert (tmp_path / "split_manifest.csv").exists()
    assert (tmp_path / "model_config.json").exists()


def test_lightgbm_tree_baseline_is_available():
    model = build_baseline_model(
        "lightgbm",
        {"n_estimators": 5, "num_leaves": 4},
    )
    features = pd.DataFrame({"factor": [0.0, 1.0, 2.0, 3.0]})
    target = pd.Series([0.0, 0.2, 0.8, 1.0])
    model.fit(features, target)
    predictions = model.predict(features)
    assert len(predictions) == len(target)
