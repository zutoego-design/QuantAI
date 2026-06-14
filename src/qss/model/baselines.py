from __future__ import annotations

from typing import Any

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_baseline_model(model_type: str, parameters: dict[str, Any] | None = None):
    parameters = dict(parameters or {})
    if model_type == "ridge":
        estimator = Ridge(**({"alpha": 1.0} | parameters))
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", estimator),
            ]
        )
    if model_type == "elastic_net":
        estimator = ElasticNet(**({"alpha": 0.01, "l1_ratio": 0.5, "max_iter": 5000} | parameters))
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", estimator),
            ]
        )
    if model_type == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise RuntimeError(
                "The lightgbm baseline requires the project lightgbm dependency."
            ) from exc
        defaults = {
            "n_estimators": 100,
            "learning_rate": 0.05,
            "num_leaves": 15,
            "random_state": 42,
            "verbosity": -1,
        }
        return LGBMRegressor(**(defaults | parameters))
    if model_type == "hist_gradient_boosting":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        **({"max_iter": 100, "random_state": 42} | parameters)
                    ),
                ),
            ]
        )
    raise ValueError(f"Unsupported baseline model: {model_type}")
