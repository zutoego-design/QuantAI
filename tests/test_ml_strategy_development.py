import pandas as pd

from qss.research.ml_strategy_development import (
    BEST_VALUE_LOW_RISK_FACTORS,
    residualize_by_date,
    style_neutralized_score_frame,
    style_residual_rank_labels,
)


def _factor_rows(date, symbols):
    rows = []
    sectors = {"A": "Tech", "B": "Tech", "C": "Health", "D": "Health"}
    exposures = {"A": 2.0, "B": 1.0, "C": -1.0, "D": -2.0}
    for symbol in symbols:
        for factor in BEST_VALUE_LOW_RISK_FACTORS:
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "factor_name": factor,
                    "processed_value": exposures[symbol] if factor == "beta_to_spy" else 0.1,
                    "sector": sectors[symbol],
                    "market_cap": 1_000_000.0,
                }
            )
    return rows


def test_residualize_by_date_removes_same_date_exposure():
    frame = pd.DataFrame(
        {
            "date": [pd.Timestamp("2025-01-31")] * 4,
            "symbol": ["A", "B", "C", "D"],
            "target": [0.04, 0.02, -0.02, -0.04],
            "beta_to_spy": [2.0, 1.0, -1.0, -2.0],
            "sector": ["Tech", "Tech", "Health", "Health"],
        }
    )

    result = residualize_by_date(
        frame,
        value_column="target",
        exposure_columns=["beta_to_spy"],
        include_sector=False,
    )

    assert result["residual_value"].abs().max() < 1e-10


def test_style_residual_rank_labels_preserve_label_timing():
    dates = pd.date_range("2025-01-31", periods=3, freq="ME")
    symbols = ["A", "B", "C", "D"]
    factor_values = pd.DataFrame(
        [row for date in dates for row in _factor_rows(date, symbols)]
    )
    labels = []
    for date in dates:
        for idx, symbol in enumerate(symbols):
            labels.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "label_name": "forward_return",
                    "label_value": idx / 100.0,
                    "horizon": 21,
                    "label_start_time": date,
                    "label_end_time": date + pd.Timedelta(days=21),
                    "overlap": False,
                    "purge_required": True,
                    "embargo_days": 5,
                    "version": "v1",
                }
            )

    result = style_residual_rank_labels(factor_values, pd.DataFrame(labels))

    assert set(result["label_name"]) == {"cross_sectional_rank"}
    assert result["label_value"].between(0, 1).all()
    assert (pd.to_datetime(result["label_end_time"]) > pd.to_datetime(result["date"])).all()
    assert set(result["version"]) == {"exploratory_style_residual_rank_v1"}


def test_style_neutralized_score_frame_reduces_linear_exposure():
    date = pd.Timestamp("2025-01-31")
    symbols = ["A", "B", "C", "D"]
    factor_values = pd.DataFrame(_factor_rows(date, symbols))
    predictions = pd.DataFrame(
        {
            "date": [date] * 4,
            "symbol": symbols,
            "prediction": [2.0, 1.0, -1.0, -2.0],
        }
    )

    result = style_neutralized_score_frame(predictions, factor_values)

    merged = result.merge(
        factor_values.loc[factor_values["factor_name"] == "beta_to_spy", ["symbol", "processed_value"]],
        on="symbol",
    )
    assert merged["total_score"].abs().max() < 1e-10
