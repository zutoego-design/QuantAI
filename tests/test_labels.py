import pandas as pd

from qss.labels.builders import (
    build_cross_sectional_rank_labels,
    build_event_window_labels,
    build_forward_return_labels,
)
from qss.labels.schema import LabelDefinition
from qss.labels.validation import validate_label_artifact


def _prices() -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-01", periods=80)
    rows = []
    for offset, symbol in enumerate(["AAA", "BBB"], start=1):
        for index, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "adj_close": 100 + offset * index,
                }
            )
    return pd.DataFrame(rows)


def test_forward_return_labels_align_horizon_and_flag_overlap():
    definition = LabelDefinition(
        name="forward_return",
        horizon_days=10,
        embargo_days=3,
    )
    labels = build_forward_return_labels(
        _prices(),
        pd.to_datetime(["2025-01-10", "2025-01-17", "2025-02-14"]),
        definition,
    )
    assert not labels.empty
    assert (labels["label_end_time"] > labels["label_start_time"]).all()
    assert labels["overlap"].any()
    assert (labels.loc[labels["overlap"], "purge_required"]).all()
    checks = validate_label_artifact(labels, labels[["date", "symbol"]])
    assert bool(checks["passed"].all())


def test_cross_sectional_rank_and_event_labels_are_reproducible():
    forward_definition = LabelDefinition(name="forward_return", horizon_days=5)
    forward = build_forward_return_labels(
        _prices(),
        pd.to_datetime(["2025-01-10", "2025-02-10"]),
        forward_definition,
    )
    rank = build_cross_sectional_rank_labels(
        forward,
        LabelDefinition(name="cross_sectional_rank", horizon_days=5),
    )
    assert rank["label_value"].between(0, 1).all()
    events = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "filing_timestamp": [pd.Timestamp("2025-01-10 16:30")],
        }
    )
    event_labels = build_event_window_labels(
        events,
        _prices(),
        LabelDefinition(name="event_window_return", horizon_days=5),
    )
    assert list(event_labels["symbol"].unique()) == ["AAA"]
