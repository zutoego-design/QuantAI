from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from qss.config.schema import WalkForwardConfig


@dataclass(frozen=True)
class WalkForwardFold:
    fold: int
    train_dates: tuple[pd.Timestamp, ...]
    test_dates: tuple[pd.Timestamp, ...]
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    purged_rows: int


def walk_forward_splits(
    samples: pd.DataFrame,
    config: WalkForwardConfig,
) -> list[tuple[WalkForwardFold, pd.Index, pd.Index]]:
    required = {"date", "label_end_time"}
    if samples.empty or not required.issubset(samples.columns):
        return []
    frame = samples.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["label_end_time"] = pd.to_datetime(frame["label_end_time"]).dt.normalize()
    unique_dates = tuple(sorted(frame["date"].unique()))
    minimum = max(config.min_train_periods, 1)
    first_test = max(config.train_periods, minimum)
    folds = []
    fold_number = 0
    for test_position in range(first_test, len(unique_dates), config.step_periods):
        test_dates = unique_dates[test_position : test_position + config.test_periods]
        if not test_dates:
            continue
        train_start_position = max(0, test_position - config.train_periods) if config.rolling else 0
        train_dates = unique_dates[train_start_position:test_position]
        if len(train_dates) < minimum:
            continue
        test_start = pd.Timestamp(test_dates[0])
        embargo_cutoff = test_start - pd.Timedelta(days=config.embargo_days)
        base_train = frame["date"].isin(train_dates)
        train_mask = base_train.copy()
        if config.purge:
            train_mask &= frame["label_end_time"] < embargo_cutoff
        test_mask = frame["date"].isin(test_dates)
        train_index = frame.index[train_mask]
        test_index = frame.index[test_mask]
        if train_index.empty or test_index.empty:
            continue
        fold_number += 1
        fold = WalkForwardFold(
            fold=fold_number,
            train_dates=tuple(pd.Timestamp(value) for value in train_dates),
            test_dates=tuple(pd.Timestamp(value) for value in test_dates),
            train_start=pd.Timestamp(train_dates[0]),
            train_end=pd.Timestamp(train_dates[-1]),
            test_start=test_start,
            test_end=pd.Timestamp(test_dates[-1]),
            purged_rows=int(base_train.sum() - train_mask.sum()),
        )
        folds.append((fold, train_index, test_index))
    return folds
