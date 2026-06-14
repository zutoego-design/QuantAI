from __future__ import annotations

import pandas as pd

from qss.labels.schema import LabelDefinition

LABEL_COLUMNS = [
    "date",
    "symbol",
    "label_name",
    "label_value",
    "horizon",
    "label_start_time",
    "label_end_time",
    "overlap",
    "purge_required",
    "embargo_days",
    "version",
]


def _empty_labels() -> pd.DataFrame:
    return pd.DataFrame(columns=LABEL_COLUMNS)


def _mark_overlaps(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    output = frame.sort_values(["symbol", "label_start_time", "label_end_time"]).copy()
    previous_end = output.groupby("symbol")["label_end_time"].shift()
    next_start = output.groupby("symbol")["label_start_time"].shift(-1)
    output["overlap"] = (
        (previous_end.notna() & (output["label_start_time"] <= previous_end))
        | (next_start.notna() & (next_start <= output["label_end_time"]))
    )
    output["purge_required"] = output["overlap"]
    return output.sort_values(["date", "symbol"]).reset_index(drop=True)


def build_forward_return_labels(
    prices: pd.DataFrame,
    signal_dates: list[pd.Timestamp] | pd.DatetimeIndex,
    definition: LabelDefinition,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    required = {"date", "symbol", "adj_close"}
    if prices.empty or not required.issubset(prices.columns):
        return _empty_labels()
    frame = prices.loc[:, ["date", "symbol", "adj_close"]].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
    if symbols is not None:
        frame = frame.loc[frame["symbol"].isin(symbols)]
    dates = sorted({pd.Timestamp(value).normalize() for value in signal_dates})
    rows: list[dict] = []
    for symbol, history in frame.groupby("symbol"):
        history = history.dropna(subset=["adj_close"]).sort_values("date").drop_duplicates("date")
        trading_dates = pd.DatetimeIndex(history["date"])
        closes = history.set_index("date")["adj_close"].astype(float)
        for signal_date in dates:
            start_position = int(trading_dates.searchsorted(signal_date, side="left"))
            start_position += definition.start_offset_days
            end_position = start_position + definition.horizon_days
            if start_position >= len(trading_dates) or end_position >= len(trading_dates):
                continue
            start_time = pd.Timestamp(trading_dates[start_position])
            end_time = pd.Timestamp(trading_dates[end_position])
            start_price = float(closes.loc[start_time])
            end_price = float(closes.loc[end_time])
            if start_price <= 0:
                continue
            rows.append(
                {
                    "date": signal_date,
                    "symbol": symbol,
                    "label_name": definition.name,
                    "label_value": end_price / start_price - 1.0,
                    "horizon": definition.horizon_days,
                    "label_start_time": start_time,
                    "label_end_time": end_time,
                    "overlap": False,
                    "purge_required": False,
                    "embargo_days": definition.embargo_days,
                    "version": definition.version,
                }
            )
    if not rows:
        return _empty_labels()
    return _mark_overlaps(pd.DataFrame(rows)[LABEL_COLUMNS])


def build_cross_sectional_rank_labels(
    forward_returns: pd.DataFrame,
    definition: LabelDefinition,
) -> pd.DataFrame:
    if forward_returns.empty:
        return _empty_labels()
    output = forward_returns.copy()
    output["label_name"] = definition.name
    output["label_value"] = output.groupby("date")["label_value"].rank(
        method="average", pct=True
    )
    output["version"] = definition.version
    output["horizon"] = definition.horizon_days
    output["embargo_days"] = definition.embargo_days
    return output[LABEL_COLUMNS].sort_values(["date", "symbol"]).reset_index(drop=True)


def build_event_window_labels(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    definition: LabelDefinition,
) -> pd.DataFrame:
    if events.empty or not {"symbol", "filing_timestamp"}.issubset(events.columns):
        return _empty_labels()
    event_dates = events[["symbol", "filing_timestamp"]].copy()
    event_dates["date"] = (
        pd.to_datetime(event_dates["filing_timestamp"]).dt.tz_localize(None).dt.normalize()
    )
    frames = []
    for symbol, group in event_dates.groupby("symbol"):
        labels = build_forward_return_labels(
            prices,
            pd.DatetimeIndex(group["date"]),
            definition,
            symbols=[symbol],
        )
        if not labels.empty:
            frames.append(labels)
    return pd.concat(frames, ignore_index=True) if frames else _empty_labels()
