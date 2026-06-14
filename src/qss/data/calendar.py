from __future__ import annotations

import exchange_calendars as xcals
import pandas as pd


def business_days(start_date: str | pd.Timestamp, end_date: str | pd.Timestamp) -> pd.DatetimeIndex:
    return pd.bdate_range(pd.Timestamp(start_date), pd.Timestamp(end_date))


def latest_expected_us_trading_day(
    end_date: str | pd.Timestamp,
) -> pd.Timestamp:
    end = pd.Timestamp(end_date).normalize()
    calendar = xcals.get_calendar("XNYS")
    session = calendar.date_to_session(end, direction="previous")
    return pd.Timestamp(session).tz_localize(None).normalize()


def month_end_dates(start_date: str | pd.Timestamp, end_date: str | pd.Timestamp) -> list[pd.Timestamp]:
    idx = pd.date_range(pd.Timestamp(start_date), pd.Timestamp(end_date), freq="ME")
    return [pd.Timestamp(x).normalize() for x in idx]


def next_trading_day(date: pd.Timestamp, trading_dates: pd.DatetimeIndex, lag_days: int = 1) -> pd.Timestamp:
    date = pd.Timestamp(date)
    if lag_days <= 0:
        if date in trading_dates:
            return date.normalize()
        later_or_equal = trading_dates[trading_dates >= date]
        return (
            pd.Timestamp(later_or_equal[0]).normalize()
            if len(later_or_equal)
            else date.normalize()
        )
    later = trading_dates[trading_dates > date]
    if len(later) == 0:
        return pd.Timestamp(date)
    position = min(lag_days - 1, len(later) - 1)
    return pd.Timestamp(later[position]).normalize()
