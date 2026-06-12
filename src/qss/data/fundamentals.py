from __future__ import annotations

import pandas as pd

FUNDAMENTAL_METRICS = [
    "revenue",
    "gross_profit",
    "operating_income",
    "net_income",
    "total_assets",
    "total_liabilities",
    "shareholders_equity",
    "operating_cash_flow",
    "capital_expenditure",
    "free_cash_flow",
    "shares_outstanding",
]


def latest_fundamentals_as_of(
    fundamentals: pd.DataFrame,
    as_of_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """Select the latest available observation independently for every metric."""
    if fundamentals.empty:
        return pd.DataFrame(columns=["symbol", *FUNDAMENTAL_METRICS])
    as_of = pd.Timestamp(as_of_date).normalize()
    frame = fundamentals.loc[pd.to_datetime(fundamentals["available_date"]) <= as_of].copy()
    if frame.empty:
        return pd.DataFrame(columns=["symbol", *FUNDAMENTAL_METRICS])

    if {"metric", "value"}.issubset(frame.columns):
        latest = (
            frame.sort_values(
                ["symbol", "metric", "available_date", "period_end_date", "filing_date"]
            )
            .groupby(["symbol", "metric"], as_index=False)
            .tail(1)
        )
        pivot = latest.pivot(index="symbol", columns="metric", values="value").reset_index()
        if {"operating_cash_flow", "capital_expenditure"}.issubset(pivot.columns):
            pivot["free_cash_flow"] = (
                pivot["operating_cash_flow"] - pivot["capital_expenditure"].fillna(0.0)
            )
        return pivot

    rows: list[dict] = []
    metrics = [metric for metric in FUNDAMENTAL_METRICS if metric in frame.columns]
    for symbol, group in frame.groupby("symbol"):
        row: dict = {"symbol": symbol}
        sort_columns = [
            column
            for column in ["available_date", "period_end_date", "filing_date"]
            if column in group
        ]
        ordered = group.sort_values(sort_columns)
        for metric in metrics:
            available = ordered.loc[ordered[metric].notna()]
            row[metric] = available.iloc[-1][metric] if not available.empty else pd.NA
        rows.append(row)
    return pd.DataFrame(rows)


def observations_to_wide(observations: pd.DataFrame) -> pd.DataFrame:
    if observations.empty:
        return pd.DataFrame()
    keys = [
        "symbol",
        "period_end_date",
        "filing_date",
        "available_date",
        "fiscal_year",
        "fiscal_period",
        "form",
        "accession",
        "source",
        "quality_status",
        "ingestion_time",
    ]
    wide = observations.pivot_table(
        index=keys,
        columns="metric",
        values="value",
        aggfunc="last",
        dropna=True,
    ).reset_index()
    wide.columns.name = None
    if {"operating_cash_flow", "capital_expenditure"}.issubset(wide.columns):
        wide["free_cash_flow"] = (
            wide["operating_cash_flow"] - wide["capital_expenditure"].fillna(0.0)
        )
    return wide
