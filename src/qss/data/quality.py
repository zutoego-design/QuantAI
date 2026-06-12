from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from qss.data.storage import write_csv


@dataclass
class QualityIssue:
    dataset: str
    rule: str
    severity: str
    details: str


def _issue(dataset: str, rule: str, severity: str, details: str) -> QualityIssue:
    return QualityIssue(dataset=dataset, rule=rule, severity=severity, details=details)


def check_data_quality(
    dataset_name: str,
    df: pd.DataFrame,
    primary_keys: Iterable[str],
    as_of_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    issues: list[QualityIssue] = []
    primary_keys = list(primary_keys)
    if df.empty:
        issues.append(_issue(dataset_name, "empty_dataset", "warning", "Dataset is empty."))
        return pd.DataFrame([issue.__dict__ for issue in issues])

    duplicates = int(df.duplicated(primary_keys).sum())
    if duplicates:
        issues.append(_issue(dataset_name, "duplicate_primary_keys", "error", str(duplicates)))

    for column, rule in [("symbol", "missing_symbols"), ("date", "missing_dates")]:
        if column in df.columns:
            missing = int(df[column].isna().sum())
            if missing:
                issues.append(_issue(dataset_name, rule, "error", f"{missing} missing values"))

    if "adj_close" in df.columns:
        missing = int(df["adj_close"].isna().sum())
        if missing:
            issues.append(_issue(dataset_name, "missing_adjusted_close", "error", f"{missing} rows"))

    if "close" in df.columns:
        negatives = int((df["close"] <= 0).fillna(False).sum())
        if negatives:
            issues.append(_issue(dataset_name, "negative_prices", "error", f"{negatives} rows"))

    if "volume" in df.columns:
        non_positive = int((df["volume"] <= 0).fillna(False).sum())
        if non_positive:
            issues.append(_issue(dataset_name, "non_positive_volume", "warning", f"{non_positive} rows"))

    if "return_1d" in df.columns:
        extreme = int((df["return_1d"].abs() > 0.40).fillna(False).sum())
        if extreme:
            issues.append(_issue(dataset_name, "extreme_returns", "warning", f"{extreme} rows"))

    if "sector" in df.columns:
        missing = int(df["sector"].fillna("").eq("").sum())
        if missing:
            issues.append(_issue(dataset_name, "missing_sector_mapping", "warning", f"{missing} rows"))

    if "market_cap" in df.columns:
        missing = int(df["market_cap"].isna().sum())
        if missing:
            issues.append(_issue(dataset_name, "missing_market_cap", "warning", f"{missing} rows"))

    if "available_date" in df.columns:
        missing = int(df["available_date"].isna().sum())
        if missing:
            issues.append(_issue(dataset_name, "missing_available_date", "error", f"{missing} rows"))
        if "period_end_date" in df.columns:
            invalid = int((df["available_date"] < df["period_end_date"]).fillna(False).sum())
            if invalid:
                issues.append(
                    _issue(dataset_name, "available_date_before_period_end", "error", f"{invalid} rows")
                )

    if as_of_date is not None and "date" in df.columns:
        stale = int((pd.Timestamp(as_of_date) - pd.to_datetime(df["date"]).max()).days > 10)
        if stale:
            issues.append(_issue(dataset_name, "stale_data", "warning", "Latest row is older than 10 days"))

    if not issues:
        issues.append(_issue(dataset_name, "ok", "info", "No quality issues detected"))

    return pd.DataFrame([issue.__dict__ for issue in issues])


def write_quality_report(report: pd.DataFrame, report_path: str | Path) -> Path:
    path = Path(report_path)
    if path.exists():
        existing = pd.read_csv(path)
        report = pd.concat([existing, report], ignore_index=True).drop_duplicates(
            ["dataset", "rule", "severity", "details"],
            keep="last",
        )
    return write_csv(report, path)
