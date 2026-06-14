from __future__ import annotations

import pandas as pd


def validate_label_artifact(
    labels: pd.DataFrame,
    feature_dates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    checks: list[dict] = []
    required = {
        "date",
        "symbol",
        "label_value",
        "label_start_time",
        "label_end_time",
        "overlap",
        "purge_required",
        "embargo_days",
    }
    checks.append(
        {
            "check": "required_columns",
            "passed": required.issubset(labels.columns),
            "details": str(sorted(required - set(labels.columns))),
        }
    )
    if labels.empty or not required.issubset(labels.columns):
        checks.append(
            {"check": "labels_nonempty", "passed": False, "details": f"rows={len(labels)}"}
        )
        return pd.DataFrame(checks)
    starts = pd.to_datetime(labels["label_start_time"])
    ends = pd.to_datetime(labels["label_end_time"])
    checks.extend(
        [
            {
                "check": "positive_horizon_alignment",
                "passed": bool((ends > starts).all()),
                "details": f"invalid={int((ends <= starts).sum())}",
            },
            {
                "check": "overlap_requires_purge",
                "passed": bool((~labels["overlap"] | labels["purge_required"]).all()),
                "details": (
                    f"overlap={int(labels['overlap'].sum())}; "
                    f"purge={int(labels['purge_required'].sum())}"
                ),
            },
        ]
    )
    if feature_dates is not None and not feature_dates.empty:
        features = feature_dates[["date", "symbol"]].copy()
        features["date"] = pd.to_datetime(features["date"])
        merged = labels.merge(features, on=["date", "symbol"], how="inner")
        leaked = pd.to_datetime(merged["date"]) > pd.to_datetime(merged["label_start_time"])
        checks.append(
            {
                "check": "no_future_feature_leakage",
                "passed": not bool(leaked.any()),
                "details": f"violations={int(leaked.sum())}",
            }
        )
    return pd.DataFrame(checks)
