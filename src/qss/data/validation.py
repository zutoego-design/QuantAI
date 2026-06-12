from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from qss.config.schema import AppConfig
from qss.data.storage import read_parquet, write_csv
from qss.runs.manifest import RunContext, create_run_context
from qss.universe.sync import universe_coverage_report


@dataclass
class DataValidationResult:
    checks: pd.DataFrame
    status: str
    run_id: str
    run_path: Path


def _check(name: str, passed: bool, value: object, severity: str = "error") -> dict:
    return {
        "check": name,
        "passed": bool(passed),
        "value": value,
        "severity": severity,
    }


def validate_research_data(
    config: AppConfig,
    start_date: str | None = None,
    end_date: str | None = None,
    context: RunContext | None = None,
) -> DataValidationResult:
    context = context or create_run_context(config, "data-validation", end_date)
    prices = read_parquet(Path(config.paths.silver_data) / "prices" / "prices_daily.parquet")
    fundamentals = read_parquet(
        Path(config.paths.silver_data) / "fundamentals" / "fundamental_observations.parquet"
    )
    if fundamentals.empty:
        fundamentals = read_parquet(
            Path(config.paths.silver_data) / "fundamentals" / "fundamentals_quarterly.parquet"
        )
    membership = read_parquet(
        Path(config.paths.silver_data) / "universe" / "universe_membership.parquet"
    )
    macro = read_parquet(
        Path(config.paths.silver_data) / "macro" / "macro_observations.parquet"
    )
    validation = read_parquet(
        Path(config.paths.silver_data) / "universe" / "universe_validation.parquet"
    )
    security_master = read_parquet(
        Path(config.paths.silver_data) / "universe" / "security_master.parquet"
    )
    checks: list[dict] = []
    checks.append(_check("prices_nonempty", not prices.empty, len(prices)))
    checks.append(_check("fundamentals_nonempty", not fundamentals.empty, len(fundamentals)))
    checks.append(_check("membership_nonempty", not membership.empty, len(membership)))
    checks.append(_check("macro_nonempty", not macro.empty, len(macro)))
    if not security_master.empty:
        sector = security_master.get(
            "sector", pd.Series("Unknown", index=security_master.index)
        ).fillna("Unknown")
        sector_coverage = float(
            (
                ~sector.astype(str)
                .str.lower()
                .isin(["", "unknown", "unclassified"])
            ).mean()
        )
        checks.append(
            _check(
                "sector_mapping_coverage",
                sector_coverage >= config.universe.min_sector_coverage,
                sector_coverage,
            )
        )
    if not membership.empty and "date" in membership:
        membership_dates = pd.to_datetime(membership["date"]).dt.to_period("M").nunique()
        expected_dates = len(
            pd.date_range(
                pd.Timestamp(start_date or config.universe.start_date),
                pd.Timestamp(end_date or pd.Timestamp.today()),
                freq="ME",
            )
        )
        snapshot_coverage = membership_dates / expected_dates if expected_dates else 0.0
        checks.append(
            _check(
                "monthly_membership_history",
                snapshot_coverage >= config.universe.min_long_price_coverage,
                snapshot_coverage,
            )
        )

    synthetic_rows = 0
    for frame in [prices, fundamentals, macro]:
        if "source" in frame:
            synthetic_rows += int(
                frame["source"].astype(str).str.contains("synthetic", case=False).sum()
            )
    checks.append(_check("synthetic_rows_zero", synthetic_rows == 0, synthetic_rows))

    if not fundamentals.empty and {"available_date", "period_end_date"}.issubset(fundamentals):
        invalid_dates = int(
            (
                pd.to_datetime(fundamentals["available_date"])
                < pd.to_datetime(fundamentals["period_end_date"])
            ).fillna(False).sum()
        )
        checks.append(_check("fundamental_available_dates", invalid_dates == 0, invalid_dates))

    if not prices.empty:
        prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
        if start_date:
            prices = prices.loc[prices["date"] >= pd.Timestamp(start_date)]
        if end_date:
            prices = prices.loc[prices["date"] <= pd.Timestamp(end_date)]
        benchmark = prices.loc[prices["symbol"] == config.backtest.primary_benchmark]
        checks.append(
            _check(
                "primary_benchmark_present",
                not benchmark.empty,
                config.backtest.primary_benchmark,
            )
        )
        if not benchmark.empty:
            missing_benchmark = int(benchmark["return_1d"].iloc[1:].isna().sum())
            checks.append(
                _check("primary_benchmark_complete", missing_benchmark == 0, missing_benchmark)
            )
        secondary = prices.loc[
            prices["symbol"] == config.backtest.secondary_benchmark
        ]
        checks.append(
            _check(
                "secondary_benchmark_present",
                not secondary.empty,
                config.backtest.secondary_benchmark,
            )
        )
        if not secondary.empty:
            missing_secondary = int(secondary["return_1d"].iloc[1:].isna().sum())
            checks.append(
                _check(
                    "secondary_benchmark_complete",
                    missing_secondary == 0,
                    missing_secondary,
                )
            )

    coverage = universe_coverage_report(config, prices)
    if coverage.empty:
        checks.append(_check("universe_coverage", False, "no coverage rows"))
    else:
        failed = int((~coverage["passed"]).sum())
        checks.append(_check("universe_coverage", failed == 0, f"{failed}/{len(coverage)} failed"))
        write_csv(coverage, context.path("universe_coverage.csv"))

    validation_end = pd.Timestamp(end_date or pd.Timestamp.today())
    validation_start = validation_end - pd.DateOffset(
        years=config.universe.recent_validation_years
    )
    expected_validation_months = len(
        pd.date_range(validation_start, validation_end, freq="ME")
    )
    observed_validation_months = (
        pd.to_datetime(validation["date"]).dt.to_period("M").nunique()
        if not validation.empty and "date" in validation
        else 0
    )
    validation_coverage = (
        observed_validation_months / expected_validation_months
        if expected_validation_months
        else 0.0
    )
    minimum_jaccard = (
        float(validation["jaccard"].min()) if not validation.empty else 0.0
    )
    checks.append(
        _check(
            "recent_cross_source_validation",
            validation_coverage >= 0.95
            and minimum_jaccard >= config.universe.min_recent_jaccard,
            {
                "month_coverage": validation_coverage,
                "minimum_jaccard": minimum_jaccard,
            },
        )
    )
    required_credentials = {
        "ALPHAVANTAGE_API_KEY": bool(os.getenv("ALPHAVANTAGE_API_KEY")),
        "MASSIVE_API_KEY": bool(os.getenv("MASSIVE_API_KEY") or os.getenv("POLYGON_API_KEY")),
        "FRED_API_KEY": bool(os.getenv("FRED_API_KEY")),
        "SEC_USER_AGENT": bool(os.getenv("SEC_USER_AGENT")),
    }
    for name, present in required_credentials.items():
        checks.append(_check(f"credential_{name.lower()}", present, present))

    checks_frame = pd.DataFrame(checks)
    status = "valid" if bool(checks_frame["passed"].all()) else "invalid"
    write_csv(checks_frame, context.path("checks.csv"))
    context.update(
        status=status,
        quality_gates={
            row["check"]: bool(row["passed"]) for row in checks_frame.to_dict("records")
        },
        bias_flags=["free_data_long_history_approximate"],
    )
    return DataValidationResult(checks_frame, status, context.manifest.run_id, context.root)
