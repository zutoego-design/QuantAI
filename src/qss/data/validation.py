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


def _check(
    name: str,
    passed: bool,
    value: object,
    severity: str = "error",
    detail: str = "",
) -> dict:
    return {
        "check": name,
        "passed": bool(passed),
        "value": value,
        "severity": severity,
        "detail": detail,
    }


def _format_month_ranges(months: list[pd.Period]) -> str:
    if not months:
        return "none"
    ranges: list[tuple[pd.Period, pd.Period]] = []
    range_start = months[0]
    previous = months[0]
    for month in months[1:]:
        if month.ordinal != previous.ordinal + 1:
            ranges.append((range_start, previous))
            range_start = month
        previous = month
    ranges.append((range_start, previous))
    return ", ".join(
        str(start) if start == end else f"{start} through {end}"
        for start, end in ranges
    )


def monthly_membership_coverage(
    dates: pd.Series,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> tuple[int, int, float, str]:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    expected = list(pd.date_range(start, end, freq="ME").to_period("M"))
    observed_dates = pd.to_datetime(dates, errors="coerce").dropna()
    observed = sorted(
        set(
            observed_dates.loc[
                (observed_dates >= start) & (observed_dates <= end)
            ].dt.to_period("M")
        )
    )
    expected_set = set(expected)
    observed_in_window = [month for month in observed if month in expected_set]
    observed_count = len(observed_in_window)
    expected_count = len(expected)
    coverage = observed_count / expected_count if expected_count else 0.0
    missing = sorted(expected_set - set(observed_in_window))
    detail = (
        f"{observed_count}/{expected_count} months; "
        f"missing {_format_month_ranges(missing)}"
    )
    return observed_count, expected_count, coverage, detail


def required_research_credentials(
    config: AppConfig | None = None,
) -> dict[str, bool]:
    configured_sec_agent = bool(
        os.getenv("SEC_USER_AGENT")
        or (
            config is not None
            and config.data_sources.fundamentals.user_agent.strip()
        )
    )
    required = {
        "SEC_USER_AGENT": configured_sec_agent,
    }
    needs_listing_credentials = (
        config is None
        or (
            config.universe.membership_mode == "point_in_time"
            and config.universe.long_history_provider != "sp500_wikipedia"
        )
    )
    if needs_listing_credentials:
        required = {
            "ALPHAVANTAGE_API_KEY": bool(os.getenv("ALPHAVANTAGE_API_KEY")),
            "MASSIVE_API_KEY": bool(
                os.getenv("MASSIVE_API_KEY") or os.getenv("POLYGON_API_KEY")
            ),
            **required,
        }
    return required


def failed_check_summary(checks: pd.DataFrame, limit: int = 6) -> str:
    if checks.empty or not {"check", "passed", "value"}.issubset(checks.columns):
        return "validation did not produce readable check results"
    failed = checks.loc[~checks["passed"].astype(bool)]
    if failed.empty:
        return "no failed checks"
    details = []
    for row in failed.head(limit).itertuples(index=False):
        summary = f"{row.check}={row.value}"
        detail = getattr(row, "detail", "")
        if isinstance(detail, str) and detail:
            summary = f"{summary} ({detail})"
        details.append(summary)
    remaining = len(failed) - len(details)
    if remaining:
        details.append(f"+{remaining} more")
    return "; ".join(details)


def validate_research_data(
    config: AppConfig,
    start_date: str | None = None,
    end_date: str | None = None,
    context: RunContext | None = None,
) -> DataValidationResult:
    context = context or create_run_context(config, "data-validation", end_date)
    research_start = pd.Timestamp(
        start_date or config.backtest.start_date
    ).normalize()
    research_end = pd.Timestamp(end_date or pd.Timestamp.today()).normalize()
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
    research_symbols: set[str] = set()
    membership_window = pd.DataFrame()
    if not membership.empty and {"date", "symbol"}.issubset(membership):
        membership_dates = pd.to_datetime(membership["date"]).dt.normalize()
        membership_window = membership.loc[
            (membership_dates >= research_start)
            & (membership_dates <= research_end)
        ]
        included_membership = membership_window
        if "included" in included_membership:
            included_membership = included_membership.loc[
                included_membership["included"]
            ]
        research_symbols = set(
            included_membership["symbol"].dropna().astype(str)
        )
    if not security_master.empty:
        sector_frame = security_master
        if research_symbols and "symbol" in security_master:
            sector_frame = security_master.loc[
                security_master["symbol"].astype(str).isin(research_symbols)
            ].drop_duplicates("symbol")
        sector = sector_frame.get(
            "sector", pd.Series("Unknown", index=sector_frame.index)
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
    if (
        config.universe.membership_mode == "point_in_time"
        and not membership.empty
        and "date" in membership
    ):
        _, _, snapshot_coverage, coverage_detail = monthly_membership_coverage(
            membership["date"],
            research_start,
            research_end,
        )
        checks.append(
            _check(
                "monthly_membership_history",
                snapshot_coverage >= config.universe.min_long_price_coverage,
                snapshot_coverage,
                detail=coverage_detail,
            )
        )

    synthetic_rows = 0
    for frame in [prices, fundamentals, macro]:
        if "source" in frame:
            synthetic_rows += int(
                frame["source"].astype(str).str.contains("synthetic", case=False).sum()
            )
    checks.append(_check("synthetic_rows_zero", synthetic_rows == 0, synthetic_rows))

    fundamentals_for_checks = fundamentals
    if research_symbols and "symbol" in fundamentals_for_checks:
        fundamentals_for_checks = fundamentals_for_checks.loc[
            fundamentals_for_checks["symbol"].astype(str).isin(research_symbols)
        ]
    if (
        not fundamentals_for_checks.empty
        and {"available_date", "period_end_date"}.issubset(fundamentals_for_checks)
    ):
        invalid_dates = int(
            (
                pd.to_datetime(fundamentals_for_checks["available_date"])
                < pd.to_datetime(fundamentals_for_checks["period_end_date"])
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

    if (
        config.universe.membership_mode == "point_in_time"
        and config.universe.validation_provider != "disabled"
    ):
        coverage = universe_coverage_report(config, prices)
        if coverage.empty:
            checks.append(_check("universe_coverage", False, "no coverage rows"))
        else:
            failed = int((~coverage["passed"]).sum())
            checks.append(
                _check(
                    "universe_coverage",
                    failed == 0,
                    f"{failed}/{len(coverage)} failed",
                )
            )
            write_csv(coverage, context.path("universe_coverage.csv"))
    elif config.universe.membership_mode == "point_in_time":
        has_sp500_source = (
            not membership.empty
            and "source" in membership
            and membership["source"].astype(str).str.contains("sp500_wikipedia").any()
        )
        checks.append(
            _check(
                "sp500_source_audit",
                has_sp500_source
                and config.universe.long_history_provider == "sp500_wikipedia",
                config.universe.long_history_provider,
            )
        )
    else:
        checks.append(
            _check(
                "current_membership_bias_disclosed",
                True,
                "current Nasdaq membership is backfilled across the research window",
                severity="warning",
            )
        )

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
        float(validation["jaccard"].min())
        if not validation.empty and "jaccard" in validation
        else 0.0
    )
    if (
        config.universe.membership_mode == "point_in_time"
        and config.universe.validation_provider != "disabled"
    ):
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
    for name, present in required_research_credentials(config).items():
        checks.append(_check(f"credential_{name.lower()}", present, present))

    checks_frame = pd.DataFrame(checks)
    status = "valid" if bool(checks_frame["passed"].all()) else "invalid"
    write_csv(checks_frame, context.path("checks.csv"))
    context.update(
        status=status,
        quality_gates={
            row["check"]: bool(row["passed"]) for row in checks_frame.to_dict("records")
        },
        bias_flags=(
            ["sp500_point_in_time_wikipedia_reconstruction"]
            if config.universe.membership_mode == "point_in_time"
            and config.universe.long_history_provider == "sp500_wikipedia"
            else ["free_data_long_history_approximate"]
            if config.universe.membership_mode == "point_in_time"
            else ["current_membership_backfilled_survivorship_bias"]
        ),
    )
    return DataValidationResult(checks_frame, status, context.manifest.run_id, context.root)
