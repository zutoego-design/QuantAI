from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from qss.config.schema import AppConfig
from qss.data.calendar import latest_expected_us_trading_day
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
    if "severity" in failed:
        failed = failed.assign(
            _priority=failed["severity"].map({"error": 0, "warning": 1}).fillna(2)
        ).sort_values("_priority", kind="stable")
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
    filings = read_parquet(
        Path(config.paths.silver_data) / "events" / "sec_filings.parquet"
    )
    corporate_actions_path = (
        Path(config.paths.silver_data)
        / "corporate_actions"
        / "corporate_actions.parquet"
    )
    delisting_returns_path = (
        Path(config.paths.silver_data)
        / "corporate_actions"
        / "delisting_returns.parquet"
    )
    checks: list[dict] = []
    checks.append(_check("prices_nonempty", not prices.empty, len(prices)))
    checks.append(_check("fundamentals_nonempty", not fundamentals.empty, len(fundamentals)))
    checks.append(_check("membership_nonempty", not membership.empty, len(membership)))
    checks.append(_check("macro_nonempty", not macro.empty, len(macro)))
    checks.append(
        _check(
            "macro_vintage_history",
            "vintage_date" in macro,
            "available" if "vintage_date" in macro else "unavailable",
            severity="warning",
            detail="Use ALFRED or another revision-aware source for confirmatory macro research.",
        )
    )
    checks.append(
        _check(
            "corporate_action_ledger",
            corporate_actions_path.exists(),
            str(corporate_actions_path),
            severity="warning",
            detail="Independent split, dividend, merger, spin-off, and exchange terms are required.",
        )
    )
    checks.append(
        _check(
            "exact_delisting_returns",
            delisting_returns_path.exists(),
            str(delisting_returns_path),
            severity="warning",
            detail="Scenario sensitivity is not a substitute for observed terminal cash flows.",
        )
    )
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
        stable_id_coverage = (
            float(included_membership["security_id"].notna().mean())
            if "security_id" in included_membership and not included_membership.empty
            else 0.0
        )
        checks.append(
            _check(
                "stable_security_identifier_coverage",
                stable_id_coverage >= 0.99,
                stable_id_coverage,
                severity="warning",
            )
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
        expected_price_cutoff = latest_expected_us_trading_day(research_end)
        benchmark_history = prices.loc[
            prices["symbol"] == config.backtest.primary_benchmark
        ]
        latest_benchmark_date = (
            benchmark_history["date"].max()
            if not benchmark_history.empty
            else pd.NaT
        )
        benchmark_sessions = set(benchmark_history["date"])
        checks.append(
            _check(
                "price_data_cutoff",
                pd.notna(latest_benchmark_date)
                and expected_price_cutoff in benchmark_sessions,
                (
                    str(latest_benchmark_date.date())
                    if pd.notna(latest_benchmark_date)
                    else "missing"
                ),
                detail=(
                    f"requested={research_end.date()}; "
                    f"expected_session={expected_price_cutoff.date()}"
                ),
            )
        )
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
        checks.append(
            _check(
                "authorized_constituent_history",
                False,
                config.universe.long_history_provider,
                severity="warning",
                detail="Wikipedia reconstruction is not a licensed constituent master.",
            )
        )
        checks.append(
            _check(
                "cross_source_validation_enabled",
                False,
                config.universe.validation_provider,
                severity="warning",
                detail="Configure a second provider before institutional use.",
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

    eligible_filings = filings
    if not eligible_filings.empty and "filing_type" in eligible_filings:
        eligible_filings = eligible_filings.loc[
            eligible_filings["filing_type"]
            .astype(str)
            .isin(config.text_factors.filing_types)
        ]
    if not eligible_filings.empty and "filing_timestamp" in eligible_filings:
        filing_dates = pd.to_datetime(
            eligible_filings["filing_timestamp"],
            errors="coerce",
            utc=True,
        ).dt.tz_localize(None)
        eligible_filings = eligible_filings.loc[
            filing_dates.between(research_start, research_end)
        ]
    text_coverage = (
        float(eligible_filings["text_cached"].fillna(False).astype(bool).mean())
        if not eligible_filings.empty and "text_cached" in eligible_filings
        else 0.0
    )
    checks.append(
        _check(
            "sec_text_history_coverage",
            text_coverage >= 0.80,
            text_coverage,
            severity="warning",
            detail=f"target=80%; eligible_filings={len(eligible_filings)}",
        )
    )

    checks_frame = pd.DataFrame(checks)
    blocking_failures = checks_frame.loc[
        (~checks_frame["passed"].astype(bool))
        & checks_frame["severity"].eq("error")
    ]
    status = "valid" if blocking_failures.empty else "invalid"
    write_csv(checks_frame, context.path("checks.csv"))
    context.update(
        status=status,
        quality_gates={
            row["check"]: bool(row["passed"]) for row in checks_frame.to_dict("records")
        },
        bias_flags=[
            *(
                ["sp500_point_in_time_wikipedia_reconstruction"]
                if config.universe.membership_mode == "point_in_time"
                and config.universe.long_history_provider == "sp500_wikipedia"
                else ["free_data_long_history_approximate"]
                if config.universe.membership_mode == "point_in_time"
                else ["current_membership_backfilled_survivorship_bias"]
            ),
            *(
                ["single_source_universe_validation"]
                if config.universe.validation_provider == "disabled"
                else []
            ),
            *(["macro_vintage_history_unavailable"] if "vintage_date" not in macro else []),
            *(["corporate_action_ledger_unavailable"] if not corporate_actions_path.exists() else []),
            *(["exact_delisting_returns_unavailable"] if not delisting_returns_path.exists() else []),
            *(["sec_text_coverage_below_80pct"] if text_coverage < 0.80 else []),
        ],
    )
    return DataValidationResult(checks_frame, status, context.manifest.run_id, context.root)
