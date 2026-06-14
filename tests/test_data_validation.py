import pandas as pd

from qss.config.loader import get_config
from qss.data.storage import write_parquet
from qss.data.validation import (
    monthly_membership_coverage,
    validate_research_data,
)


def _config(tmp_path):
    config = get_config(["configs/default.yaml"])
    config.paths.raw_data = str(tmp_path / "raw")
    config.paths.silver_data = str(tmp_path / "silver")
    config.paths.gold_data = str(tmp_path / "gold")
    config.paths.reports = str(tmp_path / "reports")
    return config


def _write_validation_inputs(tmp_path, latest_price_date="2026-06-12"):
    silver = tmp_path / "silver"
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2016-01-04",
                    latest_price_date,
                    "2016-01-04",
                    latest_price_date,
                ]
            ),
            "symbol": ["^GSPC", "^GSPC", "SPY", "SPY"],
            "return_1d": [pd.NA, 0.01, pd.NA, 0.01],
            "source": ["yfinance"] * 4,
        }
    )
    fundamentals = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "period_end_date": [pd.Timestamp("2025-12-31")],
            "available_date": [pd.Timestamp("2026-02-01")],
            "source": ["sec_edgar"],
        }
    )
    months = pd.date_range("2023-01-01", "2026-06-13", freq="ME")
    membership = pd.DataFrame(
        {
            "date": months,
            "symbol": ["AAA"] * len(months),
            "source": ["sp500_wikipedia_point_in_time"] * len(months),
            "included": [True] * len(months),
        }
    )
    macro = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-06-01")],
            "series_id": ["DGS10"],
            "value": [4.0],
            "source": ["fred"],
        }
    )
    security_master = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "sector": ["Industrials"],
        }
    )
    validation = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-06-12")],
            "source": ["sp500_wikipedia_single_source_audit"],
        }
    )
    write_parquet(prices, silver / "prices" / "prices_daily.parquet")
    write_parquet(
        fundamentals,
        silver / "fundamentals" / "fundamental_observations.parquet",
    )
    write_parquet(
        membership,
        silver / "universe" / "universe_membership.parquet",
    )
    write_parquet(macro, silver / "macro" / "macro_observations.parquet")
    write_parquet(
        security_master,
        silver / "universe" / "security_master.parquet",
    )
    write_parquet(
        validation,
        silver / "universe" / "universe_validation.parquet",
    )


def test_monthly_membership_gate_reports_counts_and_missing_range(tmp_path):
    config = _config(tmp_path)
    _write_validation_inputs(tmp_path)

    result = validate_research_data(
        config,
        start_date="2016-01-01",
        end_date="2026-06-13",
    )
    check = result.checks.set_index("check").loc["monthly_membership_history"]

    assert check["passed"] == False  # noqa: E712
    assert check["value"] == 41 / 125
    assert check["detail"] == (
        "41/125 months; missing 2016-01 through 2022-12"
    )


def test_monthly_membership_coverage_ignores_out_of_window_months():
    dates = pd.Series(
        pd.to_datetime(
            [
                "2024-12-31",
                "2025-01-31",
                "2025-02-28",
                "2025-03-31",
                "2025-04-30",
            ]
        )
    )

    observed, expected, coverage, detail = monthly_membership_coverage(
        dates,
        "2025-01-01",
        "2025-03-31",
    )

    assert (observed, expected, coverage) == (3, 3, 1.0)
    assert detail == "3/3 months; missing none"


def test_price_cutoff_requires_requested_trading_session(tmp_path):
    config = _config(tmp_path)
    _write_validation_inputs(tmp_path, latest_price_date="2026-06-11")

    result = validate_research_data(
        config,
        start_date="2026-01-01",
        end_date="2026-06-12",
    )
    check = result.checks.set_index("check").loc["price_data_cutoff"]

    assert result.status == "invalid"
    assert check["passed"] == False  # noqa: E712
    assert check["value"] == "2026-06-11"
    assert "expected_session=2026-06-12" in check["detail"]


def test_price_cutoff_accepts_prior_session_for_weekend_request(tmp_path):
    config = _config(tmp_path)
    _write_validation_inputs(tmp_path, latest_price_date="2026-06-12")

    result = validate_research_data(
        config,
        start_date="2026-01-01",
        end_date="2026-06-14",
    )
    check = result.checks.set_index("check").loc["price_data_cutoff"]

    assert result.status == "valid"
    assert check["passed"] == True  # noqa: E712
    assert "expected_session=2026-06-12" in check["detail"]
