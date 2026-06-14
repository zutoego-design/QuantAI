import pandas as pd

from qss.config.loader import get_config
from qss.data.status import membership_symbols, research_data_status
from qss.data.storage import write_parquet


def _config(tmp_path):
    config = get_config(["configs/default.yaml"])
    config.paths.raw_data = str(tmp_path / "raw")
    config.paths.silver_data = str(tmp_path / "silver")
    config.paths.gold_data = str(tmp_path / "gold")
    config.paths.reports = str(tmp_path / "reports")
    config.universe.start_date = "2025-01-01"
    config.backtest.start_date = "2025-01-01"
    return config


def test_membership_symbols_respects_research_window(tmp_path):
    config = _config(tmp_path)
    membership = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-12-31", "2025-01-31", "2025-02-28"]),
            "symbol": ["OLD", "AAA", "BBB"],
            "source": ["alpha_vantage_listing_status"] * 3,
        }
    )
    write_parquet(
        membership,
        tmp_path / "silver" / "universe" / "universe_membership.parquet",
    )

    assert membership_symbols(
        config,
        start_date="2025-01-01",
        end_date="2025-02-28",
    ) == ["AAA", "BBB"]


def test_data_status_does_not_treat_empty_parquet_as_ready(tmp_path, monkeypatch):
    config = _config(tmp_path)
    config.universe.membership_mode = "point_in_time"
    config.universe.long_history_provider = "alpha_vantage"
    config.universe.validation_provider = "massive"
    membership = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-31"]),
            "symbol": ["AAA"],
            "source": ["alpha_vantage_listing_status"],
        }
    )
    write_parquet(
        membership,
        tmp_path / "silver" / "universe" / "universe_membership.parquet",
    )
    write_parquet(
        pd.DataFrame(columns=["date"]),
        tmp_path / "silver" / "universe" / "universe_validation.parquet",
    )
    for name in ["ALPHAVANTAGE_API_KEY", "MASSIVE_API_KEY", "SEC_USER_AGENT"]:
        monkeypatch.setenv(name, "configured")

    status = research_data_status(
        config,
        start_date="2025-01-01",
        end_date="2025-03-31",
    )
    checks = status.checks.set_index("component")

    assert checks.loc["Universe history", "status"] == "partial"
    assert checks.loc["Cross-source validation", "status"] == "missing"
    assert status.ready is False


def test_current_membership_mode_replaces_historical_provider_gates(
    tmp_path,
    monkeypatch,
):
    config = _config(tmp_path)
    config.universe.membership_mode = "current_snapshot"
    config.universe.long_history_provider = "nasdaq_trader_current"
    membership = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2025-01-31", "2025-02-28", "2025-03-31"]
            ),
            "symbol": ["AAA", "AAA", "AAA"],
            "source": ["nasdaq_trader_current_backfill"] * 3,
        }
    )
    write_parquet(
        membership,
        tmp_path / "silver" / "universe" / "universe_membership.parquet",
    )
    monkeypatch.setenv("SEC_USER_AGENT", "configured")
    monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)

    status = research_data_status(
        config,
        start_date="2025-01-01",
        end_date="2025-03-31",
    )
    checks = status.checks.set_index("component")

    assert checks.loc["Current universe baseline", "status"] == "ready"
    assert checks.loc["Survivorship-bias disclosure", "status"] == "ready"
    assert checks.loc["Provider credentials", "progress"] == "1/1 configured"


def test_sp500_history_mode_does_not_require_listing_api_keys(tmp_path, monkeypatch):
    config = _config(tmp_path)
    membership = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2025-01-31", "2025-02-28", "2025-03-31"]
            ),
            "symbol": ["AAA", "AAA", "AAA"],
            "source": ["sp500_wikipedia_point_in_time"] * 3,
            "included": [True, True, True],
        }
    )
    validation = pd.DataFrame(
        {
            "date": [pd.Timestamp("2025-03-31")],
            "source": ["sp500_wikipedia_single_source_audit"],
        }
    )
    write_parquet(
        membership,
        tmp_path / "silver" / "universe" / "universe_membership.parquet",
    )
    write_parquet(
        validation,
        tmp_path / "silver" / "universe" / "universe_validation.parquet",
    )
    monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)

    status = research_data_status(
        config,
        start_date="2025-01-01",
        end_date="2025-03-31",
    )
    checks = status.checks.set_index("component")

    assert checks.loc["Universe history", "status"] == "ready"
    assert checks.loc["Universe source audit", "status"] == "ready"
    assert checks.loc["Provider credentials", "progress"] == "1/1 configured"


def test_data_status_uses_requested_research_window_for_membership(tmp_path):
    config = _config(tmp_path)
    months = pd.date_range("2023-01-01", "2026-06-13", freq="ME")
    membership = pd.DataFrame(
        {
            "date": months,
            "symbol": ["AAA"] * len(months),
            "source": ["sp500_wikipedia_point_in_time"] * len(months),
            "included": [True] * len(months),
        }
    )
    write_parquet(
        membership,
        tmp_path / "silver" / "universe" / "universe_membership.parquet",
    )

    status = research_data_status(
        config,
        start_date="2016-01-01",
        end_date="2026-06-13",
    )
    universe = status.checks.set_index("component").loc["Universe history"]

    assert universe["status"] == "partial"
    assert universe["progress"] == "41/125 months"
    assert "missing 2016-01 through 2022-12" in universe["detail"]
