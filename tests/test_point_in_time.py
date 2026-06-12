import pandas as pd

from qss.data.fundamentals import latest_fundamentals_as_of
from qss.universe.providers import (
    classify_security,
    normalize_alpha_vantage,
    permanent_security_id,
)


def test_latest_fundamentals_selects_each_metric_independently():
    frame = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA"],
            "available_date": pd.to_datetime(["2025-02-01", "2025-05-01"]),
            "period_end_date": pd.to_datetime(["2024-12-31", "2025-03-31"]),
            "filing_date": pd.to_datetime(["2025-02-01", "2025-05-01"]),
            "revenue": [100.0, 110.0],
            "shares_outstanding": [10.0, pd.NA],
        }
    )
    latest = latest_fundamentals_as_of(frame, "2025-06-01").set_index("symbol")
    assert latest.loc["AAA", "revenue"] == 110.0
    assert latest.loc["AAA", "shares_outstanding"] == 10.0


def test_observation_model_derives_free_cash_flow_without_lookahead():
    frame = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "AAA"],
            "metric": ["operating_cash_flow", "capital_expenditure", "operating_cash_flow"],
            "value": [50.0, 10.0, 80.0],
            "available_date": pd.to_datetime(["2025-02-01", "2025-02-01", "2025-08-01"]),
            "period_end_date": pd.to_datetime(["2024-12-31", "2024-12-31", "2025-06-30"]),
            "filing_date": pd.to_datetime(["2025-02-01", "2025-02-01", "2025-08-01"]),
        }
    )
    latest = latest_fundamentals_as_of(frame, "2025-06-01").set_index("symbol")
    assert latest.loc["AAA", "operating_cash_flow"] == 50.0
    assert latest.loc["AAA", "free_cash_flow"] == 40.0


def test_security_id_survives_ticker_change_and_etfs_are_classified():
    assert permanent_security_id("XNAS", "Example Corporation", "OLD") == permanent_security_id(
        "XNAS", "Example Corporation", "NEW"
    )
    assert classify_security("Example Innovation ETF") == "ETF"
    assert classify_security("Example REIT") == "REIT"


def test_alpha_vantage_normalization_keeps_operating_types_for_filtering():
    raw = pd.DataFrame(
        {
            "symbol": ["AAA", "FUND"],
            "name": ["Example Corporation", "Example ETF"],
            "exchange": ["NASDAQ", "NASDAQ"],
            "assetType": ["Stock", "ETF"],
            "ipoDate": ["2020-01-01", "2020-01-01"],
            "delistingDate": [pd.NA, pd.NA],
            "status": ["Active", "Active"],
        }
    )
    normalized = normalize_alpha_vantage(raw, pd.Timestamp("2025-01-31"))
    types = normalized.set_index("symbol")["security_type"].to_dict()
    assert types == {"AAA": "Common Stock", "FUND": "ETF"}
