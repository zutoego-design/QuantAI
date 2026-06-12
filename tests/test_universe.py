import pandas as pd

from qss.config.loader import get_config
from qss.universe.builder import build_universe


def test_universe_filters_respect_sector_and_market_cap():
    config = get_config(["configs/default.yaml"])
    as_of = pd.Timestamp("2025-12-31")
    prices = pd.DataFrame(
        {
            "symbol": ["A"] * 5 + ["B"] * 5,
            "date": list(pd.date_range("2025-12-25", periods=5, freq="B")) * 2,
            "adj_close": [98, 99, 100, 100.5, 101, 48, 49, 50, 50.5, 51],
            "volume": [100000, 100000, 100000, 110000, 110000] * 2,
            "return_1d": [0.0, 0.01, 0.01, 0.005, 0.005] * 2,
        }
    )
    fundamentals = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "available_date": [pd.Timestamp("2025-11-01"), pd.Timestamp("2025-11-01")],
            "period_end_date": [pd.Timestamp("2025-09-30"), pd.Timestamp("2025-09-30")],
            "shares_outstanding": [50_000_000, 50_000_000],
        }
    )
    security_master = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "sector": ["Industrials", "Financials"],
            "security_type": ["Common Stock", "Common Stock"],
        }
    )
    config.universe.filters.min_market_cap = 2_000_000_000
    config.universe.filters.min_price = 5.0
    config.universe.filters.min_adv_20d = 1.0
    config.universe.filters.min_history_days = 1
    config.universe.filters.min_price_data_completeness = 0.0
    config.universe.exclude.sectors = ["Financials"]
    universe = build_universe(as_of, prices, fundamentals, security_master, config.universe)
    included = universe.set_index("symbol")["included"].to_dict()
    assert included["A"] is True
    assert included["B"] is False
    assert "sector_excluded" in universe.set_index("symbol").loc["B", "exclusion_reason"]
