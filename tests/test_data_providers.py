import pandas as pd

from qss.data.providers import ParquetPriceProvider
from qss.data.storage import append_with_source_precedence
from qss.universe.providers import ParquetUniverseProvider


def test_source_precedence_prevents_lower_quality_overwrite(tmp_path):
    path = tmp_path / "prices.parquet"
    live = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "date": [pd.Timestamp("2025-01-02")],
            "adj_close": [100.0],
            "source": ["yfinance"],
            "ingestion_time": [pd.Timestamp("2025-01-03")],
        }
    )
    fallback = live.assign(
        adj_close=90.0,
        source="stooq",
        ingestion_time=pd.Timestamp("2025-01-04"),
    )
    append_with_source_precedence(
        live, path, ["symbol", "date"], {"yfinance": 100, "stooq": 80}
    )
    append_with_source_precedence(
        fallback, path, ["symbol", "date"], {"yfinance": 100, "stooq": 80}
    )
    saved = pd.read_parquet(path)
    assert saved.iloc[0]["adj_close"] == 100.0
    assert saved.iloc[0]["source"] == "yfinance"


def test_duckdb_price_provider_resolves_ticker_history(tmp_path):
    prices_path = tmp_path / "prices.parquet"
    history_path = tmp_path / "history.parquet"
    pd.DataFrame(
        {
            "symbol": ["OLD", "NEW"],
            "date": pd.to_datetime(["2024-01-02", "2025-01-02"]),
            "adj_close": [10.0, 20.0],
        }
    ).to_parquet(prices_path, index=False)
    pd.DataFrame(
        {
            "security_id": ["sec_1", "sec_1"],
            "symbol": ["OLD", "NEW"],
            "valid_from": pd.to_datetime(["2020-01-01", "2024-06-01"]),
            "valid_to": pd.to_datetime(["2024-05-31", "2030-01-01"]),
        }
    ).to_parquet(history_path, index=False)
    result = ParquetPriceProvider(prices_path, history_path).fetch(
        ["sec_1"], "2024-01-01", "2025-12-31"
    )
    assert list(result["symbol"]) == ["OLD", "NEW"]
    assert set(result["security_id"]) == {"sec_1"}


def test_partitioned_universe_provider_uses_latest_snapshot(tmp_path):
    root = tmp_path / "membership"
    for year, date, symbols in [
        (2024, "2024-12-31", ["OLD"]),
        (2025, "2025-01-31", ["NEW"]),
    ]:
        target = root / f"year={year}"
        target.mkdir(parents=True)
        pd.DataFrame(
            {
                "date": pd.Timestamp(date),
                "security_id": [f"sec_{symbol}" for symbol in symbols],
                "symbol": symbols,
                "included": True,
            }
        ).to_parquet(target / "part.parquet", index=False)
    result = ParquetUniverseProvider(root).snapshot("2025-02-15")
    assert list(result["symbol"]) == ["NEW"]
