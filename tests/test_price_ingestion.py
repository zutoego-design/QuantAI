import pandas as pd
import pytest

from qss.config.loader import get_config
from qss.data.identifiers import normalize_symbol
from qss.data.storage import write_parquet
from qss.ingestion.prices_yfinance import (
    PRICE_COLUMNS,
    YFinancePriceProvider,
    _retry_yfinance_symbols,
    _stooq_supports,
    ingest_prices,
)


class _FlakyProvider:
    def __init__(self):
        self.calls = 0

    def fetch(self, tickers, start_date, end_date=None):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("temporary failure")
        symbol = tickers[0]
        columns = pd.MultiIndex.from_product(
            [[symbol], ["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
        )
        return pd.DataFrame(
            [[1.0, 1.0, 1.0, 1.0, 1.0, 100]],
            index=pd.to_datetime(["2025-01-02"]),
            columns=columns,
        ).rename_axis("Date")

    def normalize(self, raw_data):
        symbol = raw_data.columns.get_level_values(0)[0]
        return pd.DataFrame(
            {
                "symbol": [symbol],
                "date": [pd.Timestamp("2025-01-02")],
                "open": [1.0],
                "high": [1.0],
                "low": [1.0],
                "close": [1.0],
                "adj_close": [1.0],
                "volume": [100],
                "return_1d": [pd.NA],
                "source": ["yfinance"],
                "quality_status": ["live"],
                "ingestion_time": [pd.Timestamp("2025-01-03")],
            }
        )[PRICE_COLUMNS]


def test_yahoo_individual_retry_recovers_transient_failure():
    provider = _FlakyProvider()
    result = _retry_yfinance_symbols(
        provider,
        ["^IXIC"],
        "2025-01-01",
        "2025-02-01",
    )
    assert provider.calls == 2
    assert list(result["symbol"]) == ["^IXIC"]


def test_stooq_fallback_skips_exchange_indices():
    assert normalize_symbol("^IXIC") == "^IXIC"
    assert _stooq_supports("AAPL") is True
    assert _stooq_supports("^IXIC") is False


def test_yahoo_class_share_symbol_round_trips():
    columns = pd.MultiIndex.from_product(
        [["BRK-B"], ["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    )
    raw = pd.DataFrame(
        [[1.0, 1.0, 1.0, 1.0, 1.0, 100]],
        index=pd.to_datetime(["2025-01-02"]),
        columns=columns,
    ).rename_axis("Date")

    result = YFinancePriceProvider().normalize(raw)

    assert normalize_symbol("BRK-B") == "BRK.B"
    assert list(result["symbol"]) == ["BRK.B"]


def test_strict_ingestion_saves_partial_live_progress_before_raising(
    tmp_path,
    monkeypatch,
):
    config = get_config(["configs/default.yaml"]).model_copy(deep=True)
    config.paths.raw_data = str(tmp_path / "raw")
    config.paths.silver_data = str(tmp_path / "silver")
    config.paths.reports = str(tmp_path / "reports")
    config.universe.min_long_price_coverage = 0.95
    master = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "is_active": [True, True],
            "source": ["test", "test"],
            "ingestion_time": [pd.Timestamp("2025-01-03")] * 2,
        }
    )
    write_parquet(
        master,
        tmp_path / "silver" / "universe" / "security_master.parquet",
    )

    def fake_fetch(self, tickers, start_date, end_date=None):
        available = [symbol for symbol in tickers if symbol == "AAA"]
        if not available:
            return pd.DataFrame()
        columns = pd.MultiIndex.from_product(
            [available, ["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
        )
        return pd.DataFrame(
            [[1.0] * len(columns)],
            index=pd.to_datetime(["2025-01-02"]),
            columns=columns,
        ).rename_axis("Date")

    monkeypatch.setattr(YFinancePriceProvider, "fetch", fake_fetch)
    monkeypatch.setattr(
        "qss.ingestion.prices_yfinance.StooqPriceProvider.fetch",
        lambda *args, **kwargs: pd.DataFrame(),
    )
    monkeypatch.setattr(
        "qss.ingestion.prices_yfinance._flatten_etf_tickers",
        lambda config: [],
    )

    with pytest.raises(RuntimeError, match="Successful live rows were saved"):
        ingest_prices(
            config,
            start_date="2025-01-01",
            end_date="2025-02-01",
            tickers=["AAA", "BBB"],
        )

    saved = pd.read_parquet(
        tmp_path / "silver" / "prices" / "prices_daily.parquet"
    )
    assert set(saved["symbol"]) == {"AAA"}


def test_strict_ingestion_counts_cached_live_symbols(
    tmp_path,
    monkeypatch,
):
    config = get_config(["configs/default.yaml"]).model_copy(deep=True)
    config.paths.raw_data = str(tmp_path / "raw")
    config.paths.silver_data = str(tmp_path / "silver")
    config.paths.reports = str(tmp_path / "reports")
    config.universe.min_long_price_coverage = 0.50
    master = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "is_active": [True, True],
            "source": ["test", "test"],
            "ingestion_time": [pd.Timestamp("2025-01-03")] * 2,
        }
    )
    write_parquet(
        master,
        tmp_path / "silver" / "universe" / "security_master.parquet",
    )
    cached = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "date": [pd.Timestamp("2025-01-02")],
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "adj_close": [1.0],
            "volume": [100],
            "return_1d": [pd.NA],
            "source": ["yfinance"],
            "quality_status": ["live"],
            "ingestion_time": [pd.Timestamp("2025-01-03")],
        }
    )[PRICE_COLUMNS]
    write_parquet(
        cached,
        tmp_path / "silver" / "prices" / "prices_daily.parquet",
    )

    monkeypatch.setattr(
        YFinancePriceProvider,
        "fetch",
        lambda *args, **kwargs: pd.DataFrame(),
    )
    monkeypatch.setattr(
        "qss.ingestion.prices_yfinance.StooqPriceProvider.fetch",
        lambda *args, **kwargs: pd.DataFrame(),
    )
    monkeypatch.setattr(
        "qss.ingestion.prices_yfinance._flatten_etf_tickers",
        lambda config: [],
    )

    ingest_prices(
        config,
        start_date="2025-01-01",
        end_date="2025-02-01",
        tickers=["AAA", "BBB"],
    )


def test_incremental_ingestion_recomputes_boundary_returns(
    tmp_path,
    monkeypatch,
):
    config = get_config(["configs/default.yaml"]).model_copy(deep=True)
    config.paths.raw_data = str(tmp_path / "raw")
    config.paths.silver_data = str(tmp_path / "silver")
    config.paths.reports = str(tmp_path / "reports")
    master = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "is_active": [True],
            "source": ["test"],
            "ingestion_time": [pd.Timestamp("2025-01-01")],
        }
    )
    write_parquet(
        master,
        tmp_path / "silver" / "universe" / "security_master.parquet",
    )
    cached = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA"],
            "date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            "open": [100.0, 101.0],
            "high": [100.0, 101.0],
            "low": [100.0, 101.0],
            "close": [100.0, 101.0],
            "adj_close": [100.0, 101.0],
            "volume": [100, 100],
            "return_1d": [pd.NA, 0.01],
            "source": ["yfinance", "yfinance"],
            "quality_status": ["live", "live"],
            "ingestion_time": [pd.Timestamp("2025-01-02")] * 2,
        }
    )[PRICE_COLUMNS]
    write_parquet(
        cached,
        tmp_path / "silver" / "prices" / "prices_daily.parquet",
    )

    def fake_fetch(self, tickers, start_date, end_date=None):
        columns = pd.MultiIndex.from_product(
            [["AAA"], ["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
        )
        return pd.DataFrame(
            [
                [101.0, 101.0, 101.0, 101.0, 101.0, 100],
                [103.02, 103.02, 103.02, 103.02, 103.02, 100],
            ],
            index=pd.to_datetime(["2025-01-02", "2025-01-03"]),
            columns=columns,
        ).rename_axis("Date")

    monkeypatch.setattr(YFinancePriceProvider, "fetch", fake_fetch)
    monkeypatch.setattr(
        "qss.ingestion.prices_yfinance._flatten_etf_tickers",
        lambda config: [],
    )

    ingest_prices(
        config,
        start_date="2025-01-02",
        end_date="2025-01-04",
        tickers=["AAA"],
    )

    saved = pd.read_parquet(
        tmp_path / "silver" / "prices" / "prices_daily.parquet"
    ).sort_values("date")
    returns = saved.set_index("date")["return_1d"]
    assert returns.loc[pd.Timestamp("2025-01-02")] == pytest.approx(0.01)
    assert returns.loc[pd.Timestamp("2025-01-03")] == pytest.approx(0.02)
