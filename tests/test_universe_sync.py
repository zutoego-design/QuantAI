import pandas as pd
import pytest
import requests

from qss.config.loader import get_config
from qss.data.storage import write_parquet
from qss.universe.providers import (
    AlphaVantageListingProvider,
    NasdaqTraderProvider,
)
from qss.universe.sp500 import _monthly_trading_ends
from qss.universe.sync import sync_universe


class _Response:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        return None

    def json(self):
        return {}


def _listing(date, source="alpha_vantage_listing_status"):
    return pd.DataFrame(
        {
            "date": [pd.Timestamp(date)],
            "security_id": ["sec_aaa"],
            "symbol": ["AAA"],
            "name": ["AAA Inc"],
            "exchange": ["NASDAQ"],
            "security_type": ["Common Stock"],
            "listing_date": [pd.Timestamp("2000-01-01")],
            "delisting_date": [pd.NaT],
            "status": ["Active"],
            "source": [source],
        }
    )


def test_alpha_vantage_empty_json_is_retried(monkeypatch):
    csv_text = (
        "symbol,name,exchange,assetType,ipoDate,delistingDate,status\n"
        "AAA,AAA Inc,NASDAQ,Stock,2000-01-01,null,Active\n"
    )
    responses = iter([_Response("{}"), _Response(csv_text)])
    monkeypatch.setattr(
        "qss.universe.providers.requests.get",
        lambda *args, **kwargs: next(responses),
    )
    monkeypatch.setattr("qss.universe.providers.time.sleep", lambda seconds: None)
    result = AlphaVantageListingProvider(
        api_key="test",
        retry_delays=(0.0,),
    ).fetch("2025-01-31")
    assert list(result["symbol"]) == ["AAA"]


def test_alpha_vantage_network_error_is_retried_and_redacted(monkeypatch):
    calls = 0

    def _fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise requests.exceptions.SSLError(
            "https://example.test?apikey=secret"
        )

    monkeypatch.setattr("qss.universe.providers.requests.get", _fail)
    monkeypatch.setattr("qss.universe.providers.time.sleep", lambda seconds: None)
    provider = AlphaVantageListingProvider(
        api_key="secret",
        retry_delays=(0.0,),
    )
    with pytest.raises(RuntimeError, match="credentials were redacted") as exc_info:
        provider.fetch("2025-01-31")
    assert calls == 2
    assert "secret" not in str(exc_info.value)


def test_nasdaq_trader_ignores_timestamped_footer(monkeypatch):
    response = _Response(
        "Symbol|Security Name|Test Issue|ETF|Financial Status\n"
        "AAA|AAA Inc|N|N|N\n"
        "File Creation Time: 0612202609:01||N|N|N\n"
    )
    monkeypatch.setattr(
        "qss.universe.providers.requests.get",
        lambda *args, **kwargs: response,
    )
    result = NasdaqTraderProvider().fetch()
    assert list(result["symbol"]) == ["AAA"]
    assert result["security_id"].notna().all()


def test_sync_preserves_cached_progress_when_provider_pauses(tmp_path, monkeypatch):
    config = get_config(["configs/default.yaml"])
    config.universe.membership_mode = "point_in_time"
    config.universe.long_history_provider = "alpha_vantage"
    config.universe.validation_provider = "massive"
    config.paths.raw_data = str(tmp_path / "raw")
    config.paths.silver_data = str(tmp_path / "silver")
    config.universe.remote_request_interval_seconds = 0
    cache = tmp_path / "raw" / "universe" / "alpha_vantage" / "2010-01-31.parquet"
    write_parquet(_listing("2010-01-31"), cache)

    class _PausedAlpha:
        def fetch(self, date):
            raise RuntimeError("temporary throttle")

    class _CurrentNasdaq:
        def fetch(self):
            return _listing("2010-03-31", source="nasdaq_trader")

    monkeypatch.setattr(
        "qss.universe.sync.AlphaVantageListingProvider",
        _PausedAlpha,
    )
    monkeypatch.setattr("qss.universe.sync.NasdaqTraderProvider", _CurrentNasdaq)

    result = sync_universe(
        config,
        start_date="2010-01-01",
        end_date="2010-03-31",
        validate_recent=False,
    )
    assert result.historical_months == 1
    assert result.next_missing_date == "2010-02-28"
    assert "Cached progress was preserved" in result.warning
    assert not result.membership.empty


def test_sync_is_idempotent_with_cached_history(tmp_path, monkeypatch):
    config = get_config(["configs/default.yaml"])
    config.universe.membership_mode = "point_in_time"
    config.universe.long_history_provider = "alpha_vantage"
    config.universe.validation_provider = "massive"
    config.paths.raw_data = str(tmp_path / "raw")
    config.paths.silver_data = str(tmp_path / "silver")
    config.universe.remote_request_interval_seconds = 0
    cache = tmp_path / "raw" / "universe" / "alpha_vantage" / "2010-01-31.parquet"
    write_parquet(_listing("2010-01-31"), cache)

    class _UnusedAlpha:
        def fetch(self, date):
            raise AssertionError("cached history should not be refetched")

    class _CurrentNasdaq:
        def fetch(self):
            return _listing("2010-01-31", source="nasdaq_trader")

    monkeypatch.setattr(
        "qss.universe.sync.AlphaVantageListingProvider",
        _UnusedAlpha,
    )
    monkeypatch.setattr("qss.universe.sync.NasdaqTraderProvider", _CurrentNasdaq)

    first = sync_universe(
        config,
        start_date="2010-01-01",
        end_date="2010-01-31",
        validate_recent=False,
    )
    second = sync_universe(
        config,
        start_date="2010-01-01",
        end_date="2010-01-31",
        validate_recent=False,
    )

    keys = ["date", "security_id", "symbol", "source"]
    assert len(second.membership) == len(first.membership)
    assert second.membership.duplicated(keys).sum() == 0


def test_current_snapshot_mode_backfills_without_historical_providers(
    tmp_path,
    monkeypatch,
):
    config = get_config(["configs/default.yaml"])
    config.universe.membership_mode = "current_snapshot"
    config.universe.long_history_provider = "nasdaq_trader_current"
    config.paths.raw_data = str(tmp_path / "raw")
    config.paths.silver_data = str(tmp_path / "silver")

    class _CurrentNasdaq:
        calls = 0

        def fetch(self):
            self.calls += 1
            return _listing("2025-03-15", source="nasdaq_trader")

    monkeypatch.setattr("qss.universe.sync.NasdaqTraderProvider", _CurrentNasdaq)
    monkeypatch.setattr(
        "qss.universe.sync.AlphaVantageListingProvider",
        lambda: (_ for _ in ()).throw(
            AssertionError("Alpha Vantage must not be used")
        ),
    )
    monkeypatch.setattr(
        "qss.universe.sync.MassiveTickerProvider",
        lambda: (_ for _ in ()).throw(
            AssertionError("Massive must not be used")
        ),
    )

    result = sync_universe(
        config,
        start_date="2025-01-01",
        end_date="2025-03-31",
    )

    assert result.historical_months == 3
    assert result.requested_months == 3
    assert result.membership["date"].nunique() == 3
    assert result.membership["date"].min() == pd.Timestamp("2025-01-01")
    assert set(result.membership["source"]) == {
        "nasdaq_trader_current_backfill"
    }
    assert "survivorship bias" in result.warning


def test_sp500_sync_reconstructs_membership_from_changes(tmp_path, monkeypatch):
    config = get_config(["configs/default.yaml"])
    config.paths.raw_data = str(tmp_path / "raw")
    config.paths.silver_data = str(tmp_path / "silver")
    config.universe.start_date = "2025-01-01"
    html = """
    <table id="constituents">
      <tr><th>Symbol</th><th>Security</th><th>GICS Sector</th><th>GICS Sub-Industry</th><th>Date added</th><th>CIK</th></tr>
      <tr><td>AAA</td><td>AAA Corp</td><td>Information Technology</td><td>Software</td><td>2020-01-01</td><td>0000000001</td></tr>
      <tr><td>CCC</td><td>CCC Corp</td><td>Health Care</td><td>Equipment</td><td>2020-01-01</td><td>0000000003</td></tr>
      <tr><td>DDD</td><td>DDD Corp</td><td>Industrials</td><td>Machinery</td><td>2025-02-15</td><td>0000000004</td></tr>
    </table>
    <table id="changes">
      <tr><th>Effective Date</th><th>Added</th><th>Removed</th><th>Reason</th></tr>
      <tr><th>Ticker</th><th>Security</th><th>Ticker</th><th>Security</th></tr>
      <tr><td>February 15, 2025</td><td>DDD</td><td>DDD Corp</td><td>BBB</td><td>BBB Corp</td><td>Index change.</td></tr>
    </table>
    """
    monkeypatch.setattr("qss.universe.sp500._fetch_wikipedia_html", lambda: html)

    result = sync_universe(
        config,
        start_date="2025-01-01",
        end_date="2025-03-31",
    )
    january = set(
        result.membership.loc[
            result.membership["date"] == pd.Timestamp("2025-01-31"),
            "symbol",
        ]
    )
    march = set(
        result.membership.loc[
            result.membership["date"] == pd.Timestamp("2025-03-31"),
            "symbol",
        ]
    )

    assert january == {"AAA", "BBB", "CCC"}
    assert march == {"AAA", "CCC", "DDD"}
    assert result.historical_months == 3
    assert set(result.membership["source"]) == {"sp500_wikipedia_point_in_time"}


def test_sp500_monthly_snapshots_use_nyse_trading_month_end():
    dates = _monthly_trading_ends(
        pd.Timestamp("2016-01-01"),
        pd.Timestamp("2016-03-31"),
    )
    good_friday = _monthly_trading_ends(
        pd.Timestamp("2024-03-01"),
        pd.Timestamp("2024-03-31"),
    )

    assert dates == [
        pd.Timestamp("2016-01-29"),
        pd.Timestamp("2016-02-29"),
        pd.Timestamp("2016-03-31"),
    ]
    assert good_friday == [pd.Timestamp("2024-03-28")]
