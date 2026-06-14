import gzip
import json

import pandas as pd

from qss.config.loader import get_config
from qss.ingestion.sec_edgar import _sic_to_sector, ingest_fundamentals


def _company_facts():
    return {
        "sic": "7372",
        "sicDescription": "Software",
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {
                                "form": "10-K",
                                "end": "2025-12-31",
                                "filed": "2026-02-01",
                                "fy": 2025,
                                "fp": "FY",
                                "accn": "0001",
                                "val": 100.0,
                            }
                        ]
                    }
                }
            }
        },
    }


def _company_submission():
    return {
        "sic": "7372",
        "sicDescription": "Prepackaged Software",
    }


def test_fundamentals_reuses_cache_and_skips_unmapped_symbols(
    tmp_path,
    monkeypatch,
):
    config = get_config(["configs/default.yaml"])
    config.paths.raw_data = str(tmp_path / "raw")
    config.paths.silver_data = str(tmp_path / "silver")
    config.paths.reports = str(tmp_path / "reports")
    cache = tmp_path / "raw" / "fundamentals" / "20260612" / "AAA.json.gz"
    cache.parent.mkdir(parents=True)
    with gzip.open(cache, "wt", encoding="utf-8") as handle:
        json.dump(_company_facts(), handle)
    submission_cache = (
        tmp_path
        / "raw"
        / "fundamentals_submissions"
        / "20260612"
        / "AAA.json.gz"
    )
    submission_cache.parent.mkdir(parents=True)
    with gzip.open(submission_cache, "wt", encoding="utf-8") as handle:
        json.dump(_company_submission(), handle)

    monkeypatch.setattr(
        "qss.ingestion.sec_edgar._load_ticker_cik_map",
        lambda user_agent: {"AAA": "0000000001"},
    )

    def _unexpected_request(*args, **kwargs):
        raise AssertionError("fresh cached SEC data should avoid network requests")

    monkeypatch.setattr("qss.ingestion.sec_edgar.requests.get", _unexpected_request)

    result = ingest_fundamentals(config, ["AAA", "OLD"])

    assert result.requested == 2
    assert result.mapped == 1
    assert result.cached == 1
    assert result.fetched == 0
    assert result.no_mapping == 1
    assert set(result.fundamentals["symbol"]) == {"AAA"}


def test_fundamentals_refetches_truncated_cache(tmp_path, monkeypatch):
    config = get_config(["configs/default.yaml"])
    config.paths.raw_data = str(tmp_path / "raw")
    config.paths.silver_data = str(tmp_path / "silver")
    config.paths.reports = str(tmp_path / "reports")
    cache = tmp_path / "raw" / "fundamentals" / "20260612" / "AAA.json.gz"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"\x1f\x8b\x08")

    monkeypatch.setattr(
        "qss.ingestion.sec_edgar._load_ticker_cik_map",
        lambda user_agent: {"AAA": "0000000001"},
    )

    class _Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    def _get(url, *args, **kwargs):
        payload = (
            _company_submission()
            if "/submissions/" in url
            else _company_facts()
        )
        return _Response(payload)

    monkeypatch.setattr("qss.ingestion.sec_edgar.requests.get", _get)

    result = ingest_fundamentals(config, ["AAA"])

    assert result.cached == 0
    assert result.fetched == 1
    assert set(result.fundamentals["symbol"]) == {"AAA"}


def test_sic_mapping_covers_common_industries():
    assert _sic_to_sector("1311") == "Energy"
    assert _sic_to_sector("3312") == "Materials"
    assert _sic_to_sector("3711") == "Industrials"
    assert _sic_to_sector("3826") == "Information Technology"
    assert _sic_to_sector("8731") == "Industrials"


def test_unknown_sec_sector_does_not_replace_existing_gics(
    tmp_path,
    monkeypatch,
):
    config = get_config(["configs/default.yaml"])
    config.paths.raw_data = str(tmp_path / "raw")
    config.paths.silver_data = str(tmp_path / "silver")
    config.paths.reports = str(tmp_path / "reports")
    cache = tmp_path / "raw" / "fundamentals" / "20260612" / "AAA.json.gz"
    cache.parent.mkdir(parents=True)
    with gzip.open(cache, "wt", encoding="utf-8") as handle:
        json.dump(_company_facts(), handle)
    submission_cache = (
        tmp_path
        / "raw"
        / "fundamentals_submissions"
        / "20260612"
        / "AAA.json.gz"
    )
    submission_cache.parent.mkdir(parents=True)
    with gzip.open(submission_cache, "wt", encoding="utf-8") as handle:
        json.dump({"sic": None, "sicDescription": None}, handle)
    master_path = tmp_path / "silver" / "universe" / "security_master.parquet"
    master_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "symbol": ["AAA"],
            "sector": ["Information Technology"],
        }
    ).to_parquet(master_path, index=False)

    monkeypatch.setattr(
        "qss.ingestion.sec_edgar._load_ticker_cik_map",
        lambda user_agent: {"AAA": "0000000001"},
    )
    monkeypatch.setattr(
        "qss.ingestion.sec_edgar.requests.get",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("fresh cached SEC data should avoid network requests")
        ),
    )

    ingest_fundamentals(config, ["AAA"])

    saved = pd.read_parquet(master_path)
    assert saved.loc[0, "sector"] == "Information Technology"
