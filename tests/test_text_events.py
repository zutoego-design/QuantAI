import pandas as pd

from qss.config.loader import get_config
from qss.factors.text_event import compute_risk_disclosure_factor
from qss.ingestion.sec_edgar import _extract_filing_metadata
from qss.ingestion.sec_text import cache_sec_filing_text
from qss.labels.builders import build_event_window_labels
from qss.labels.schema import LabelDefinition
from qss.nlp.interface import NLPFeatureProvider


def test_sec_filing_metadata_and_cached_text_factor_are_reproducible(tmp_path):
    payload = {
        "filings": {
            "recent": {
                "accessionNumber": ["0001-25-000001"],
                "form": ["10-K"],
                "filingDate": ["2025-02-01"],
                "acceptanceDateTime": ["2025-02-01T16:30:00.000Z"],
                "primaryDocument": ["annual.htm"],
            }
        }
    }
    filings = _extract_filing_metadata(payload, "AAA", "0000000001")
    assert {
        "filing_type",
        "filing_timestamp",
        "event_type",
        "text_cache_key",
    }.issubset(filings.columns)
    provider = NLPFeatureProvider(tmp_path, ["material weakness"])
    cache_key = filings.iloc[0]["text_cache_key"]
    provider.cache_text(cache_key, "Material weakness material weakness.")
    first = compute_risk_disclosure_factor(
        pd.Timestamp("2025-03-01"),
        ["AAA"],
        filings,
        tmp_path,
        ["material weakness"],
    )
    second = provider.compute_features(["AAA"], pd.Timestamp("2025-03-01"), filings)
    assert first.iloc[0]["risk_disclosure_score"] > 0
    assert first.iloc[0]["risk_disclosure_score"] == second.iloc[0]["risk_disclosure_score"]


def test_event_window_labels_use_filing_timestamp():
    dates = pd.bdate_range("2025-01-01", periods=20)
    prices = pd.DataFrame(
        {
            "date": dates,
            "symbol": "AAA",
            "adj_close": range(100, 120),
        }
    )
    events = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "filing_timestamp": [pd.Timestamp("2025-01-08 16:30")],
        }
    )
    labels = build_event_window_labels(
        events,
        prices,
        LabelDefinition(name="event_window_return", horizon_days=5),
    )
    assert not labels.empty
    assert labels.iloc[0]["label_end_time"] > labels.iloc[0]["label_start_time"]


def test_sec_text_ingestion_uses_stable_cache_key(tmp_path):
    class Response:
        text = "<html><body>Material weakness disclosed.</body></html>"

        @staticmethod
        def raise_for_status():
            return None

    config = get_config(["configs/default.yaml"])
    config.paths.silver_data = str(tmp_path / "silver")
    config.text_factors.cache_directory = str(tmp_path / "cache")
    filings = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "cik": ["0000000001"],
            "filing_type": ["10-K"],
            "filing_timestamp": [pd.Timestamp("2025-02-01")],
            "event_type": ["annual_report"],
            "accession": ["0001-25-000001"],
            "primary_document": ["annual.htm"],
            "text_cache_key": ["stable-key"],
            "source": ["sec_edgar_submissions"],
            "ingestion_time": [pd.Timestamp("2025-02-01")],
        }
    )
    cached = cache_sec_filing_text(
        config,
        filings,
        fetcher=lambda *args, **kwargs: Response(),
    )
    assert list(cached["text_cache_key"]) == ["stable-key"]
    assert bool(cached.iloc[0]["text_cached"])
    assert (tmp_path / "cache" / "stable-key.txt").read_text(
        encoding="utf-8"
    ) == "Material weakness disclosed."


def test_text_factor_keeps_uncached_filings_missing(tmp_path):
    filings = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "filing_type": ["10-K"],
            "filing_timestamp": [pd.Timestamp("2025-02-01")],
            "text_cache_key": ["not-cached"],
        }
    )

    result = compute_risk_disclosure_factor(
        pd.Timestamp("2025-03-01"),
        ["AAA"],
        filings,
        tmp_path,
        ["material weakness"],
    )

    assert pd.isna(result.iloc[0]["risk_disclosure_score"])
