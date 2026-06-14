import pandas as pd

from qss.config.loader import get_config
from qss.data.storage import write_parquet
from qss.universe.sector_enrichment import enrich_sector_metadata


class _DetailsProvider:
    api_key = "test"

    def __init__(self):
        self.calls = []

    def fetch_details(self, symbol, as_of_date):
        self.calls.append((symbol, pd.Timestamp(as_of_date)))
        return {
            "symbol": symbol,
            "as_of_date": pd.Timestamp(as_of_date),
            "cik": "0000000001",
            "sic": "7372",
            "sic_description": "Software",
            "name": symbol,
            "active": False,
            "metadata_source": "massive_ticker_details",
            "metadata_ingestion_time": pd.Timestamp("2026-06-13"),
        }


def test_sector_enrichment_stops_after_reaching_threshold(tmp_path):
    config = get_config(["configs/default.yaml"]).model_copy(deep=True)
    config.paths.silver_data = str(tmp_path / "silver")
    config.universe.min_sector_coverage = 0.75
    config.universe.remote_request_interval_seconds = 0
    membership = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-31",
                    "2025-02-28",
                    "2025-01-31",
                    "2025-02-28",
                    "2025-01-31",
                    "2025-02-28",
                    "2025-01-31",
                    "2025-02-28",
                ]
            ),
            "symbol": ["AAA", "AAA", "BBB", "BBB", "CCC", "CCC", "DDD", "DDD"],
            "included": [True] * 8,
        }
    )
    master = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC", "DDD"],
            "sector": ["Information Technology", "Unknown", "Unknown", "Unknown"],
        }
    )
    root = tmp_path / "silver" / "universe"
    write_parquet(membership, root / "universe_membership.parquet")
    write_parquet(master, root / "security_master.parquet")
    provider = _DetailsProvider()

    result = enrich_sector_metadata(
        config,
        start_date="2025-01-01",
        end_date="2025-02-28",
        provider=provider,
        sleep_fn=lambda seconds: None,
    )

    assert result.classified == 2
    assert result.coverage == 0.75
    assert len(provider.calls) == 2
    saved = pd.read_parquet(root / "security_master.parquet")
    known = ~saved["sector"].fillna("").str.lower().isin(["", "unknown"])
    assert known.sum() == 3
