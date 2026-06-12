from __future__ import annotations

from qss.config.schema import AppConfig
from qss.ingestion.prices_yfinance import ingest_prices


def ingest_etf_proxies(config: AppConfig, start_date: str, end_date: str | None = None):
    tickers: list[str] = []
    for ticker_map in config.data_sources.etf.tickers.values():
        tickers.extend(ticker_map.values())
    return ingest_prices(config, start_date=start_date, end_date=end_date, tickers=sorted(set(tickers)))
