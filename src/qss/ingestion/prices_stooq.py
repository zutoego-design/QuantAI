from __future__ import annotations

import io

import pandas as pd
import requests

from qss.data.identifiers import normalize_symbol
from qss.ingestion.base import DataProvider


class StooqPriceProvider(DataProvider):
    def fetch(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        url = f"https://stooq.com/q/d/l/?s={symbol.lower()}.us&i=d"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = pd.read_csv(io.StringIO(response.text))
        if data.empty:
            return data
        data["symbol"] = normalize_symbol(symbol)
        return data

    def normalize(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        if raw_data.empty:
            return pd.DataFrame(
                columns=[
                    "symbol",
                    "date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "adj_close",
                    "volume",
                    "return_1d",
                    "source",
                    "quality_status",
                    "ingestion_time",
                ]
            )
        frame = raw_data.rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        frame["date"] = pd.to_datetime(frame["date"])
        frame["adj_close"] = frame["close"]
        frame["return_1d"] = frame["adj_close"].pct_change()
        frame["source"] = "stooq"
        frame["quality_status"] = "fallback_live"
        frame["ingestion_time"] = pd.Timestamp.utcnow().tz_localize(None)
        return frame[
            [
                "symbol",
                "date",
                "open",
                "high",
                "low",
                "close",
                "adj_close",
                "volume",
                "return_1d",
                "source",
                "quality_status",
                "ingestion_time",
            ]
        ]
