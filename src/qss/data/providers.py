from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from qss.data.storage import query_parquet


class UniverseProvider(ABC):
    @abstractmethod
    def snapshot(self, as_of_date: str | pd.Timestamp) -> pd.DataFrame:
        """Return securities eligible for consideration on the requested date."""


class PriceProvider(ABC):
    @abstractmethod
    def fetch(
        self,
        security_ids: list[str],
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
    ) -> pd.DataFrame:
        """Return point-in-time price observations for permanent security identifiers."""


@dataclass
class ParquetPriceProvider(PriceProvider):
    prices_path: Path
    symbol_history_path: Path

    def fetch(
        self,
        security_ids: list[str],
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
    ) -> pd.DataFrame:
        if not security_ids:
            return pd.DataFrame()
        escaped_ids = ", ".join(f"'{value.replace(chr(39), chr(39) * 2)}'" for value in security_ids)
        return query_parquet(
            f"""
            SELECT p.*, h.security_id
            FROM prices p
            JOIN symbols h
              ON p.symbol = h.symbol
             AND p.date >= h.valid_from
             AND p.date <= COALESCE(h.valid_to, DATE '9999-12-31')
            WHERE h.security_id IN ({escaped_ids})
              AND p.date BETWEEN DATE '{pd.Timestamp(start).date()}'
                             AND DATE '{pd.Timestamp(end).date()}'
            ORDER BY p.date, h.security_id
            """,
            prices=self.prices_path,
            symbols=self.symbol_history_path,
        )
