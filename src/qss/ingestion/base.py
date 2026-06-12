from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd

from qss.data.storage import ensure_data_directories, write_parquet
from qss.utils import ensure_parent


class DataProvider(ABC):
    @abstractmethod
    def fetch(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def normalize(self, raw_data: Any) -> pd.DataFrame:
        raise NotImplementedError

    def save_raw(self, raw_data: Any, path: str | Path) -> None:
        target = ensure_parent(Path(path))
        if isinstance(raw_data, pd.DataFrame):
            raw_data.to_csv(target, index=False)
            return
        if isinstance(raw_data, (dict, list)):
            target.write_text(json.dumps(raw_data, indent=2), encoding="utf-8")
            return
        target.write_text(str(raw_data), encoding="utf-8")

    def save_silver(self, data: pd.DataFrame, path: str | Path) -> None:
        write_parquet(data, path)


__all__ = ["DataProvider", "ensure_data_directories"]
