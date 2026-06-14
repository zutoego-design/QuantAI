from __future__ import annotations

from pathlib import Path

import pandas as pd

from qss.factors.text_event import compute_risk_disclosure_factor


class NLPFeatureProvider:
    def __init__(
        self,
        cache_directory: str | Path = "data/raw/sec_text",
        risk_terms: list[str] | None = None,
        lookback_days: int = 365,
    ):
        self.cache_directory = Path(cache_directory)
        self.risk_terms = risk_terms or [
            "material weakness",
            "going concern",
            "litigation",
            "cybersecurity",
            "restatement",
            "default",
        ]
        self.lookback_days = lookback_days

    def compute_features(
        self,
        symbols: list[str],
        as_of_date: pd.Timestamp,
        filings: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        features = compute_risk_disclosure_factor(
            as_of_date,
            symbols,
            filings if filings is not None else pd.DataFrame(),
            self.cache_directory,
            self.risk_terms,
            self.lookback_days,
        )
        features["date"] = pd.Timestamp(as_of_date).normalize()
        features["nlp_score"] = -features["risk_disclosure_score"]
        features["nlp_risk_flag"] = features["risk_disclosure_score"].fillna(0) >= 2.0
        features["nlp_summary"] = "Deterministic cached SEC filing risk disclosure score."
        return features

    def cache_text(self, text_cache_key: str, text: str) -> Path:
        if not text_cache_key.strip():
            raise ValueError("text_cache_key must not be empty")
        self.cache_directory.mkdir(parents=True, exist_ok=True)
        target = self.cache_directory / f"{text_cache_key}.txt"
        target.write_text(text, encoding="utf-8")
        return target
