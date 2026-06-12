from __future__ import annotations

import pandas as pd


class NLPFeatureProvider:
    def compute_features(
        self,
        symbols: list[str],
        as_of_date: pd.Timestamp,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "symbol": symbols,
                "date": pd.Timestamp(as_of_date).normalize(),
                "nlp_score": 0.0,
                "nlp_risk_flag": False,
                "nlp_summary": "NLP module is replaced by the bounded research orchestrator in v0.2",
            }
        )
