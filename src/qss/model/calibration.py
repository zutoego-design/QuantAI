from __future__ import annotations

import pandas as pd


def clip_score_extremes(scores: pd.DataFrame, score_columns: list[str]) -> pd.DataFrame:
    out = scores.copy()
    for column in score_columns:
        out[column] = out[column].clip(lower=out[column].quantile(0.01), upper=out[column].quantile(0.99))
    return out
