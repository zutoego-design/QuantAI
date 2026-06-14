from __future__ import annotations

from pathlib import Path

import pandas as pd

FORM_SEVERITY = {
    "8-K": 1.0,
    "10-Q": 0.4,
    "10-K": 0.6,
    "10-K/A": 0.9,
    "10-Q/A": 0.8,
}


def compute_risk_disclosure_factor(
    as_of_date: pd.Timestamp,
    symbols: list[str],
    filings: pd.DataFrame,
    cache_directory: str | Path,
    risk_terms: list[str],
    lookback_days: int = 365,
) -> pd.DataFrame:
    if filings.empty:
        return pd.DataFrame(
            {"symbol": symbols, "risk_disclosure_score": [float("nan")] * len(symbols)}
        )
    frame = filings.copy()
    frame["filing_timestamp"] = pd.to_datetime(frame["filing_timestamp"]).dt.tz_localize(None)
    cutoff = pd.Timestamp(as_of_date).tz_localize(None) - pd.Timedelta(days=lookback_days)
    frame = frame.loc[
        (frame["symbol"].isin(symbols))
        & (frame["filing_timestamp"] <= pd.Timestamp(as_of_date).tz_localize(None))
        & (frame["filing_timestamp"] > cutoff)
    ]
    cache_root = Path(cache_directory)
    rows = []
    for symbol in symbols:
        company = frame.loc[frame["symbol"] == symbol]
        cached_company = company.loc[
            company["text_cache_key"].astype(str).map(
                lambda key: bool(key) and (cache_root / f"{key}.txt").exists()
            )
        ]
        if cached_company.empty:
            rows.append({"symbol": symbol, "risk_disclosure_score": float("nan")})
            continue
        scores = []
        for filing in cached_company.itertuples(index=False):
            form_score = FORM_SEVERITY.get(str(filing.filing_type), 0.2)
            text_score = 0.0
            cache_key = getattr(filing, "text_cache_key", "")
            cache_path = cache_root / f"{cache_key}.txt"
            text = cache_path.read_text(encoding="utf-8", errors="ignore").lower()
            words = max(len(text.split()), 1)
            matches = sum(text.count(term.lower()) for term in risk_terms)
            text_score = min(matches * 1000.0 / words, 10.0)
            scores.append(form_score + text_score)
        rows.append(
            {
                "symbol": symbol,
                "risk_disclosure_score": float(pd.Series(scores).mean()),
            }
        )
    return pd.DataFrame(rows)
