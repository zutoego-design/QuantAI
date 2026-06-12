from __future__ import annotations

import pandas as pd


def build_sector_masks(symbols: list[str], sector_map: pd.Series) -> dict[str, list[int]]:
    masks: dict[str, list[int]] = {}
    mapped = sector_map.reindex(symbols).fillna("Unknown")
    for sector in sorted(mapped.unique()):
        if str(sector).strip().lower() in {"", "unknown", "unclassified"}:
            continue
        masks[sector] = [1 if mapped.loc[symbol] == sector else 0 for symbol in symbols]
    return masks


def validate_weights(weights: pd.DataFrame, max_weight: float, max_sector_weight: float) -> None:
    tolerance = 1e-6
    if weights.empty:
        return
    if abs(weights["target_weight"].sum() - 1.0) > 1e-4:
        raise ValueError("Portfolio weights do not sum to 1.")
    if (weights["target_weight"] < -tolerance).any():
        raise ValueError("Portfolio contains negative weights.")
    if (weights["target_weight"] > max_weight + tolerance).any():
        raise ValueError("Portfolio breaches max single-name weight.")
    sector_exposure = weights.groupby("sector")["target_weight"].sum()
    sector_exposure = sector_exposure.loc[
        ~sector_exposure.index.astype(str).str.lower().isin(
            ["", "unknown", "unclassified"]
        )
    ]
    if (sector_exposure > max_sector_weight + tolerance).any():
        raise ValueError("Portfolio breaches max sector weight.")
