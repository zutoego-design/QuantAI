from __future__ import annotations

import pandas as pd


def simple_sector_stress(weights: pd.DataFrame) -> pd.DataFrame:
    scenarios = []
    for sector, exposure in weights.groupby("sector")["target_weight"].sum().items():
        scenarios.append({"scenario": f"{sector} shock -5%", "estimated_pnl": -0.05 * exposure})
    return pd.DataFrame(scenarios)
