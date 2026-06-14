import numpy as np
import pandas as pd

from qss.config.loader import get_config
from qss.factors.preprocessing import (
    neutralize_factor,
    winsorize_cross_section,
    zscore_cross_section,
)
from qss.factors.registry import _melt_factor_frame
from qss.model.scoring import compute_alpha_scores


def test_winsorize_and_zscore():
    frame = pd.DataFrame({"raw_value": [1.0, 2.0, 3.0, 100.0]})
    winsorized = winsorize_cross_section(frame, "raw_value", 0.0, 0.75)
    assert winsorized["raw_value"].max() < 100.0
    zscored = zscore_cross_section(winsorized, "raw_value")
    assert abs(zscored["raw_value"].mean()) < 1e-8


def test_zscore_preserves_missing_values_for_constant_cross_section():
    frame = pd.DataFrame({"raw_value": [1.0, 1.0, np.nan]})

    result = zscore_cross_section(frame, "raw_value")

    assert result["raw_value"].iloc[:2].eq(0.0).all()
    assert pd.isna(result["raw_value"].iloc[2])


def test_sector_neutralization_reduces_group_bias():
    frame = pd.DataFrame(
        {
            "factor": [5.0, 6.0, -5.0, -6.0],
            "sector": ["A", "A", "B", "B"],
            "market_cap": [10, 12, 11, 13],
        }
    )
    neutralized = neutralize_factor(frame, "factor", "sector", "market_cap", True, False)
    grouped = neutralized.groupby("sector")["factor"].mean().round(8)
    assert np.isclose(grouped.loc["A"], 0.0)
    assert np.isclose(grouped.loc["B"], 0.0)


def test_compute_alpha_scores_handles_missing_values():
    config = get_config(["configs/default.yaml"])
    rows = []
    for symbol in ["AAA", "BBB"]:
        for factor_group, group_config in config.factor_groups.items():
            for factor_name, factor_def in group_config.factors.items():
                rows.append(
                    {
                        "date": pd.Timestamp("2025-12-31"),
                        "symbol": symbol,
                        "factor_name": factor_name,
                        "raw_value": 0.0,
                        "processed_value": np.nan if symbol == "BBB" else 1.0,
                        "factor_group": factor_group,
                        "direction": factor_def.direction,
                        "source": factor_group,
                        "sector": "Tech",
                        "market_cap": 1_000_000_000,
                    }
                )
    scores = compute_alpha_scores(pd.DataFrame(rows), config)
    assert len(scores) == 1
    assert set(scores["symbol"]) == {"AAA"}


def test_factor_registry_outputs_only_configured_factors():
    config = get_config(["configs/default.yaml"])
    config.factor_groups["quality"].factors = {
        "roe": config.factor_groups["quality"].factors["roe"]
    }
    raw = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "roe": [0.2],
            "gross_margin": [0.5],
        }
    )
    universe = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "sector": ["Technology"],
            "market_cap": [1_000_000_000],
        }
    )

    result = _melt_factor_frame(
        raw,
        "quality",
        pd.Timestamp("2025-12-31"),
        universe,
        config,
    )

    assert set(result["factor_name"]) == {"roe"}
