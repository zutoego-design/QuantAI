import pandas as pd

from qss.research.attribution import (
    bucket_attribution,
    fama_french_contribution_attribution,
    residual_alpha_stability,
    sector_attribution,
)


def test_fama_french_contribution_reconciles_excess_return():
    dates = pd.bdate_range("2025-01-01", periods=40)
    factors = pd.DataFrame(
        {
            "date": dates,
            "Mkt-RF": 0.001,
            "SMB": 0.0,
            "HML": 0.0,
            "RMW": 0.0,
            "CMA": 0.0,
            "Mom": 0.0,
            "RF": 0.0001,
        }
    )
    daily = pd.DataFrame(
        {
            "date": dates,
            "portfolio_return": 0.0001 + 0.0002 + 0.8 * factors["Mkt-RF"],
        }
    )
    exposures = pd.DataFrame(
        [
            {"factor": "alpha", "coefficient": 0.0002},
            {"factor": "Mkt-RF", "coefficient": 0.8},
        ]
    )

    attribution, summary, residual = fama_french_contribution_attribution(
        daily,
        factors,
        exposures,
    )

    assert set(attribution["component"]) >= {"alpha", "Mkt-RF", "residual"}
    assert abs(float(residual.sum())) < 1e-12
    market = summary.set_index("component").loc["Mkt-RF", "total_contribution"]
    assert market == 0.8 * 0.001 * len(dates)


def test_sector_and_bucket_attribution_use_saved_holdings():
    dates = pd.to_datetime(["2025-01-02", "2025-01-03"])
    holdings = pd.DataFrame(
        {
            "date": [dates[0], dates[0], dates[1], dates[1]],
            "symbol": ["AAA", "BBB", "AAA", "BBB"],
            "weight": [0.6, 0.4, 0.5, 0.5],
            "sector": ["Tech", "Health", "Tech", "Health"],
        }
    )
    prices = pd.DataFrame(
        {
            "date": [dates[0], dates[0], dates[1], dates[1]],
            "symbol": ["AAA", "BBB", "AAA", "BBB"],
            "return_1d": [0.01, -0.02, 0.02, 0.00],
        }
    )
    features = pd.DataFrame(
        {
            "date": [dates[0], dates[0], dates[1], dates[1]],
            "symbol": ["AAA", "BBB", "AAA", "BBB"],
            "factor_name": ["beta_to_spy"] * 4,
            "processed_value": [0.2, 1.4, 0.3, 1.5],
        }
    )

    _, sector_summary = sector_attribution(holdings, prices)
    tech = sector_summary.set_index("sector").loc["Tech", "total_contribution"]
    assert round(float(tech), 6) == 0.016

    daily, bucket_summary = bucket_attribution(
        holdings,
        prices,
        features,
        factor_names=["beta_to_spy"],
        exposure_name="beta_to_spy",
    )
    assert not daily.empty
    assert set(bucket_summary["bucket"]) <= {"low", "middle", "high", "missing"}


def test_residual_alpha_stability_reports_full_sample():
    residual = pd.Series([0.001] * 80)
    frame, summary = residual_alpha_stability(residual)
    assert "full" in set(frame["window"])
    assert round(summary["full_sample_annualized_residual_alpha"], 6) == 0.252
