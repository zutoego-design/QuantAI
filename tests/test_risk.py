import pandas as pd

from qss.config.loader import get_config
from qss.data.quality import check_data_quality
from qss.risk.alerts import generate_alerts


def test_risk_alerts_trigger_and_data_quality_flags():
    config = get_config(["configs/default.yaml"])
    metrics = {
        "daily_loss": -0.05,
        "drawdown": -0.20,
        "realized_vol": 0.30,
        "beta": 1.5,
        "single_name_weight": 0.08,
        "tracking_error": 0.12,
    }
    sector_df = pd.DataFrame({"sector": ["Tech"], "sector_weight": [0.40]})
    alerts = generate_alerts(metrics, sector_df, config.risk_limits)
    assert not alerts.empty
    frame = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "period_end_date": [pd.Timestamp("2025-03-31")],
            "filing_date": [pd.Timestamp("2025-03-31")],
            "available_date": [pd.NaT],
        }
    )
    quality = check_data_quality("fundamentals_quarterly", frame, ["symbol", "period_end_date", "filing_date"])
    assert "missing_available_date" in set(quality["rule"])


def test_risk_alerts_ignore_solver_rounding_at_limit():
    config = get_config(["configs/default.yaml"])
    metrics = {
        "daily_loss": 0.0,
        "drawdown": 0.0,
        "realized_vol": 0.0,
        "beta": 1.0,
        "single_name_weight": config.risk_limits.portfolio.max_single_name_weight
        + 1e-10,
        "tracking_error": 0.0,
    }
    sector_df = pd.DataFrame(
        {
            "sector": ["Tech"],
            "sector_weight": [
                config.risk_limits.portfolio.max_sector_weight + 1e-10
            ],
        }
    )

    alerts = generate_alerts(metrics, sector_df, config.risk_limits)

    assert alerts.empty
