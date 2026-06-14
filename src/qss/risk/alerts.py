from __future__ import annotations

import pandas as pd

from qss.config.schema import RiskLimitsConfig

ALERT_TOLERANCE = 1e-6


def generate_alerts(metrics: dict[str, float], sector_exposure: pd.DataFrame, config: RiskLimitsConfig) -> pd.DataFrame:
    alerts: list[dict[str, str | float]] = []
    portfolio_cfg = config.portfolio
    checks = [
        ("daily_loss", metrics.get("daily_loss", 0.0) < -portfolio_cfg.max_daily_loss - ALERT_TOLERANCE, metrics.get("daily_loss", 0.0)),
        ("drawdown", metrics.get("drawdown", 0.0) < -portfolio_cfg.max_drawdown_alert - ALERT_TOLERANCE, metrics.get("drawdown", 0.0)),
        ("realized_vol", metrics.get("realized_vol", 0.0) > portfolio_cfg.max_realized_vol_annualized + ALERT_TOLERANCE, metrics.get("realized_vol", 0.0)),
        (
            "beta_high",
            metrics.get("beta", 0.0) > portfolio_cfg.max_beta_to_benchmark + ALERT_TOLERANCE,
            metrics.get("beta", 0.0),
        ),
        (
            "beta_low",
            metrics.get("beta", 0.0) < portfolio_cfg.min_beta_to_benchmark - ALERT_TOLERANCE,
            metrics.get("beta", 0.0),
        ),
        (
            "single_name",
            metrics.get("single_name_weight", 0.0) > portfolio_cfg.max_single_name_weight + ALERT_TOLERANCE,
            metrics.get("single_name_weight", 0.0),
        ),
        (
            "tracking_error",
            metrics.get("tracking_error", 0.0) > portfolio_cfg.max_tracking_error + ALERT_TOLERANCE,
            metrics.get("tracking_error", 0.0),
        ),
    ]
    for rule, breached, value in checks:
        if breached:
            alerts.append({"rule": rule, "value": value})
    for row in sector_exposure.itertuples(index=False):
        if row.sector_weight > portfolio_cfg.max_sector_weight + ALERT_TOLERANCE:
            alerts.append({"rule": f"sector_{row.sector}", "value": row.sector_weight})
    return pd.DataFrame(alerts)
