from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JobDefinition:
    name: str
    cadence: str
    owner: str
    command: str
    approval_required: bool


JOB_DEFINITIONS = {
    "daily_risk_refresh": JobDefinition(
        name="daily_risk_refresh",
        cadence="weekdays_after_market_close",
        owner="research-operations",
        command="python -m qss.cli risk-monitor --date <YYYY-MM-DD>",
        approval_required=False,
    ),
    "monthly_rebalance_packet": JobDefinition(
        name="monthly_rebalance_packet",
        cadence="monthly_after_last_trading_day",
        owner="portfolio-research",
        command="python -m qss.cli run-monthly-pipeline --date <YYYY-MM-DD>",
        approval_required=True,
    ),
    "experiment_registry_refresh": JobDefinition(
        name="experiment_registry_refresh",
        cadence="daily",
        owner="research-operations",
        command="python -m qss.cli registry-refresh",
        approval_required=False,
    ),
    "forward_validation_refresh": JobDefinition(
        name="forward_validation_refresh",
        cadence="weekdays_after_market_close",
        owner="research-operations",
        command=(
            "python -m qss.cli forward-record "
            "--forward-root <FORWARD_ROOT> --date <YYYY-MM-DD>"
        ),
        approval_required=False,
    ),
}
