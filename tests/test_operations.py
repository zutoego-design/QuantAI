from types import SimpleNamespace

import pandas as pd

from qss.config.loader import get_config
from qss.workflows.operations import run_operations_dry_run


def test_operations_dry_run_writes_daily_logs_and_preserves_attempts(
    tmp_path,
    monkeypatch,
):
    config = get_config(["configs/default.yaml"])
    config.paths.silver_data = str(tmp_path / "silver")
    config.paths.reports = str(tmp_path / "reports")
    prices_path = tmp_path / "silver" / "prices"
    prices_path.mkdir(parents=True)
    dates = pd.bdate_range("2025-01-02", periods=3)
    pd.DataFrame(
        {
            "date": dates,
            "symbol": ["SPY"] * len(dates),
            "return_1d": [0.0, 0.01, -0.01],
        }
    ).to_parquet(prices_path / "prices_daily.parquet", index=False)
    counter = {"value": 0}

    def fake_risk(date, _config):
        counter["value"] += 1
        run_path = tmp_path / "reports" / "runs" / f"risk-{counter['value']}"
        run_path.mkdir(parents=True)
        return SimpleNamespace(
            run_id=f"risk-{counter['value']}",
            run_path=run_path,
            alerts=pd.DataFrame(),
        )

    monkeypatch.setattr(
        "qss.workflows.operations.run_daily_risk_monitor",
        fake_risk,
    )
    monkeypatch.setattr(
        "qss.workflows.operations.register_run_path",
        lambda *args, **kwargs: True,
    )

    first, markdown = run_operations_dry_run(
        config,
        end_date=dates[-1],
        trading_days=3,
    )
    second, _ = run_operations_dry_run(
        config,
        end_date=dates[-1],
        trading_days=3,
    )

    assert first["status"].eq("valid").all()
    assert second["status"].eq("valid").all()
    assert markdown.exists()
    log = (
        tmp_path
        / "reports"
        / "operations"
        / "daily_log"
        / f"{dates[0].date()}.md"
    ).read_text(encoding="utf-8")
    assert log.count("## Attempt") == 2
