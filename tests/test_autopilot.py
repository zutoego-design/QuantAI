from types import SimpleNamespace

import pandas as pd

from qss.config.loader import get_config
from qss.data.autopilot import read_autopilot_state, run_research_autopilot
from qss.data.status import ResearchDataStatus

COMPONENTS = [
    "Universe history",
    "Research prices",
    "SEC fundamentals",
    "Macro observations",
    "Sector metadata",
    "Cross-source validation",
    "Provider credentials",
]


def _config(tmp_path):
    config = get_config(["configs/default.yaml"])
    config.universe.membership_mode = "point_in_time"
    config.universe.long_history_provider = "alpha_vantage"
    config.universe.validation_provider = "massive"
    config.paths.raw_data = str(tmp_path / "raw")
    config.paths.silver_data = str(tmp_path / "silver")
    config.paths.gold_data = str(tmp_path / "gold")
    config.paths.reports = str(tmp_path / "reports")
    config.universe.remote_request_interval_seconds = 0
    return config


def _status(**overrides):
    states = {component: "ready" for component in COMPONENTS}
    states.update(overrides)
    checks = pd.DataFrame(
        [
            {
                "component": component,
                "status": states[component],
                "progress": "",
                "detail": "",
            }
            for component in COMPONENTS
        ]
    )
    return ResearchDataStatus(
        checks=checks,
        ready=bool((checks["status"] == "ready").all()),
    )


def test_autopilot_pauses_without_losing_progress_at_provider_quota(
    tmp_path,
    monkeypatch,
):
    config = _config(tmp_path)
    partial = _status(
        **{
            "Universe history": "partial",
            "Research prices": "missing",
            "SEC fundamentals": "missing",
            "Sector metadata": "missing",
            "Cross-source validation": "missing",
        }
    )
    monkeypatch.setattr(
        "qss.data.autopilot.research_data_status",
        lambda *args, **kwargs: partial,
    )
    sync_calls = []

    def _sync(*args, **kwargs):
        sync_calls.append(kwargs)
        return SimpleNamespace(
            warning="Remote request budget reached; rerun to continue.",
        )

    monkeypatch.setattr("qss.data.autopilot.sync_universe", _sync)
    monkeypatch.setattr(
        "qss.data.autopilot.ingest_prices",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("prices must wait for universe history")
        ),
    )

    state = run_research_autopilot(
        config,
        "2015-01-01",
        "2026-06-12",
        wait_for_quota=False,
    )

    assert len(sync_calls) == 1
    assert sync_calls[0]["start_date"] == "2015-01-01"
    assert sync_calls[0]["end_date"] == "2026-06-12"
    assert state.status == "waiting"
    assert state.stage == "universe"
    assert state.next_resume_at
    assert read_autopilot_state(config).status == "waiting"


def test_autopilot_waits_after_transient_universe_provider_error(
    tmp_path,
    monkeypatch,
):
    config = _config(tmp_path)
    partial = _status(
        **{
            "Universe history": "partial",
            "Cross-source validation": "missing",
        }
    )
    monkeypatch.setattr(
        "qss.data.autopilot.research_data_status",
        lambda *args, **kwargs: partial,
    )
    monkeypatch.setattr(
        "qss.data.autopilot.sync_universe",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("temporary network failure")
        ),
    )

    state = run_research_autopilot(
        config,
        "2015-01-01",
        "2026-06-12",
        wait_for_quota=False,
    )

    assert state.status == "waiting"
    assert "cached progress is safe" in state.message


def test_autopilot_runs_remaining_stages_and_backtest(tmp_path, monkeypatch):
    config = _config(tmp_path)
    progress = {"prices": False, "fundamentals": False}

    def _current_status(*args, **kwargs):
        return _status(
            **{
                "Research prices": "ready" if progress["prices"] else "partial",
                "SEC fundamentals": (
                    "ready" if progress["fundamentals"] else "partial"
                ),
                "Sector metadata": (
                    "ready" if progress["fundamentals"] else "partial"
                ),
            }
        )

    monkeypatch.setattr(
        "qss.data.autopilot.research_data_status",
        _current_status,
    )
    monkeypatch.setattr(
        "qss.data.autopilot.membership_symbols",
        lambda *args, **kwargs: ["AAA"],
    )
    monkeypatch.setattr(
        "qss.data.autopilot.ingest_prices",
        lambda *args, **kwargs: progress.update(prices=True),
    )
    monkeypatch.setattr(
        "qss.data.autopilot.ingest_fundamentals",
        lambda *args, **kwargs: progress.update(fundamentals=True),
    )
    monkeypatch.setattr(
        "qss.data.autopilot.validate_research_data",
        lambda *args, **kwargs: SimpleNamespace(
            status="valid",
            run_path=tmp_path / "validation",
        ),
    )
    backtests = []

    def _backtest(*args, **kwargs):
        backtests.append(args)
        return SimpleNamespace(run_path=tmp_path / "backtest")

    monkeypatch.setattr("qss.data.autopilot.run_backtest", _backtest)

    state = run_research_autopilot(
        config,
        "2015-01-01",
        "2026-06-12",
    )

    assert progress == {"prices": True, "fundamentals": True}
    assert len(backtests) == 1
    assert state.status == "completed"
    assert state.backtest_run == str(tmp_path / "backtest")
