import json

import pandas as pd
from test_e2e_research_run import _write_fixture

from qss.research.forward_validation import (
    initialize_forward_validation,
    record_forward_day,
)


def _fake_replay(tmp_path, config):
    replay = tmp_path / "reports" / "runs" / "historical-replay-fixture"
    for strategy_id in ["v1_control", "v2_core"]:
        root = replay / "candidates" / strategy_id
        root.mkdir(parents=True, exist_ok=True)
        candidate = config.model_copy(deep=True)
        candidate.strategy.name = strategy_id
        (root / "resolved_config.json").write_text(
            candidate.model_dump_json(indent=2),
            encoding="utf-8",
        )
    (replay / "selection.json").write_text(
        json.dumps(
            {
                "decision": "selected_v2",
                "control_strategy_id": "v1_control",
                "selected_strategy_id": "v2_core",
                "challenger_strategy_id": None,
                "forward_strategy_id": "v2_core",
            }
        ),
        encoding="utf-8",
    )
    return replay


def test_forward_ledger_freezes_configs_and_daily_record_is_idempotent(tmp_path):
    config = _write_fixture(tmp_path)
    replay = _fake_replay(tmp_path, config)
    initialized = initialize_forward_validation(
        config,
        replay,
        start_date="2025-01-02",
        end_date="2025-03-31",
        study_id="forward-fixture",
    )
    assert initialized.status == "initialized"
    ledger = json.loads(
        (initialized.root / "ledger.json").read_text(encoding="utf-8")
    )
    assert ledger["immutable"] is True
    assert len(ledger["strategies"]) == 2

    first = record_forward_day(initialized.root, "2025-01-03")
    second = record_forward_day(initialized.root, "2025-01-03")
    assert first.status == "monitoring"
    assert second.status == "monitoring"
    records = pd.read_csv(initialized.root / "daily_records.csv")
    assert len(records) == 3
    assert set(records["strategy_id"]) == {"v1_control", "v2_core", "SPY"}


def test_forward_initialization_rejects_changed_frozen_identity(tmp_path):
    config = _write_fixture(tmp_path)
    replay = _fake_replay(tmp_path, config)
    initialized = initialize_forward_validation(
        config,
        replay,
        start_date="2025-01-02",
        end_date="2025-03-31",
        study_id="forward-fixture",
    )
    frozen = initialized.root / "frozen_configs" / "v2_core.json"
    payload = json.loads(frozen.read_text(encoding="utf-8"))
    payload["strategy"]["name"] = "mutated"
    frozen.write_text(json.dumps(payload), encoding="utf-8")

    try:
        initialize_forward_validation(
            config,
            replay,
            start_date="2025-01-02",
            end_date="2025-03-31",
            study_id="forward-fixture",
        )
    except ValueError as exc:
        assert "different content" in str(exc)
    else:
        raise AssertionError("Changed frozen config should be rejected.")
