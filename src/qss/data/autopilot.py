from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from qss.backtest.engine import run_backtest
from qss.config.schema import AppConfig
from qss.data.status import membership_symbols, research_data_status
from qss.data.storage import resolve_path
from qss.data.validation import failed_check_summary, validate_research_data
from qss.ingestion.fred import ingest_macro
from qss.ingestion.prices_yfinance import ingest_prices
from qss.ingestion.sec_edgar import ingest_fundamentals
from qss.logging_utils import logger
from qss.universe.sector_enrichment import enrich_sector_metadata
from qss.universe.sync import sync_universe

ACTIVE_STATUSES = {"starting", "running", "waiting", "stopping"}
FINAL_STATUSES = {"completed", "blocked", "failed", "stopped"}


@dataclass
class AutopilotState:
    status: str = "idle"
    stage: str = "idle"
    message: str = "Autopilot has not been started."
    progress: str = ""
    pid: int | None = None
    started_at: str | None = None
    updated_at: str | None = None
    next_resume_at: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    run_backtest: bool = True
    stop_requested: bool = False
    validation_run: str | None = None
    backtest_run: str | None = None


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _timestamp(value: pd.Timestamp | None = None) -> str:
    return str((value or _utc_now()).isoformat())


def autopilot_root(config: AppConfig) -> Path:
    return resolve_path(config.paths.reports) / "data_autopilot"


def autopilot_status_path(config: AppConfig) -> Path:
    return autopilot_root(config) / "status.json"


def autopilot_log_path(config: AppConfig) -> Path:
    return autopilot_root(config) / "autopilot.log"


def _write_state(config: AppConfig, state: AutopilotState) -> AutopilotState:
    root = autopilot_root(config)
    root.mkdir(parents=True, exist_ok=True)
    state.updated_at = _timestamp()
    target = autopilot_status_path(config)
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(asdict(state), indent=2),
        encoding="utf-8",
    )
    temporary.replace(target)
    return state


def read_autopilot_state(config: AppConfig) -> AutopilotState:
    path = autopilot_status_path(config)
    if not path.exists():
        return AutopilotState()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        known = {
            field: payload[field]
            for field in AutopilotState.__dataclass_fields__
            if field in payload
        }
        return AutopilotState(**known)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return AutopilotState(
            status="failed",
            stage="state",
            message="Autopilot state file is unreadable.",
        )


def _process_is_running(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def autopilot_is_active(config: AppConfig) -> bool:
    state = read_autopilot_state(config)
    return state.status in ACTIVE_STATUSES and _process_is_running(state.pid)


def _component(status_frame: pd.DataFrame, name: str) -> str:
    rows = status_frame.loc[status_frame["component"] == name, "status"]
    return str(rows.iloc[0]) if not rows.empty else "missing"


def _status_progress(config: AppConfig, start_date: str, end_date: str) -> str:
    checks = research_data_status(config, start_date, end_date).checks
    ready = int((checks["status"] == "ready").sum())
    return f"{ready}/{len(checks)} foundations ready"


def _next_retry_time(message: str | None) -> pd.Timestamp:
    now = _utc_now()
    quota_markers = ("budget reached", "rate limit", "throttl", "25 requests")
    if message and any(marker in message.lower() for marker in quota_markers):
        return (now + pd.Timedelta(days=1)).normalize() + pd.Timedelta(minutes=10)
    return now + pd.Timedelta(minutes=15)


def _stop_requested(config: AppConfig) -> bool:
    return read_autopilot_state(config).stop_requested


def _wait_until(
    config: AppConfig,
    state: AutopilotState,
    resume_at: pd.Timestamp,
    sleep_fn: Callable[[float], None],
) -> bool:
    while _utc_now() < resume_at:
        if _stop_requested(config):
            state.status = "stopped"
            state.stage = "stopped"
            state.message = "Autopilot stopped after preserving completed work."
            state.next_resume_at = None
            _write_state(config, state)
            return False
        remaining = max((resume_at - _utc_now()).total_seconds(), 1.0)
        sleep_fn(min(remaining, 60.0))
    return True


def request_autopilot_stop(config: AppConfig) -> AutopilotState:
    state = read_autopilot_state(config)
    if state.status not in ACTIVE_STATUSES:
        return state
    state.stop_requested = True
    state.status = "stopping"
    state.message = "Stop requested; the current provider operation will finish first."
    return _write_state(config, state)


def _run_retryable_stage(
    config: AppConfig,
    state: AutopilotState,
    *,
    stage: str,
    message: str,
    operation: Callable[[], object],
    wait_for_quota: bool,
    sleep_fn: Callable[[float], None],
    max_attempts: int = 3,
) -> bool:
    for attempt in range(1, max_attempts + 1):
        if _stop_requested(config):
            state.status = "stopped"
            state.stage = "stopped"
            state.message = "Autopilot stopped after preserving completed work."
            _write_state(config, state)
            return False
        state.status = "running"
        state.stage = stage
        state.message = message
        state.next_resume_at = None
        _write_state(config, state)
        try:
            operation()
            return True
        except RuntimeError as exc:
            if attempt >= max_attempts:
                raise
            retry_at = _next_retry_time(str(exc))
            state.status = "waiting"
            state.message = (
                f"{stage.title()} attempt {attempt}/{max_attempts} was interrupted; "
                "cached work is safe and retry is automatic."
            )
            state.next_resume_at = _timestamp(retry_at)
            _write_state(config, state)
            if not wait_for_quota:
                return False
            if not _wait_until(config, state, retry_at, sleep_fn):
                return False
    return False


def run_research_autopilot(
    config: AppConfig,
    start_date: str,
    end_date: str,
    *,
    run_backtest_when_ready: bool = True,
    wait_for_quota: bool = True,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> AutopilotState:
    root = autopilot_root(config)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "worker.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        current = read_autopilot_state(config)
        if autopilot_is_active(config):
            return current
        lock_path.unlink(missing_ok=True)
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)

    os.write(lock_fd, str(os.getpid()).encode("ascii"))
    os.close(lock_fd)
    previous = read_autopilot_state(config)
    state = AutopilotState(
        status="running",
        stage="starting",
        message="Inspecting cached research data.",
        pid=os.getpid(),
        started_at=previous.started_at or _timestamp(),
        start_date=start_date,
        end_date=end_date,
        run_backtest=run_backtest_when_ready,
    )
    _write_state(config, state)

    try:
        current = research_data_status(config, start_date, end_date)
        if _component(current.checks, "Provider credentials") != "ready":
            missing = current.checks.loc[
                current.checks["component"] == "Provider credentials",
                "detail",
            ].iloc[0]
            state.status = "blocked"
            state.stage = "credentials"
            state.message = f"Configure provider credentials first: {missing}"
            return _write_state(config, state)
        if _component(current.checks, "Macro observations") != "ready":
            state.progress = _status_progress(config, start_date, end_date)
            completed = _run_retryable_stage(
                config,
                state,
                stage="macro",
                message="Refreshing FRED macro observations.",
                operation=lambda: ingest_macro(config),
                wait_for_quota=wait_for_quota,
                sleep_fn=sleep_fn,
            )
            if not completed:
                return read_autopilot_state(config)

        while True:
            if _stop_requested(config):
                state.status = "stopped"
                state.stage = "stopped"
                state.message = "Autopilot stopped after preserving completed work."
                return _write_state(config, state)

            current = research_data_status(config, start_date, end_date)
            current_membership_mode = (
                config.universe.membership_mode == "current_snapshot"
            )
            single_source_history_mode = (
                config.universe.membership_mode == "point_in_time"
                and config.universe.validation_provider == "disabled"
            )
            universe_component = (
                "Current universe baseline"
                if current_membership_mode
                else "Universe history"
            )
            validation_component = (
                "Survivorship-bias disclosure"
                if current_membership_mode
                else "Universe source audit"
                if single_source_history_mode
                else "Cross-source validation"
            )
            universe_ready = (
                _component(current.checks, universe_component) == "ready"
                and _component(current.checks, validation_component) == "ready"
            )
            if universe_ready:
                break

            state.status = "running"
            state.stage = "universe"
            state.message = (
                "Preparing the current Nasdaq membership baseline."
                if current_membership_mode
                else "Reconstructing S&P 500 point-in-time membership."
                if config.universe.long_history_provider == "sp500_wikipedia"
                else "Extending point-in-time universe history and validation."
            )
            state.progress = _status_progress(config, start_date, end_date)
            state.next_resume_at = None
            _write_state(config, state)
            try:
                result = sync_universe(
                    config,
                    start_date=start_date,
                    end_date=end_date,
                    validate_recent=not current_membership_mode,
                )
            except RuntimeError as exc:
                retry_at = _next_retry_time(str(exc))
                state.status = "waiting"
                state.stage = "universe"
                state.message = (
                    f"Provider temporarily unavailable ({type(exc).__name__}); "
                    "cached progress is safe."
                )
                state.next_resume_at = _timestamp(retry_at)
                _write_state(config, state)
                if not wait_for_quota:
                    return state
                if not _wait_until(config, state, retry_at, sleep_fn):
                    return read_autopilot_state(config)
                continue
            current = research_data_status(config, start_date, end_date)
            universe_ready = (
                _component(current.checks, universe_component) == "ready"
                and _component(current.checks, validation_component) == "ready"
            )
            if universe_ready:
                break
            if current_membership_mode:
                state.status = "blocked"
                state.stage = "universe"
                state.message = (
                    "Current Nasdaq membership was fetched, but the local baseline "
                    "did not pass completeness checks."
                )
                return _write_state(config, state)

            retry_at = _next_retry_time(result.warning)
            state.status = "waiting"
            state.stage = "universe"
            state.message = (
                result.warning
                or "Universe history remains incomplete; waiting before the next retry."
            )
            state.progress = _status_progress(config, start_date, end_date)
            state.next_resume_at = _timestamp(retry_at)
            _write_state(config, state)
            if not wait_for_quota:
                return state
            if not _wait_until(config, state, retry_at, sleep_fn):
                return read_autopilot_state(config)

        current = research_data_status(config, start_date, end_date)
        if _component(current.checks, "Research prices") != "ready":
            state.progress = _status_progress(config, start_date, end_date)
            tickers = membership_symbols(
                config,
                start_date=start_date,
                end_date=end_date,
            )
            price_start_date = (
                pd.Timestamp(start_date) - pd.DateOffset(years=2)
            ).strftime("%Y-%m-%d")
            completed = _run_retryable_stage(
                config,
                state,
                stage="prices",
                message="Downloading research-window price history with factor warmup.",
                operation=lambda: ingest_prices(
                    config,
                    start_date=price_start_date,
                    end_date=end_date,
                    tickers=tickers,
                ),
                wait_for_quota=wait_for_quota,
                sleep_fn=sleep_fn,
            )
            if not completed:
                return read_autopilot_state(config)

        current = research_data_status(config, start_date, end_date)
        fundamentals_ready = (
            _component(current.checks, "SEC fundamentals") == "ready"
            and _component(current.checks, "Sector metadata") == "ready"
        )
        if not fundamentals_ready:
            state.progress = _status_progress(config, start_date, end_date)
            tickers = membership_symbols(
                config,
                start_date=start_date,
                end_date=end_date,
            )
            completed = _run_retryable_stage(
                config,
                state,
                stage="fundamentals",
                message="Refreshing SEC facts and company classification.",
                operation=lambda: (
                    ingest_fundamentals(config, tickers=tickers),
                    enrich_sector_metadata(
                        config,
                        start_date=start_date,
                        end_date=end_date,
                        tickers=tickers,
                    ),
                ),
                wait_for_quota=wait_for_quota,
                sleep_fn=sleep_fn,
            )
            if not completed:
                return read_autopilot_state(config)

        current = research_data_status(config, start_date, end_date)
        if not current.ready:
            incomplete = current.checks.loc[
                current.checks["status"] != "ready",
                "component",
            ].tolist()
            state.status = "blocked"
            state.stage = "data-quality"
            state.message = (
                "Provider passes completed, but coverage thresholds remain unmet: "
                + ", ".join(incomplete)
            )
            state.progress = _status_progress(config, start_date, end_date)
            return _write_state(config, state)

        state.stage = "validation"
        state.message = "Running the strict research-data gate."
        state.progress = _status_progress(config, start_date, end_date)
        _write_state(config, state)
        validation = validate_research_data(config, start_date, end_date)
        state.validation_run = str(validation.run_path)
        if validation.status != "valid":
            state.status = "blocked"
            state.message = (
                "Strict validation still has failed checks: "
                + failed_check_summary(validation.checks)
            )
            return _write_state(config, state)

        if run_backtest_when_ready:
            state.stage = "backtest"
            state.message = "Trusted data is ready; running the backtest."
            _write_state(config, state)
            backtest = run_backtest(start_date, end_date, config)
            state.backtest_run = str(backtest.run_path)

        state.status = "completed"
        state.stage = "completed"
        state.message = (
            "Research data preparation and backtest are complete."
            if run_backtest_when_ready
            else "Research data preparation is complete."
        )
        state.progress = _status_progress(config, start_date, end_date)
        state.next_resume_at = None
        return _write_state(config, state)
    except Exception as exc:
        logger.exception("Research data autopilot failed")
        state.status = "failed"
        state.stage = state.stage or "unknown"
        state.message = f"{type(exc).__name__}: {exc}"
        state.next_resume_at = None
        return _write_state(config, state)
    finally:
        lock_path.unlink(missing_ok=True)


def start_autopilot_process(
    config: AppConfig,
    config_paths: list[str],
    start_date: str,
    end_date: str,
    *,
    run_backtest_when_ready: bool = True,
) -> AutopilotState:
    current = read_autopilot_state(config)
    if current.status in ACTIVE_STATUSES and _process_is_running(current.pid):
        return current

    root = autopilot_root(config)
    root.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-u",
        "-m",
        "qss.cli",
        "autopilot-run",
    ]
    for path in config_paths:
        command.extend(["--config", path])
    command.extend(["--start", start_date, "--end", end_date])
    command.append("--run-backtest" if run_backtest_when_ready else "--skip-backtest")

    state = AutopilotState(
        status="starting",
        stage="starting",
        message="Launching the persistent research-data worker.",
        started_at=_timestamp(),
        start_date=start_date,
        end_date=end_date,
        run_backtest=run_backtest_when_ready,
    )
    _write_state(config, state)

    log_handle = autopilot_log_path(config).open("a", encoding="utf-8")
    creationflags = 0
    if os.name == "nt":
        creationflags = (
            subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NO_WINDOW
        )
    try:
        try:
            process = subprocess.Popen(
                command,
                cwd=resolve_path("."),
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                close_fds=True,
                creationflags=creationflags,
            )
        except OSError as exc:
            state.status = "failed"
            state.stage = "starting"
            state.message = f"Could not launch background worker: {exc}"
            return _write_state(config, state)
    finally:
        log_handle.close()
    current = read_autopilot_state(config)
    if current.status != "starting":
        return current
    current.pid = process.pid
    current.message = "Research-data worker started in the background."
    return _write_state(config, current)
