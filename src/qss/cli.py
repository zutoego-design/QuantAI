from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from qss.acceptance import run_acceptance_checks
from qss.approval.workflow import transition_approval
from qss.backtest.engine import run_backtest
from qss.config.loader import get_config
from qss.data.autopilot import (
    read_autopilot_state,
    request_autopilot_stop,
    run_research_autopilot,
    start_autopilot_process,
)
from qss.data.status import membership_symbols, research_data_status
from qss.data.storage import ensure_data_directories, read_parquet, resolve_path
from qss.data.validation import failed_check_summary, validate_research_data
from qss.experiments.registry import ExperimentRegistry, refresh_registry
from qss.factors.registry import compute_and_store_factor_values
from qss.ingestion.fama_french import load_fama_french_daily
from qss.ingestion.fred import ingest_macro
from qss.ingestion.prices_yfinance import ingest_prices
from qss.ingestion.sec_edgar import ingest_fundamentals
from qss.ingestion.sec_text import cache_sec_filing_text
from qss.logging_utils import configure_logging
from qss.macro.regime import compute_macro_regime
from qss.model.scoring import compute_and_store_scores
from qss.portfolio.rebalance import run_rebalance
from qss.progress import emit_progress
from qss.quickstart import run_quickstart
from qss.reporting.comprehensive_report import generate_comprehensive_report
from qss.reporting.service import render_saved_backtest
from qss.research.comparison import generate_baseline_comparison
from qss.research.forward_validation import (
    evaluate_forward_validation,
    initialize_forward_validation,
    record_forward_day,
)
from qss.research.historical_replay import run_historical_replay
from qss.research.orchestrator import ResearchOrchestrator, load_experiment_spec
from qss.risk.monitor import run_daily_risk_monitor
from qss.runs.baseline import snapshot_legacy_baseline
from qss.universe.builder import build_and_store_universe
from qss.universe.sector_enrichment import enrich_sector_metadata
from qss.universe.sync import sync_universe
from qss.workflows.jobs import JOB_DEFINITIONS
from qss.workflows.operations import run_operations_dry_run

app = typer.Typer(help="Quant Stock Selection System CLI")


def _load_app_config(config_paths: list[str]):
    config = get_config(config_paths)
    ensure_data_directories(config)
    configure_logging(config.runtime.log_level)
    return config


def _research_tickers(
    config,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[str]:
    symbols = membership_symbols(
        config,
        start_date=start_date or config.backtest.start_date,
        end_date=end_date,
    )
    if symbols:
        return symbols
    seed = pd.read_csv(Path(config.universe.seed_metadata_path))
    return seed["symbol"].tolist()


def _require_valid_research_data(config, start: str | None = None, end: str | None = None):
    if not config.runtime.research_mode:
        return
    result = validate_research_data(config, start, end)
    if result.status != "valid":
        raise RuntimeError(
            "Research data gate failed: "
            f"{failed_check_summary(result.checks)}. "
            f"Review validation run {result.run_path}."
        )


@app.command("snapshot-legacy-baseline")
def snapshot_legacy_baseline_cmd(
    config: list[str] = typer.Option(["configs/default.yaml"], "--config"),
    label: str = typer.Option("legacy-demo-20260612", "--label"),
):
    cfg = _load_app_config(config)
    typer.echo(str(snapshot_legacy_baseline(cfg, label)))


@app.command("sync-universe")
def sync_universe_cmd(
    config: list[str] = typer.Option(["configs/default.yaml"], "--config"),
    start: Optional[str] = typer.Option(None, "--start"),
    end: Optional[str] = typer.Option(None, "--end"),
    validate_recent: bool = typer.Option(True, "--validate-recent/--skip-validation"),
):
    cfg = _load_app_config(config)
    try:
        result = sync_universe(
            cfg, start_date=start, end_date=end, validate_recent=validate_recent
        )
    except (ValueError, RuntimeError) as exc:
        typer.echo(f"invalid: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(
        json.dumps(
            {
                "memberships": len(result.membership),
                "securities": len(result.security_master),
                "validation_rows": len(result.validation),
                "historical_months": result.historical_months,
                "requested_months": result.requested_months,
                "next_missing_date": result.next_missing_date,
                "warning": result.warning,
            }
        )
    )


@app.command("validate-data")
def validate_data_cmd(
    config: list[str] = typer.Option(["configs/default.yaml"], "--config"),
    start: Optional[str] = typer.Option(None, "--start"),
    end: Optional[str] = typer.Option(None, "--end"),
):
    cfg = _load_app_config(config)
    result = validate_research_data(cfg, start, end)
    typer.echo(f"{result.status}: {result.run_path}")
    if result.status != "valid":
        raise typer.Exit(code=2)


@app.command("data-status")
def data_status_cmd(
    config: list[str] = typer.Option(["configs/default.yaml"], "--config"),
    start: Optional[str] = typer.Option(None, "--start"),
    end: Optional[str] = typer.Option(None, "--end"),
):
    cfg = _load_app_config(config)
    status = research_data_status(cfg, start, end)
    typer.echo(status.checks.to_string(index=False))
    typer.echo(f"\noverall: {'ready' if status.ready else 'not ready'}")


@app.command("autopilot-start")
def autopilot_start_cmd(
    config: list[str] = typer.Option(["configs/default.yaml"], "--config"),
    start: Optional[str] = typer.Option(None, "--start"),
    end: Optional[str] = typer.Option(None, "--end"),
    run_backtest: bool = typer.Option(
        True,
        "--run-backtest/--skip-backtest",
    ),
):
    cfg = _load_app_config(config)
    start_date = start or cfg.backtest.start_date
    end_date = end or str(pd.Timestamp.today().date())
    state = start_autopilot_process(
        cfg,
        config,
        start_date,
        end_date,
        run_backtest_when_ready=run_backtest,
    )
    typer.echo(json.dumps(state.__dict__))


@app.command("autopilot-run", hidden=True)
def autopilot_run_cmd(
    config: list[str] = typer.Option(["configs/default.yaml"], "--config"),
    start: Optional[str] = typer.Option(None, "--start"),
    end: Optional[str] = typer.Option(None, "--end"),
    run_backtest: bool = typer.Option(
        True,
        "--run-backtest/--skip-backtest",
    ),
):
    cfg = _load_app_config(config)
    state = run_research_autopilot(
        cfg,
        start or cfg.backtest.start_date,
        end or str(pd.Timestamp.today().date()),
        run_backtest_when_ready=run_backtest,
        wait_for_quota=True,
    )
    typer.echo(json.dumps(state.__dict__))


@app.command("autopilot-status")
def autopilot_status_cmd(
    config: list[str] = typer.Option(["configs/default.yaml"], "--config"),
):
    cfg = _load_app_config(config)
    typer.echo(json.dumps(read_autopilot_state(cfg).__dict__, indent=2))


@app.command("autopilot-stop")
def autopilot_stop_cmd(
    config: list[str] = typer.Option(["configs/default.yaml"], "--config"),
):
    cfg = _load_app_config(config)
    typer.echo(json.dumps(request_autopilot_stop(cfg).__dict__, indent=2))


@app.command("run-experiment")
def run_experiment_cmd(
    spec: str = typer.Option(..., "--spec"),
    config: list[str] = typer.Option(["configs/default.yaml"], "--config"),
):
    cfg = _load_app_config(config)
    try:
        context = ResearchOrchestrator(cfg).run(load_experiment_spec(spec))
    except (ValueError, RuntimeError) as exc:
        typer.echo(f"invalid: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(f"{context.manifest.status}: {context.root}")
    if context.manifest.status != "valid":
        raise typer.Exit(code=2)


@app.command("render-report")
def render_report_cmd(
    run_path: str = typer.Option(..., "--run-path"),
):
    bundle = render_saved_backtest(run_path)
    typer.echo(str(bundle.html_report))


@app.command("comprehensive-report")
def comprehensive_report_cmd(
    reports_root: str = typer.Option("reports", "--reports-root"),
    run_path: Optional[str] = typer.Option(None, "--run-path"),
):
    try:
        bundle = generate_comprehensive_report(reports_root, run_path)
    except ValueError as exc:
        typer.echo(f"invalid: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(
        json.dumps(
            {
                "source_run_id": bundle.source_run_id,
                "html_report": str(bundle.html_report),
                "structured_report": str(bundle.structured_report),
            },
            indent=2,
        )
    )


@app.command("acceptance-check")
def acceptance_check_cmd(
    config: list[str] = typer.Option(["configs/default.yaml"], "--config"),
    run_path: Optional[str] = typer.Option(None, "--run-path"),
):
    cfg = _load_app_config(config)
    checks, context = run_acceptance_checks(cfg, run_path)
    typer.echo(checks.to_string(index=False))
    if context.manifest.status != "valid":
        raise typer.Exit(code=2)


@app.command("registry-query")
def registry_query_cmd(
    config: list[str] = typer.Option(["configs/default.yaml"], "--config"),
    strategy_id: Optional[str] = typer.Option(None, "--strategy-id"),
    model_type: Optional[str] = typer.Option(None, "--model-type"),
    approval_status: Optional[str] = typer.Option(None, "--approval-status"),
    limit: int = typer.Option(100, "--limit"),
):
    cfg = _load_app_config(config)
    frame = ExperimentRegistry.from_config(cfg).query(
        strategy_id=strategy_id,
        model_type=model_type,
        approval_status=approval_status,
        limit=limit,
    )
    typer.echo(frame.to_string(index=False))


@app.command("registry-refresh")
def registry_refresh_cmd(
    config: list[str] = typer.Option(["configs/default.yaml"], "--config"),
):
    cfg = _load_app_config(config)
    typer.echo(json.dumps({"refreshed_runs": refresh_registry(cfg)}))


@app.command("ingest-style-factors")
def ingest_style_factors_cmd(
    config: list[str] = typer.Option(["configs/default.yaml"], "--config"),
    refresh: bool = typer.Option(False, "--refresh"),
):
    cfg = _load_app_config(config)
    frame = load_fama_french_daily(
        cfg.research_validation.style_factor_cache,
        refresh=refresh,
    )
    typer.echo(
        json.dumps(
            {
                "rows": len(frame),
                "start": str(pd.to_datetime(frame["date"]).min().date()),
                "end": str(pd.to_datetime(frame["date"]).max().date()),
            }
        )
    )


@app.command("baseline-comparison")
def baseline_comparison_cmd(
    experiment_run: list[str] = typer.Option(..., "--experiment-run"),
    output: str = typer.Option(
        "reports/research/holdout_baseline_comparison.md",
        "--output",
    ),
    config: list[str] = typer.Option(["configs/default.yaml"], "--config"),
):
    cfg = _load_app_config(config)
    runs_root = resolve_path(cfg.paths.reports) / "runs"
    paths = [
        Path(value) if Path(value).exists() else runs_root / value
        for value in experiment_run
    ]
    frame, markdown_path, csv_path = generate_baseline_comparison(
        paths,
        output,
        runs_root,
    )
    typer.echo(
        json.dumps(
            {
                "experiments": len(frame),
                "markdown": str(markdown_path),
                "csv": str(csv_path),
            }
        )
    )


@app.command("historical-replay")
def historical_replay_cmd(
    suite: str = typer.Option(
        "configs/historical_replay/suite.yaml",
        "--suite",
    ),
    config: list[str] = typer.Option(["configs/default.yaml"], "--config"),
):
    cfg = _load_app_config(config)
    try:
        result = run_historical_replay(cfg, suite)
    except (ValueError, RuntimeError) as exc:
        typer.echo(f"invalid: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(
        json.dumps(
            {
                "status": "valid",
                "run_id": result.run_id,
                "run_path": str(result.run_path),
                "decision": result.decision,
                "selected_strategy_id": result.selected_strategy_id,
                "challenger_strategy_id": result.challenger_strategy_id,
            },
            indent=2,
        )
    )


@app.command("forward-init")
def forward_init_cmd(
    replay_run: str = typer.Option(..., "--replay-run"),
    start: str = typer.Option("2026-06-15", "--start"),
    end: str = typer.Option("2026-12-15", "--end"),
    study_id: Optional[str] = typer.Option(None, "--study-id"),
    config: list[str] = typer.Option(["configs/default.yaml"], "--config"),
):
    cfg = _load_app_config(config)
    try:
        result = initialize_forward_validation(
            cfg,
            replay_run,
            start_date=start,
            end_date=end,
            study_id=study_id,
        )
    except ValueError as exc:
        typer.echo(f"invalid: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(
        json.dumps(
            {
                "study_id": result.study_id,
                "root": str(result.root),
                "status": result.status,
            },
            indent=2,
        )
    )


@app.command("forward-record")
def forward_record_cmd(
    forward_root: str = typer.Option(..., "--forward-root"),
    date: str = typer.Option(..., "--date"),
):
    try:
        result = record_forward_day(forward_root, date)
    except (ValueError, RuntimeError) as exc:
        typer.echo(f"invalid: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(
        json.dumps(
            {
                "study_id": result.study_id,
                "root": str(result.root),
                "status": result.status,
            },
            indent=2,
        )
    )


@app.command("forward-status")
def forward_status_cmd(
    forward_root: str = typer.Option(..., "--forward-root"),
):
    try:
        status = evaluate_forward_validation(forward_root)
    except ValueError as exc:
        typer.echo(f"invalid: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(json.dumps(status, indent=2))


@app.command("approve-rebalance")
def approve_rebalance_cmd(
    packet: str = typer.Option(..., "--packet"),
    state: str = typer.Option(..., "--state"),
    reviewer: str = typer.Option(..., "--reviewer"),
    note: str = typer.Option("", "--note"),
    config: list[str] = typer.Option(["configs/default.yaml"], "--config"),
):
    cfg = _load_app_config(config)
    if state not in {"approved_for_candidate", "rejected"}:
        raise typer.BadParameter(
            "state must be approved_for_candidate or rejected",
            param_hint="--state",
        )
    updated = transition_approval(cfg, packet, state, reviewer, note)
    typer.echo(updated.model_dump_json(indent=2))


@app.command("job-definitions")
def job_definitions_cmd():
    typer.echo(
        json.dumps(
            {name: definition.__dict__ for name, definition in JOB_DEFINITIONS.items()},
            indent=2,
        )
    )


@app.command("ingest-prices")
def ingest_prices_cmd(
    config: list[str] = typer.Option(["configs/default.yaml"], "--config"),
    start: str = typer.Option(..., "--start"),
    end: Optional[str] = typer.Option(None, "--end"),
):
    cfg = _load_app_config(config)
    ingest_prices(
        cfg,
        start_date=start,
        end_date=end,
        tickers=_research_tickers(cfg, start, end),
    )


@app.command("ingest-fundamentals")
def ingest_fundamentals_cmd(
    config: list[str] = typer.Option(["configs/default.yaml"], "--config"),
):
    cfg = _load_app_config(config)
    result = ingest_fundamentals(
        cfg,
        tickers=_research_tickers(cfg),
    )
    typer.echo(
        json.dumps(
            {
                "requested": result.requested,
                "mapped": result.mapped,
                "cached": result.cached,
                "fetched": result.fetched,
                "failed": result.failed,
                "no_mapping": result.no_mapping,
                "fundamental_rows": len(result.fundamentals),
            }
        )
    )


@app.command("ingest-sec-text")
def ingest_sec_text_cmd(
    config: list[str] = typer.Option(["configs/default.yaml"], "--config"),
    max_filings: int = typer.Option(100, "--max-filings"),
    as_of: Optional[str] = typer.Option(None, "--as-of"),
):
    cfg = _load_app_config(config)
    cached = cache_sec_filing_text(
        cfg,
        max_filings=max_filings,
        as_of_date=as_of,
    )
    cached_count = (
        int(cached["text_cached"].fillna(False).sum())
        if "text_cached" in cached
        else 0
    )
    typer.echo(
        json.dumps(
            {
                "attempted_filings": len(cached),
                "cached_filings": cached_count,
                "failed_filings": len(cached) - cached_count,
            }
        )
    )


@app.command("ingest-macro")
def ingest_macro_cmd(
    config: list[str] = typer.Option(["configs/default.yaml"], "--config"),
):
    cfg = _load_app_config(config)
    ingest_macro(cfg)


@app.command("build-universe")
def build_universe_cmd(
    config: list[str] = typer.Option(["configs/default.yaml"], "--config"),
    date: str = typer.Option(..., "--date"),
):
    cfg = _load_app_config(config)
    build_and_store_universe(pd.Timestamp(date), cfg)


@app.command("compute-factors")
def compute_factors_cmd(
    config: list[str] = typer.Option(["configs/default.yaml"], "--config"),
    date: str = typer.Option(..., "--date"),
):
    cfg = _load_app_config(config)
    _require_valid_research_data(cfg, cfg.backtest.start_date, date)
    compute_and_store_factor_values(pd.Timestamp(date), cfg)


@app.command("score")
def score_cmd(
    config: list[str] = typer.Option(["configs/default.yaml"], "--config"),
    date: str = typer.Option(..., "--date"),
):
    cfg = _load_app_config(config)
    _require_valid_research_data(cfg, cfg.backtest.start_date, date)
    compute_and_store_scores(pd.Timestamp(date), cfg)


@app.command("rebalance")
def rebalance_cmd(
    config: list[str] = typer.Option(["configs/default.yaml"], "--config"),
    date: str = typer.Option(..., "--date"),
):
    cfg = _load_app_config(config)
    _require_valid_research_data(cfg, cfg.backtest.start_date, date)
    run_rebalance(pd.Timestamp(date), cfg, enforce_data_gate=False)


@app.command("backtest")
def backtest_cmd(
    config: list[str] = typer.Option(["configs/default.yaml"], "--config"),
    start: str = typer.Option(..., "--start"),
    end: str = typer.Option(..., "--end"),
):
    cfg = _load_app_config(config)
    try:
        run_backtest(start, end, cfg, progress_callback=emit_progress)
    except (ValueError, RuntimeError) as exc:
        typer.echo(f"invalid: {exc}", err=True)
        raise typer.Exit(code=2) from exc


@app.command("quickstart")
def quickstart_cmd(
    config: list[str] = typer.Option(["configs/quickstart.yaml"], "--config"),
    start: str = typer.Option("2023-01-01", "--start"),
    end: Optional[str] = typer.Option(None, "--end"),
    target_symbols: Optional[int] = typer.Option(None, "--target-symbols"),
):
    cfg = _load_app_config(config)
    end_date = end or str(pd.Timestamp.today().date())
    try:
        result = run_quickstart(
            cfg,
            start,
            end_date,
            target_symbols=target_symbols,
        )
    except (ValueError, RuntimeError) as exc:
        typer.echo(f"invalid: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(
        json.dumps(
            {
                "status": "valid",
                "run_id": result.backtest.run_id,
                "run_path": str(result.backtest.run_path),
                "symbol_count": result.symbol_count,
                "price_rows": result.price_rows,
                "fundamental_rows": result.fundamental_rows,
                "membership_rows": result.membership_rows,
                "macro_rows": result.macro_rows,
            }
        )
    )


@app.command("risk-monitor")
def risk_monitor_cmd(
    config: list[str] = typer.Option(["configs/default.yaml"], "--config"),
    date: str = typer.Option(..., "--date"),
):
    cfg = _load_app_config(config)
    run_daily_risk_monitor(pd.Timestamp(date), cfg)


@app.command("operations-dry-run")
def operations_dry_run_cmd(
    end: str = typer.Option(..., "--end"),
    trading_days: int = typer.Option(10, "--trading-days"),
    max_attempts: int = typer.Option(2, "--max-attempts"),
    config: list[str] = typer.Option(["configs/default.yaml"], "--config"),
):
    cfg = _load_app_config(config)
    summary, markdown = run_operations_dry_run(
        cfg,
        end_date=end,
        trading_days=trading_days,
        max_attempts=max_attempts,
    )
    typer.echo(
        json.dumps(
            {
                "status": (
                    "valid"
                    if bool(summary["status"].eq("valid").all())
                    else "invalid"
                ),
                "trading_days": len(summary),
                "summary": str(markdown),
            }
        )
    )
    if not bool(summary["status"].eq("valid").all()):
        raise typer.Exit(code=2)


@app.command("dashboard")
def dashboard_cmd():
    try:
        bundle = generate_comprehensive_report("reports")
        typer.echo(f"Comprehensive report: {bundle.html_report}")
    except ValueError as exc:
        typer.echo(f"Comprehensive report unavailable: {exc}", err=True)
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "src/qss/dashboard/streamlit_app.py", "--server.headless", "true"],
        check=False,
    )


@app.command("run-monthly-pipeline")
def run_monthly_pipeline_cmd(
    config: list[str] = typer.Option(["configs/default.yaml"], "--config"),
    date: str = typer.Option(..., "--date"),
    start: Optional[str] = typer.Option(None, "--start"),
):
    cfg = _load_app_config(config)
    as_of_date = pd.Timestamp(date).normalize()
    research_start = pd.Timestamp(start or cfg.backtest.start_date).normalize()
    if as_of_date < research_start:
        raise typer.BadParameter("--date must be on or after --start")
    start_date = str(research_start.date())
    end_date = str(as_of_date.date())
    price_start_date = str(
        (research_start - pd.DateOffset(years=2)).date()
    )
    emit_progress("universe", 0.02, "Synchronizing point-in-time universe")
    sync_universe(
        cfg,
        start_date=start_date,
        end_date=end_date,
        validate_recent=cfg.universe.validation_provider != "disabled",
    )
    tickers = _research_tickers(cfg, start_date, end_date)
    emit_progress(
        "prices",
        0.18,
        f"Downloading price history for {len(tickers)} symbols",
    )
    ingest_prices(
        cfg,
        start_date=price_start_date,
        end_date=end_date,
        tickers=tickers,
    )
    emit_progress("fundamentals", 0.42, "Updating SEC fundamentals")
    ingest_fundamentals(
        cfg,
        tickers=tickers,
    )
    emit_progress("sectors", 0.56, "Enriching sector metadata")
    enrich_sector_metadata(
        cfg,
        start_date=start_date,
        end_date=end_date,
        tickers=tickers,
    )
    emit_progress("macro", 0.62, "Updating macro observations")
    ingest_macro(cfg)
    emit_progress("universe-build", 0.70, "Building investable universe")
    build_and_store_universe(as_of_date, cfg)
    emit_progress("validation", 0.76, "Validating research data")
    _require_valid_research_data(cfg, start_date, end_date)
    emit_progress("factors", 0.82, "Computing factor values")
    compute_and_store_factor_values(as_of_date, cfg)
    emit_progress("scores", 0.88, "Computing alpha scores")
    compute_and_store_scores(as_of_date, cfg)
    emit_progress("rebalance", 0.94, "Optimizing portfolio rebalance")
    run_rebalance(as_of_date, cfg, enforce_data_gate=False)
    macro_observations = read_parquet(Path(cfg.paths.silver_data) / "macro" / "macro_observations.parquet")
    prices = read_parquet(Path(cfg.paths.silver_data) / "prices" / "prices_daily.parquet")
    compute_macro_regime(as_of_date, macro_observations, prices, cfg)
    emit_progress("complete", 1.0, "Monthly pipeline complete")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
