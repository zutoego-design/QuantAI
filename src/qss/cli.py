from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from qss.acceptance import run_acceptance_checks
from qss.backtest.engine import run_backtest
from qss.config.loader import get_config
from qss.data.storage import ensure_data_directories, read_parquet
from qss.data.validation import validate_research_data
from qss.factors.registry import compute_and_store_factor_values
from qss.ingestion.fred import ingest_macro
from qss.ingestion.prices_yfinance import ingest_prices
from qss.ingestion.sec_edgar import ingest_fundamentals
from qss.logging_utils import configure_logging
from qss.macro.regime import compute_macro_regime
from qss.model.scoring import compute_and_store_scores
from qss.portfolio.rebalance import run_rebalance
from qss.reporting.service import render_saved_backtest
from qss.research.orchestrator import ResearchOrchestrator, load_experiment_spec
from qss.risk.monitor import run_daily_risk_monitor
from qss.runs.baseline import snapshot_legacy_baseline
from qss.universe.builder import build_and_store_universe
from qss.universe.sync import sync_universe

app = typer.Typer(help="Quant Stock Selection System CLI")


def _load_app_config(config_paths: list[str]):
    config = get_config(config_paths)
    ensure_data_directories(config)
    configure_logging(config.runtime.log_level)
    return config


def _seed_tickers(config) -> list[str]:
    security_master_path = Path(config.paths.silver_data) / "universe" / "security_master.parquet"
    if security_master_path.exists():
        master = pd.read_parquet(security_master_path)
        if "symbol" in master and not master.empty:
            return sorted(master["symbol"].dropna().astype(str).unique().tolist())
    seed = pd.read_csv(Path(config.universe.seed_metadata_path))
    return seed["symbol"].tolist()


def _require_valid_research_data(config, start: str | None = None, end: str | None = None):
    if not config.runtime.research_mode:
        return
    result = validate_research_data(config, start, end)
    if result.status != "valid":
        raise RuntimeError(
            f"Research data gate failed. Review validation run {result.run_path}."
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


@app.command("ingest-prices")
def ingest_prices_cmd(
    config: list[str] = typer.Option(["configs/default.yaml"], "--config"),
    start: str = typer.Option(..., "--start"),
    end: Optional[str] = typer.Option(None, "--end"),
):
    cfg = _load_app_config(config)
    ingest_prices(cfg, start_date=start, end_date=end)


@app.command("ingest-fundamentals")
def ingest_fundamentals_cmd(
    config: list[str] = typer.Option(["configs/default.yaml"], "--config"),
):
    cfg = _load_app_config(config)
    ingest_fundamentals(cfg, tickers=_seed_tickers(cfg))


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
        run_backtest(start, end, cfg)
    except (ValueError, RuntimeError) as exc:
        typer.echo(f"invalid: {exc}", err=True)
        raise typer.Exit(code=2) from exc


@app.command("risk-monitor")
def risk_monitor_cmd(
    config: list[str] = typer.Option(["configs/default.yaml"], "--config"),
    date: str = typer.Option(..., "--date"),
):
    cfg = _load_app_config(config)
    run_daily_risk_monitor(pd.Timestamp(date), cfg)


@app.command("dashboard")
def dashboard_cmd():
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
    as_of_date = pd.Timestamp(date)
    start_date = start or cfg.backtest.start_date
    ingest_prices(cfg, start_date=start_date, end_date=str(as_of_date.date()))
    ingest_fundamentals(cfg, tickers=_seed_tickers(cfg))
    ingest_macro(cfg)
    build_and_store_universe(as_of_date, cfg)
    _require_valid_research_data(cfg, start_date, str(as_of_date.date()))
    compute_and_store_factor_values(as_of_date, cfg)
    compute_and_store_scores(as_of_date, cfg)
    run_rebalance(as_of_date, cfg, enforce_data_gate=False)
    macro_observations = read_parquet(Path(cfg.paths.silver_data) / "macro" / "macro_observations.parquet")
    prices = read_parquet(Path(cfg.paths.silver_data) / "prices" / "prices_daily.parquet")
    compute_macro_regime(as_of_date, macro_observations, prices, cfg)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
