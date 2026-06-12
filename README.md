# QuantAI Research System v0.2

Point-in-time Nasdaq equity research, factor diagnostics, ledger-based backtesting,
professional reporting, and bounded AI experiment orchestration.

This is a research and monitoring system. It does not route live orders.

## Current Status

The `v0.2.0` code architecture and deterministic acceptance suite are complete.
The existing 26-stock output is preserved as `legacy-demo` and is not trusted for
strategy decisions.

Live historical-data acceptance is currently `invalid` because the workspace does not
have the required free API credentials or complete 2010+ cached universe history.
Strict gates prevent those legacy inputs from publishing a new valid report.

See:

- [Project status](docs/PROJECT_STATUS.md)
- [Research methodology](docs/RESEARCH_METHODOLOGY.md)
- [Data limitations](docs/DATA_LIMITATIONS.md)
- [Final code review](docs/CODE_REVIEW.md)

## Setup

Use Python `>=3.11`.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Configure these environment variables before live synchronization:

```powershell
$env:ALPHAVANTAGE_API_KEY = "..."
$env:MASSIVE_API_KEY = "..."
$env:FRED_API_KEY = "..."
$env:SEC_USER_AGENT = "Your Name your.email@example.com"
```

`POLYGON_API_KEY` may be used instead of `MASSIVE_API_KEY`.

## Trusted Operating Flow

Free-tier universe synchronization is resumable and caches each date. Run it repeatedly
until `validate-data` passes.

```powershell
.\.venv\Scripts\python.exe -m qss.cli sync-universe `
  --config configs/default.yaml `
  --start 2010-01-01 `
  --end 2026-06-12

.\.venv\Scripts\python.exe -m qss.cli ingest-prices `
  --config configs/default.yaml `
  --start 2009-01-01 `
  --end 2026-06-12

.\.venv\Scripts\python.exe -m qss.cli ingest-fundamentals `
  --config configs/default.yaml

.\.venv\Scripts\python.exe -m qss.cli ingest-macro `
  --config configs/default.yaml

.\.venv\Scripts\python.exe -m qss.cli validate-data `
  --config configs/default.yaml `
  --start 2010-01-01 `
  --end 2026-06-12
```

Only after validation is `valid`:

```powershell
.\.venv\Scripts\python.exe -m qss.cli backtest `
  --config configs/default.yaml `
  --start 2010-01-01 `
  --end 2026-06-12

.\.venv\Scripts\python.exe -m qss.cli acceptance-check `
  --config configs/default.yaml
```

## AI Research Experiments

An experiment YAML or JSON defines:

- hypothesis
- point-in-time universe
- selected factors
- preprocessing overrides
- portfolio constraints
- cost assumptions
- dates and seed

Run it with:

```powershell
.\.venv\Scripts\python.exe -m qss.cli run-experiment `
  --config configs/default.yaml `
  --spec experiments/example.yaml
```

The orchestrator performs the data gate, factor diagnostics, full and subperiod
backtests, cost/delisting sensitivity, baseline comparison, and research memo. It does
not modify raw data or automatically promote strategies.

## Frontend

```powershell
start_frontend.bat
```

Open `http://127.0.0.1:8501`.

The frontend reads `reports/latest_run.json` only. If there is no valid backtest, it
does not fall back to legacy metrics.

## Main Interfaces

- `UniverseProvider.snapshot(as_of_date)`
- `PriceProvider.fetch(security_ids, start, end)`
- `ExperimentSpec`
- `BacktestRunSpec`
- `RunManifest`
- `ReportBundle`

New CLI commands:

- `snapshot-legacy-baseline`
- `sync-universe`
- `validate-data`
- `run-experiment`
- `render-report`
- `acceptance-check`

Existing pipeline commands remain available and use the same core services. In
research mode, factor computation, scoring, rebalancing, experiments, and backtests
cannot bypass the data gates.

## Outputs

Every published operation writes an isolated directory:

```text
reports/runs/<run_id>/
```

A valid backtest includes:

- `manifest.json`
- `report.json` and `report.html`
- `metrics.csv` and `metrics.parquet`
- daily returns, holdings, trades, and rebalances in CSV/Parquet
- monthly returns and drawdown episodes
- factor diagnostics, quantiles, decay, and correlations
- data diagnostics, sector exposure, concentration, and delisting sensitivity

The legacy baseline is:

```text
reports/baselines/legacy-demo-20260612.json
```

## Verification

```powershell
.\.venv\Scripts\python.exe -m compileall src tests qss
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check src tests qss
.\.venv\Scripts\python.exe -m qss.cli --help
```

## Important Limitations

- The 2010+ free-data membership track is approximate.
- Recent two-year cross-source validation does not make the long history institutional-grade.
- Yahoo Finance and Stooq are not authoritative delisting or corporate-action feeds.
- SEC SIC sector mapping is approximate and is not official GICS.
- FRED revision vintages are not reconstructed.
- This directory is not currently a Git checkout.
