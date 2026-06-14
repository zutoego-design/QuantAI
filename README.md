# QuantAI Research System

S&P 500 point-in-time equity research, factor diagnostics, ledger-based backtesting,
professional reporting, and bounded AI experiment orchestration.

This is a research and monitoring system. It does not route live orders.

## Current Status

`v0.2` and `v0.3` are archived delivery baselines. The active system adds
preregistered confirmatory studies, SHA-256 data snapshots, a shared holdout portfolio
simulator for rule and ML scores, and statistical evidence gates.

The former canonical rule experiment is retained as a `legacy_reference`:

- `reports/runs/20260613T133519Z-experiment-7ad846cd`
- full child: `reports/runs/20260613T133520Z-backtest-11107a4e`
- status: `valid`
- data cutoff: `2026-06-11`
- model: `rule_score`
- acceptance: `20260613T144054Z-acceptance-dafe14fb`, 15/15 passed
- bias review: `eligible_for_human_review`

Its engineering artifacts remain valid, but it is not confirmatory evidence under the
active methodology. New research claims must use a development period, an isolated
holdout period, a frozen data snapshot, multiple-testing controls, block bootstrap,
Deflated Sharpe, and style-factor attribution.

The first real confirmatory example under the active methodology is
[`20260614T011051Z-experiment-ec84fa0a`](reports/runs/20260614T011051Z-experiment-ec84fa0a/).
Its artifacts are `valid`, while its research evidence is `rejected` because the
Deflated Sharpe probability is below 95% after three actual attempts and the
preregistered factors did not pass the joint direction and FDR gate. See the
[holdout baseline comparison](reports/research/holdout_baseline_comparison.md).

Human approval approve/reject paths and a ten-trading-day risk dry run are validated.
These results do not authorize live or paper-broker order routing.

See:

- [Documentation](docs/README.md)
- [Research methodology](docs/RESEARCH_METHODOLOGY.md)
- [Research credibility plan](docs/RESEARCH_CREDIBILITY_PLAN.md)
- [Operations](docs/OPERATIONS.md)
- [Archived v0.3 documents](docs/archive/v0.3/README.md)
- [Archived v0.2 documents](docs/archive/v0.2/README.md)

## Setup

Use Python `>=3.11`.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Set a real SEC contact string before heavier SEC synchronization:

```powershell
$env:SEC_USER_AGENT = "Your Name your.email@example.com"
```

`FRED_API_KEY` is optional but recommended; without it Macro uses FRED's public graph
CSV endpoint.

## Recommended Operating Flow

Use the dashboard button **Run S&P 500 Data + Backtest**. It reconstructs S&P 500
membership from the current constituents table plus the historical constituent-change
log, downloads live prices, SEC fundamentals, and FRED macro data, validates that
synthetic rows are zero, and then runs the strict backtest.

The equivalent foreground CLI command is:

```powershell
.\.venv\Scripts\python.exe -m qss.cli autopilot-run `
  --config configs/default.yaml `
  --start 2023-01-01 `
  --end 2026-06-12 `
  --run-backtest
```

For a background worker, use:

The equivalent CLI commands are:

```powershell
.\.venv\Scripts\python.exe -m qss.cli autopilot-start `
  --config configs/default.yaml `
  --start 2023-01-01 `
  --end 2026-06-12

.\.venv\Scripts\python.exe -m qss.cli autopilot-status `
  --config configs/default.yaml
```

Autopilot state and logs are stored under `reports/data_autopilot/`.

Quickstart remains available only as an optional smoke test. It intentionally uses
simulated fundamentals/macro and writes to `data/quickstart`, so do not use it for
research conclusions.

The individual commands below remain available for diagnostics:

```powershell
.\.venv\Scripts\python.exe -m qss.cli data-status `
  --config configs/default.yaml `
  --start 2023-01-01 `
  --end 2026-06-12

.\.venv\Scripts\python.exe -m qss.cli sync-universe `
  --config configs/default.yaml `
  --start 2023-01-01 `
  --end 2026-06-12

.\.venv\Scripts\python.exe -m qss.cli ingest-prices `
  --config configs/default.yaml `
  --start 2021-01-01 `
  --end 2026-06-12

.\.venv\Scripts\python.exe -m qss.cli ingest-fundamentals `
  --config configs/default.yaml

.\.venv\Scripts\python.exe -m qss.cli ingest-macro `
  --config configs/default.yaml

.\.venv\Scripts\python.exe -m qss.cli validate-data `
  --config configs/default.yaml `
  --start 2023-01-01 `
  --end 2026-06-12
```

Only after validation is `valid`:

```powershell
.\.venv\Scripts\python.exe -m qss.cli backtest `
  --config configs/default.yaml `
  --start 2023-01-01 `
  --end 2026-06-12

.\.venv\Scripts\python.exe -m qss.cli acceptance-check `
  --config configs/default.yaml
```

## AI Research Experiments

An experiment YAML or JSON defines:

- hypothesis
- research stage and study/trial identity
- development and independent holdout periods for confirmatory studies
- primary metric, threshold, null hypothesis, and expected factor directions
- configured S&P 500 historical universe
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
backtests, cost/delisting/Top-N/rebalance-day sensitivity, optional purged walk-forward
ML evaluation, baseline comparison, critic audit, and research memo. It does not modify
raw data or automatically promote strategies.

Within one experiment, normalized datasets and per-signal factor snapshots are reused.
Robustness children emit metrics-only artifacts and run with two workers by default,
while the canonical full child retains the complete acceptance bundle.

## Registry And Approval

```powershell
.\.venv\Scripts\python.exe -m qss.cli registry-refresh
.\.venv\Scripts\python.exe -m qss.cli registry-query --model-type ridge
```

The confirmatory example is `experiments/confirmatory_rule_score.yaml`.

Monthly rebalances write candidate weights and internal orders. Only an explicit human
transition can create approved target weights:

```powershell
.\.venv\Scripts\python.exe -m qss.cli approve-rebalance `
  --packet reports/approvals/<run_id>/approval_packet.json `
  --state approved_for_candidate `
  --reviewer reviewer@example.com
```

Cache SEC filing text for the event-factor pipeline with:

```powershell
.\.venv\Scripts\python.exe -m qss.cli ingest-sec-text --max-filings 100
```

## Frontend

```powershell
start_frontend.bat
```

Open `http://127.0.0.1:8501`.

Before the frontend starts, it generates a comprehensive report from the newest
valid standalone backtest or research experiment. Internal metrics-only robustness
runs are excluded. Confirmatory experiments use their holdout ledger and keep
artifact validity separate from the research evidence decision. The latest report pointer is
`reports/comprehensive/latest.json`; `reports/latest_run.json` is only a fallback
for older backtest views.

## Main Interfaces

- `UniverseProvider.snapshot(as_of_date)`
- `PriceProvider.fetch(security_ids, start, end)`
- `ExperimentSpec`
- `ResearchProtocol`
- `BacktestRunSpec`
- `RunManifest`
- `ReportBundle`

New CLI commands:

- `snapshot-legacy-baseline`
- `sync-universe`
- `data-status`
- `autopilot-start`
- `autopilot-status`
- `autopilot-stop`
- `validate-data`
- `quickstart`
- `run-experiment`
- `render-report`
- `acceptance-check`
- `registry-query`
- `registry-refresh`
- `ingest-style-factors`
- `approve-rebalance`
- `ingest-sec-text`
- `job-definitions`
- `historical-replay`
- `forward-init`
- `forward-record`
- `forward-status`

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
- feature snapshots, factor metadata, labels, and label validation
- data snapshot identity and preregistered protocol metadata
- ML split manifests and fold metrics when enabled
- common-ledger rule and ML holdout evaluations
- HAC/FDR factor evidence, block-bootstrap intervals, Deflated Sharpe, and
  Fama-French style attribution for confirmatory experiments
- an independent `supported`, `inconclusive`, or `rejected` research decision
- deterministic bias review and final research report
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

- S&P 500 membership is reconstructed from the free Wikipedia constituents and
  changes tables; licensed index membership remains the institutional upgrade.
- Some removed constituents have incomplete free Yahoo/Stooq price histories.
- Yahoo Finance and Stooq are not authoritative delisting or corporate-action feeds.
- SEC SIC sector mapping is approximate and is not official GICS.
- FRED revision vintages are not reconstructed.
