# Development Acceptance Checklist

Last updated: `2026-06-13`

## Status

All five milestones from the construction-report audit are implemented. Remaining
items are operating choices or institutional data upgrades, not missing acceptance
capabilities.

| Milestone | Status | Primary evidence |
| --- | --- | --- |
| Labels and validation | Complete | `src/qss/labels/`, label snapshots, `label_validation.csv` |
| Walk-forward and ML baselines | Complete | purged/embargo splits, Ridge, ElasticNet, LightGBM, fold artifacts |
| Factor metadata and critic audit | Complete | `factor_metadata.json`, `bias_review.md`, deterministic checks |
| Registry and approval workflow | Complete | `experiments/registry.duckdb`, approval packets and human transition CLI |
| Text/event factors and operations | Complete | SEC event metadata, filing text cache, event labels, job definitions |

## Milestone 1: Labels and validation

- [x] Standalone `src/qss/labels/` package.
- [x] Persisted `forward_return` and `cross_sectional_rank` labels.
- [x] Horizon, label timestamps, overlap, purge, embargo, and version metadata.
- [x] Versioned gold label paths and run-level label configuration.
- [x] Horizon, overlap, event-window, and no-future-leakage tests.

Acceptance: every backtest writes reproducible label artifacts and explicitly marks
overlap.

## Milestone 2: Walk-forward and ML

- [x] Rolling/expanding walk-forward engine.
- [x] Purge and embargo enforcement from label end timestamps.
- [x] Ridge and ElasticNet linear baselines.
- [x] LightGBM tree baseline as an installed project dependency.
- [x] Split manifests, predictions, fold metrics, aggregate metrics, and model config.
- [x] Top-N equal-weight portfolio mapping with gross/net returns and transaction costs.
- [x] YAML/config switch through `ml` and experiment `model` sections.

Acceptance: an ML-enabled backtest runs end to end and the acceptance suite requires
non-empty fold metrics.

## Milestone 3: Metadata and critic audit

- [x] Metadata for every production factor.
- [x] Description, inputs, lookback, skip window, horizon, cost sensitivity, PIT needs,
  leakage checks, and version.
- [x] Run-level metadata snapshot.
- [x] Deterministic `bias_review.md` and JSON outputs.
- [x] Survivorship, sector, concentration, cost coverage, and sample coverage checks.

Acceptance: `configured_factor_metadata()` rejects any configured factor without a
metadata definition.

## Milestone 4: Registry and approval

- [x] DuckDB experiment registry with query CLI.
- [x] Run, strategy, universe, factors, label, model, date range, validation, metrics,
  status, and approval fields.
- [x] Candidate rebalance weights and internal orders are isolated.
- [x] Approval packet starts in `review_required`.
- [x] Only an explicit human CLI transition can create
  `approved_target_weights.csv`.

Acceptance: no rebalance path publishes approved weights automatically.

## Milestone 5: Text/events and operations

- [x] SEC filing type, timestamp, event type, accession, document, and cache key.
- [x] Filing text cache command using SEC document URLs and stable keys.
- [x] Deterministic risk-disclosure text factor.
- [x] Event-window return labels.
- [x] Text-factor prototype strategy config.
- [x] Daily risk, monthly rebalance, and registry refresh job definitions with owners.
- [x] Documented job runner; Prefect remains an optional deployment choice.

Acceptance: cached filing text produces reproducible factor values and event labels.

## Robustness coverage

- [x] Subperiod tests.
- [x] Cost sensitivity at configurable bps levels.
- [x] Top-N sensitivity.
- [x] Rebalance-day shift sensitivity.
- [x] Existing delisting sensitivity.

## Final verification

```powershell
.\.venv\Scripts\python.exe -m compileall src tests qss
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check src tests qss
.\.venv\Scripts\python.exe -m qss.cli --help
```

For a generated backtest run:

```powershell
.\.venv\Scripts\python.exe -m qss.cli acceptance-check `
  --config configs/default.yaml `
  --run-path reports/runs/<run_id>
```
