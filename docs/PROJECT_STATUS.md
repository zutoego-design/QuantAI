# Project Status

Last updated: `2026-06-12`

## Status Summary

QuantAI `v0.2.0` implements the trusted research architecture:

- isolated run manifests under `reports/runs/<run_id>/`
- strict research-mode data gates with no synthetic inputs
- approximate point-in-time Nasdaq operating-equity membership
- permanent security IDs, ticker history, listing intervals, and partitioned membership
- per-metric SEC observation selection by market-available date
- holdings, cash, trade, transaction-cost, and daily valuation ledgers
- explicit delisting sensitivity at `0%`, `-30%`, and `-100%`
- professional performance, drawdown, trading, concentration, factor, and data diagnostics
- bounded `ExperimentSpec` and `ResearchOrchestrator`
- shared structured report schema for CLI, HTML, JSON, Parquet, and Streamlit

The implementation and deterministic code acceptance pass. The live historical-data
acceptance is intentionally `invalid` until the required free API credentials are
configured and the resumable universe/data synchronization is completed.

## Current Validation

The existing 26-stock output is preserved at:

- `reports/baselines/legacy-demo-20260612.json`

It is marked `trusted_for_strategy_decisions: false`.

The current workspace data fails strict publication gates for these concrete reasons:

- 1,118 synthetic rows remain across legacy price, fundamental, and macro artifacts
- only two historical membership months exist instead of monthly history from 2010
- Nasdaq Composite (`^IXIC`) is absent from the legacy price file
- the legacy QQQ series has one incomplete return row
- no recent Massive cross-source validation artifact exists
- Alpha Vantage, Massive/Polygon, FRED, and SEC user-agent environment settings are absent

This is expected behavior: no new valid strategy report or `reports/latest_run.json`
is published when a gate fails.

## Verification Record

- `python -m compileall src tests qss`: passed
- `python -m pytest -q`: passed, 25 tests
- `ruff check src tests qss`: passed
- CLI help and all new command registrations: passed
- deterministic point-in-time end-to-end backtest: passed twice with identical results
- strict legacy-data backtest rejection: passed
- strict data validation rejection: passed
- Streamlit smoke on `http://127.0.0.1:8512`: passed, HTTP 200

Latest strict validation run:

- run ID: `20260612T084608Z-data-validation-5830ace3`
- status: `invalid`
- monthly membership coverage: `1.02%`
- recent cross-source validation month coverage: `0%`

## Remaining External Work

Configure:

- `ALPHAVANTAGE_API_KEY`
- `MASSIVE_API_KEY` or `POLYGON_API_KEY`
- `FRED_API_KEY`
- `SEC_USER_AGENT`

Then repeatedly run `sync-universe`; the free-tier synchronizer caches each monthly
snapshot and limits each provider to 25 new requests per run. After synchronization,
ingest full-universe prices and SEC fundamentals, run `validate-data`, and only then
run experiments or published backtests.

Free data cannot support a claim of fully eliminated survivorship bias. Reports must
continue to label the 2010+ track as approximate and the recent two-year track as
cross-validated.
