# Project Status

Last updated: `2026-06-12`

## Status Summary

QuantAI `v0.2.0` now defaults to a strict S&P 500 research path:

- S&P 500 point-in-time membership reconstructed from Wikipedia constituents and constituent changes
- strict research-mode data gates with no synthetic prices, fundamentals, or macro rows
- live Yahoo/Stooq prices with explicit coverage gaps
- SEC company facts and SIC-derived sector metadata
- FRED macro observations through the public CSV/API path
- holdings, cash, trade, transaction-cost, and daily valuation ledgers
- explicit delisting sensitivity at `0%`, `-30%`, and `-100%`
- professional performance, drawdown, trading, concentration, factor, and data diagnostics
- bounded `ExperimentSpec` and `ResearchOrchestrator`
- shared structured report schema for CLI, HTML, JSON, Parquet, and Streamlit

The dashboard default action is **Run S&P 500 Data + Backtest**. Quickstart remains
only as an optional smoke test and is labeled as simulated-data output.

## Current Validation

The latest strict live-data run is valid:

- run: `reports/runs/20260612T153324Z-backtest-8f5a55b3`
- universe history: 41 of 41 months
- S&P 500 research securities: 559 unique symbols
- research prices: 96.4% required-symbol coverage
- SEC fundamentals: 96.4% research-symbol coverage
- sector metadata: 85.7% research-symbol coverage
- macro observations: all 6 configured FRED series ready
- synthetic rows: 0
- acceptance checks: passed

Manifest bias/source flag:

- `sp500_point_in_time_wikipedia_reconstruction`

## Verification Record

- `python -m compileall src tests qss`: passed
- `python -m pytest -q`: passed, 51 tests
- `ruff check src tests qss`: passed
- strict data validation: passed
- strict S&P 500 backtest: passed
- acceptance check: passed
- Streamlit AppTest: passed with 0 exceptions

The in-app Browser control tool was not exposed in this session, so frontend
verification used Streamlit AppTest rather than live browser automation.

## External Dependencies

Set `SEC_USER_AGENT` to a real contact string before heavier SEC use. The checked-in
config has a fallback user-agent string so local smoke and validation commands can
run, but responsible SEC usage should identify the operator.

`FRED_API_KEY` is optional. Without it, macro ingestion uses FRED's public graph CSV
endpoint.
