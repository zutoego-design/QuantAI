# Research Methodology

## Universe

The default research universe is the S&P 500 historical membership set:

- current constituents are read from the S&P 500 constituents table
- historical adds/removes are read from the constituent-change log
- monthly snapshots are reconstructed by reversing changes from the current table
- future-dated changes are reversed when the requested end date is before their
  effective date

This avoids current-membership backfill. The manifest marks this source model as:

- `sp500_point_in_time_wikipedia_reconstruction`

The source is still a free-data reconstruction, not a licensed index membership
feed.

## Point-In-Time Data

SEC company facts are stored as metric observations with:

- report period
- filing and available dates
- form and accession
- unit, source, ingestion time, and quality status

Every metric is selected independently using the latest observation available on the
signal date. Available dates are never allowed to precede period-end dates. Missing
factors remain missing; securities below the configured factor-coverage threshold are
excluded.

## Portfolio And Accounting

Signals are generated on the last trading day of each month and execute after the
configured trading-day lag. The backtest keeps explicit holdings and cash values.
Weights drift with market returns and turnover is measured against actual pre-trade
weights.

Held-security missing returns are never silently set to zero. Intermediate missing
returns invalidate the run. Terminal disappearance is evaluated under three explicit
liquidation scenarios: last tradable value, `-30%`, and `-100%`.

Costs include commission, fixed slippage, ADV participation, volatility impact, and
a capacity estimate. Strict runs reject trades above the configured ADV limit.

## Optimizer

The optimizer uses a candidate pool plus existing holdings so turnover constraints
remain feasible across monthly rebalances. `target_num_holdings` is a target, not an
exact cardinality constraint; strict runs reject optimizer fallback but allow the
constrained optimizer to produce a variable number of valid active holdings.

## Research Workflow

The orchestrator executes:

1. S&P 500 universe reconstruction
2. live price, SEC fundamentals, and FRED macro ingestion
3. strict data gates
4. full-period portfolio backtest
5. transaction-cost and delisting sensitivity
6. structured report generation
7. acceptance checks

The orchestrator cannot write raw inputs, overwrite baseline artifacts, bypass gates,
or automatically promote a strategy.

## Publication Gates

A result is publishable only when:

- synthetic rows equal zero
- S&P 500 monthly membership is present for the requested period
- source audit for the reconstructed universe is present
- long-track member price coverage is at least 95%
- sector mapping coverage is at least the configured threshold
- primary, secondary, and internal benchmarks are complete
- optimizer fallback is not used
- ADV and missing-return constraints are satisfied
