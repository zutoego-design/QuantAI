# Research Methodology

## Universe

The research universe is Nasdaq (`XNAS`) operating equity:

- included: common shares, ADRs, and REITs
- excluded: ETF/ETN, funds, preferred shares, warrants, units, and test securities

Alpha Vantage monthly listing snapshots provide the approximate history from
`2010-01-01`. Nasdaq Trader provides the current and forward snapshot. Massive/Polygon
validates the most recent two years. Membership and symbol history use permanent
`security_id` values; a ticker is not the primary identity.

## Point-In-Time Data

SEC company facts are stored as metric observations with:

- report period
- filing and available dates
- form and accession
- unit, source, ingestion time, and quality status

Every metric is selected independently using the latest observation available on the
signal date. Period-end dates are never treated as publication dates. Missing factors
remain missing; securities below 80% configured factor coverage are excluded.

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

## Research Workflow

The orchestrator executes:

1. data and bias gates
2. factor diagnostics
3. full-period portfolio backtest
4. first-half and second-half checks when the sample spans at least two years
5. transaction-cost and delisting sensitivity
6. legacy baseline comparison
7. a structured research memo for human review

The orchestrator cannot write raw inputs, overwrite baseline artifacts, bypass gates,
or automatically promote a strategy.

## Publication Gates

A result is publishable only when:

- synthetic rows equal zero
- monthly membership history covers at least 95% of the requested period
- recent validation month coverage is at least 95% and minimum Jaccard is at least 95%
- recent member price coverage is at least 98%
- long-track member price coverage is at least 95%
- sector mapping coverage is at least 90%
- primary, secondary, and internal benchmarks are complete
- optimizer fallback is not used
- required holding count and ADV constraints are satisfied
