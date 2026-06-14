# Final Code Review

Review date: `2026-06-12`

## Findings Fixed

### P0

- Removed missing held-security return zero filling from the backtest and risk monitor.
- Prevented legacy synthetic data from producing a valid research report.
- Separated source membership from strategy-eligible universe output.
- Prevented the frontend from presenting legacy metrics as the latest valid run.

### P1

- Replaced fixed target-weight accounting with holdings/cash/trade ledgers.
- Corrected monthly returns from summation to compounding.
- Rejected optimizer fallback in research mode while allowing constrained optimizers
  to produce a variable number of valid active holdings.
- Disabled false tracking-error constraints without real benchmark weights.
- Changed fundamentals from latest-row selection to latest observation per metric.
- Removed factor median/zero imputation and added factor-coverage gating.
- Removed zero-filled asset returns from covariance estimation.
- Added source precedence so lower-quality refreshes cannot overwrite live records.
- Added S&P 500 point-in-time universe reconstruction and source audit gates.
- Added SIC sector enrichment and prevented unknown sectors from making constraints infeasible.
- Made ExperimentSpec overrides effective and added subperiod/baseline workflow stages.

### P2

- Added run manifests, schema versioning, structured reports, baseline hashes, and invalid-run records.
- Versioned rebalance and risk reports.
- Added primary, secondary, equal-weight, and cap-weight benchmark diagnostics.
- Added factor IC, Rank IC, ICIR, t-statistic, quantile, monotonicity, turnover, decay, and correlation reports.
- Added drawdown episodes, cost attribution, ADV participation, capacity, concentration, and active sector exposure.

## Residual Risks

- Historical membership and delisting information remain free-data approximations.
- Permanent identity and SIC sector mapping are inferred rather than licensed reference data.
- Live acceptance depends on free Yahoo/Stooq, SEC, FRED, and Wikipedia availability.
- The workspace is not a Git repository, so review evidence is based on files, manifests, tests, and generated hashes rather than commits.

No unresolved P0, P1, or P2 code defects were found after the final fix pass.
