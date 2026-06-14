# Research Memo

## Hypothesis
Configured rule-score factors provide the canonical non-ML baseline.

## Data Gate
Passed all configured point-in-time and coverage checks.

## Workflow
- Single-factor diagnostics: generated in the full child run.
- Portfolio backtest: completed.
- Subperiod checks: completed when the sample spans at least two years.
- Cost and delisting sensitivity: generated in every child run.
- Legacy baseline comparison: generated when the legacy metric file is available.

## Result
- Full child backtest run: `20260613T133520Z-backtest-11107a4e`
- CAGR: `0.2095`
- Sharpe: `1.1342`
- Max drawdown: `-0.3360`

## Promotion Decision
Eligible for human review. No automated strategy promotion is performed.