# Research Memo

## Hypothesis
Ridge improves cost-aware out-of-sample ranking over the rule-score baseline.

## Data Gate
Passed all configured point-in-time and coverage checks.

## Workflow
- Single-factor diagnostics: generated in the full child run.
- Portfolio backtest: completed.
- Subperiod checks: completed when the sample spans at least two years.
- Cost and delisting sensitivity: generated in every child run.
- Legacy baseline comparison: generated when the legacy metric file is available.

## Result
- Full child backtest run: `20260613T112925Z-backtest-6f94ed26`
- CAGR: `0.2095`
- Sharpe: `1.1342`
- Max drawdown: `-0.3360`

## Promotion Decision
Eligible for human review. No automated strategy promotion is performed.