# Research Memo

## Hypothesis
The preregistered multifactor rule score has positive out-of-sample risk-adjusted value.

## Data Gate
Passed all configured point-in-time and coverage checks.

## Workflow
- Single-factor diagnostics: confirmatory evidence uses holdout-only diagnostics.
- Portfolio backtest: completed.
- Subperiod checks: completed when the sample spans at least two years.
- Cost and delisting sensitivity: generated in every child run.
- Legacy baseline comparison: generated when the legacy metric file is available.

- Research stage: `confirmatory`
- Trial family: `multifactor-rule-confirmatory`
- Trial number: `5`

## Result
- Full child backtest run: `20260615T132129Z-backtest-a3307fe2`
- CAGR: `0.2282`
- Sharpe: `1.4321`
- Max drawdown: `-0.1386`

## Research Evidence Decision
- `rejected`
- Artifact validity and research evidence are separate decisions.