# Holdout Baseline Comparison

Generated from 2 canonical experiment runs.

| model_type | evaluation_scope | evidence_status | acceptance_status | net_total_return | net_sharpe | average_turnover | mean_rank_ic | text_coverage | bias_recommendation | recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rule_score | holdout | rejected | valid | 0.3369 | 1.4321 | 0.1027 |  |  | hold_for_further_testing | rejected |
| rule_score | legacy_reference | legacy_reference | valid | 6.2496 | 1.1342 | 0.0483 |  |  | eligible_for_human_review | legacy_reference |

## Decisions

- Highest net Sharpe: `rule_score` at `1.4321`.
- Rule-score holdout net Sharpe: `1.4321`.
- Rule robustness matrix complete: `false`.

Recommendations are evidence labels, not trading approvals.

## Sources

- `20260614T011051Z-experiment-ec84fa0a` -> `20260614T011053Z-backtest-01d3a9b7`
- `20260613T133519Z-experiment-7ad846cd` -> `20260613T133520Z-backtest-11107a4e`
