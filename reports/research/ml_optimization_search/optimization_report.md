# ML Optimization Search

- Source run: `20260613T115415Z-backtest-01e41c77`
- Candidates evaluated: `59`
- Validation conditions changed: `false`
- Official confirmatory claim: `false`

## Top Candidates

| rank | candidate | net_sharpe | net_total_return | mean_rank_ic | positive_rank_ic_share | avg_turnover | shadow_dsp_trial1 | shadow_lower_95 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `value_low_risk__lgbm_steady__cross_sectional_rank__top25__roll60__blend0.0` | 1.1796 | 2.0007 | 0.0295 | 0.77 | 0.4653 | 0.9969 | 0.4617 |
| 2 | `value_low_risk_momentum3__lgbm_steady__cross_sectional_rank__top35__roll60__blend0.0` | 1.1645 | 2.1032 | 0.0293 | 0.68 | 0.4623 | 0.9958 | 0.4466 |
| 3 | `value_low_risk_momentum3__lgbm_steady__cross_sectional_rank__top30__roll60__blend0.0` | 1.1555 | 2.1220 | 0.0293 | 0.68 | 0.4750 | 0.9958 | 0.4377 |
| 4 | `value_low_risk__lgbm_steady__cross_sectional_rank__top30__roll60__blend0.0` | 1.1484 | 1.9759 | 0.0295 | 0.77 | 0.4391 | 0.9962 | 0.4306 |
| 5 | `value_low_risk__lgbm_steady__cross_sectional_rank__top35__roll60__blend0.0` | 1.1263 | 1.8971 | 0.0295 | 0.77 | 0.4243 | 0.9952 | 0.4085 |
| 6 | `value_low_risk_momentum3__lgbm_steady__cross_sectional_rank__top25__roll60__blend0.0` | 1.1029 | 1.9718 | 0.0293 | 0.68 | 0.4953 | 0.9953 | 0.3851 |
| 7 | `value_low_risk_momentum3__lgbm_ranker_gain100__cross_sectional_rank__top25__roll36__blend0.0` | 1.0267 | 3.9921 | 0.0213 | 0.53 | 0.5161 | 0.9978 | 0.4158 |
| 8 | `value_low_risk_momentum3__lgbm_steady__cross_sectional_rank__top40__roll36__blend0.0` | 1.0069 | 2.9533 | 0.0072 | 0.53 | 0.4724 | 0.9963 | 0.3961 |
| 9 | `value_low_risk_momentum3__lgbm_ranker_gain100__cross_sectional_rank__top30__roll36__blend0.0` | 1.0051 | 3.7148 | 0.0213 | 0.53 | 0.4894 | 0.9974 | 0.3942 |
| 10 | `value_low_risk_momentum3__lgbm_steady__cross_sectional_rank__top30__roll36__blend0.0` | 0.9866 | 3.0513 | 0.0072 | 0.53 | 0.4962 | 0.9965 | 0.3757 |
| 11 | `value_low_risk_momentum3__lgbm_steady__cross_sectional_rank__top35__roll36__blend0.0` | 0.9704 | 2.8119 | 0.0072 | 0.53 | 0.4898 | 0.9954 | 0.3596 |
| 12 | `value_low_risk_sales__lgbm_ranker_gain100__cross_sectional_rank__top25__roll36__blend0.0` | 0.9592 | 3.5906 | 0.0140 | 0.63 | 0.4834 | 0.9948 | 0.3483 |
| 13 | `value_low_risk_sales__lgbm_ranker_gain100__cross_sectional_rank__top25__roll60__blend0.0` | 0.9518 | 1.6564 | 0.0352 | 0.68 | 0.5122 | 0.9852 | 0.2339 |
| 14 | `value_low_risk_momentum3__lgbm_shallow__cross_sectional_rank__top30__roll36__blend0.0` | 0.9515 | 2.8819 | 0.0048 | 0.53 | 0.4777 | 0.9949 | 0.3406 |
| 15 | `value_low_risk_momentum3__lgbm_ranker_gain100__cross_sectional_rank__top35__roll36__blend0.0` | 0.9509 | 3.2281 | 0.0213 | 0.53 | 0.4758 | 0.9958 | 0.3400 |

## Best Candidate

- Candidate: `value_low_risk__lgbm_steady__cross_sectional_rank__top25__roll60__blend0.0`
- Repro spec: `experiments/ml_value_low_risk_lgbm_steady.yaml`
- Development walk-forward Sharpe: `1.1796`
- Development mean Rank IC: `0.0295`
- Search-adjusted Deflated Sharpe probability: `0.6364`

## Daily Portfolio Simulator Check

- Daily net total return: `1.2245`
- Daily Sharpe: `0.7414`
- Daily bootstrap Sharpe one-sided lower 95%: `0.1128`
- Daily Deflated Sharpe probability over `59` candidates: `0.2657`
- Required Deflated Sharpe probability: `0.9500`
- Max drawdown: `-0.3546`
- Beta: `1.2086`
- FF alpha annualized: `-0.0039`
- FF alpha t-stat: `-0.0893`

## Audit

- Label gap / purge check: `True`
- Label validation artifact reports `no_future_feature_leakage=True`.
- Training folds use purge with 5-day embargo; no fold has train label_end_time crossing into the test window.
- Data diagnostics show point-in-time universe membership and fundamentals coverage for the source run; no synthetic-input blocker was introduced by this search.
- Main residual concern: daily simulation is dominated by market/value style exposure; FF alpha is not significant.

## Verdict

- The optimized ML candidate improved over the previous ML baselines in development search.
- It did not pass the unchanged confirmatory standard: daily Deflated Sharpe probability remains below 0.95.
- Do not promote this ML strategy to confirmed alpha. Treat it as a candidate for a future preregistered holdout only after clean-git reproducibility and protocol registration.
