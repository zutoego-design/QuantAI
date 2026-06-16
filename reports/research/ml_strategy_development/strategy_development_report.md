# ML Strategy Development

- Source run: `20260613T115415Z-backtest-01e41c77`
- Research stage: `exploratory`
- Validation conditions changed: `false`
- Official confirmatory claim: `false`
- New exploratory candidates counted: `11`
- Cumulative exploratory trial count used for DSP: `70`

## Walk-Forward Label Experiments

| candidate | mean_rank_ic | positive_rank_ic_share | net_sharpe | net_total_return | average_turnover |
| --- | --- | --- | --- | --- | --- |
| style_residual_rank_lgbm | 0.0208 | 0.7727 | 0.7559 | 0.8622 | 0.5353 |
| sector_relative_rank_lgbm | 0.0314 | 0.7727 | 0.9796 | 1.7413 | 0.4359 |

## Daily Portfolio Simulations

| candidate | sharpe_ratio | net_total_return | max_drawdown | beta | target_num_holdings | max_sector_weight | ff_alpha_annualized | ff_alpha_t_stat | deflated_sharpe_probability | passes_daily_dsp_gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current_best__style_neutral_score__sector_cap_15 | 1.0413 | 1.5391 | -0.2372 | 0.9721 | 25.0000 | 0.1500 | 0.0405 | 1.3318 | 0.5038 | False |
| style_residual_rank_lgbm__style_neutral_score | 0.9448 | 1.1816 | -0.2324 | 0.9324 | 25.0000 | 0.2500 | 0.0303 | 1.1561 | 0.4335 | False |
| style_residual_rank_lgbm__style_neutral_score__holdings35_sector_cap20 | 0.9406 | 1.3466 | -0.2724 | 1.0462 | 35.0000 | 0.2000 | 0.0274 | 1.0717 | 0.4225 | False |
| current_best__style_neutral_score__sector_cap_20 | 0.9350 | 1.4269 | -0.2688 | 1.0454 | 25.0000 | 0.2000 | 0.0318 | 0.9124 | 0.4088 | False |
| style_residual_rank_lgbm__style_neutral_score__sector_cap_20 | 0.9277 | 1.2540 | -0.2634 | 1.0012 | 25.0000 | 0.2000 | 0.0176 | 0.6722 | 0.4156 | False |
| current_best__style_neutral_score__holdings35_sector_cap20 | 0.8908 | 1.2374 | -0.2606 | 1.0144 | 35.0000 | 0.2000 | 0.0255 | 0.8124 | 0.3656 | False |
| current_best__style_neutral_score | 0.8658 | 1.2618 | -0.2805 | 1.0732 | 25.0000 | 0.2500 | 0.0284 | 0.8578 | 0.3500 | False |
| style_residual_rank_lgbm__raw_score | 0.8652 | 1.0727 | -0.2739 | 0.9585 | 25.0000 | 0.2500 | 0.0216 | 0.7800 | 0.3651 | False |
| sector_relative_rank_lgbm__style_neutral_score | 0.8570 | 1.0832 | -0.2469 | 0.9413 | 25.0000 | 0.2500 | 0.0257 | 0.8045 | 0.3449 | False |
| style_residual_rank_lgbm__style_neutral_score__sector_cap_15 | 0.8544 | 1.1036 | -0.2467 | 0.9904 | 25.0000 | 0.1500 | 0.0121 | 0.4361 | 0.3488 | False |
| sector_relative_rank_lgbm__raw_score | 0.7543 | 1.1576 | -0.3336 | 1.0918 | 25.0000 | 0.2500 | 0.0031 | 0.0722 | 0.2641 | False |

## Audit

- Uses closed v1 holdout as confirmation: `False`
- Official confirmatory claim: `False`
- Purge enabled: `True`
- Embargo days: `5`
- Label leakage audit: `{'style_residual_rank_lgbm': {'passed': True, 'rule': 'max train label_end_time < test_start - embargo_days for every fold'}, 'sector_relative_rank_lgbm': {'passed': True, 'rule': 'max train label_end_time < test_start - embargo_days for every fold'}}`
- Output directory: `D:/QuantAI/reports/research/ml_strategy_development`

## Research Readout

- These experiments are only development evidence.
- Passing the daily DSP gate here would still not be a confirmation because the strategy was selected after exploratory search.
- A strategy can move to v2 only after clean-git reproducibility and a fresh preregistered forward holdout.
