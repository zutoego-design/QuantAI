# ML Strategy Development

- Source run: `20260613T115415Z-backtest-01e41c77`
- Research stage: `historical_pseudo_out_of_sample`
- Validation conditions changed: `true`
- Official live/paper claim: `false`
- Development search algorithm: `bounded_bayesian_expected_improvement`
- Development search trials: `18`
- Final historical holdout candidates inspected: `1`

## Protocol

- Development window: `2016-01-01` to `2023-12-29`
- Historical holdout window: `2024-02-01` to `2026-06-11`
- Search family: `value_low_risk_style_neutral_lightgbm`
- Selection objective: `net_sharpe + 8*mean_rank_ic + 0.20*positive_ic_share - turnover_penalty`
- Search-space precommitment: `True`

## Bayesian Development Search

| trial | objective_score | net_sharpe | net_total_return | mean_rank_ic | positive_rank_ic_share | average_turnover | portfolio_top_n | train_periods |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 14 | 1.2321 | 0.8776 | 0.6716 | 0.0256 | 0.7500 | 0.4044 | 50 | 60 |
| 8 | 0.9958 | 0.7985 | 1.2163 | 0.0109 | 0.5500 | 0.4413 | 50 | 36 |
| 9 | 0.9628 | 0.6968 | 0.8128 | 0.0145 | 0.7500 | 0.4479 | 50 | 48 |
| 18 | 0.9360 | 0.7359 | 0.5709 | 0.0209 | 0.7500 | 0.5283 | 25 | 60 |
| 3 | 0.9316 | 0.8213 | 1.2063 | 0.0085 | 0.6500 | 0.5087 | 50 | 36 |
| 2 | 0.9180 | 0.7017 | 0.5050 | 0.0164 | 0.7500 | 0.4933 | 35 | 60 |
| 4 | 0.9176 | 0.7680 | 1.1315 | 0.0089 | 0.5500 | 0.4713 | 50 | 36 |
| 5 | 0.9139 | 0.6980 | 0.5223 | 0.0208 | 0.7500 | 0.5172 | 25 | 60 |
| 7 | 0.8999 | 0.8215 | 1.3549 | 0.0089 | 0.5500 | 0.5183 | 25 | 36 |
| 11 | 0.8837 | 0.7485 | 1.0930 | 0.0044 | 0.5000 | 0.4333 | 50 | 36 |

## Frozen Historical Holdout

| candidate | sharpe_ratio | net_total_return | max_drawdown | beta | target_num_holdings | max_sector_weight | ff_alpha_annualized | ff_alpha_t_stat | deflated_sharpe_probability | passes_daily_dsp_gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| frozen_bayesian_style_neutral_holdout | 1.1421 | 0.5525 | -0.2072 | 0.9747 | 50 | 0.1500 | 0.0207 | 0.4592 | 0.9567 | True |

## Holdout Model Fit

- Train rows: `40340`
- Test rows: `12746`
- Purged rows: `467`
- Mean holdout Rank IC: `0.03906915412146355`

## Audit

- Uses future unavailable data: `False`
- Uses closed v1 holdout as a clean confirmation: `False`
- Purge enabled: `True`
- Embargo days: `5`
- Development search counted separately from final holdout: `True`
- Output directory: `D:/QuantAI/reports/research/ml_strategy_development`

## Research Readout

- This is historical pseudo-out-of-sample evidence, not live forward validation.
- Bayesian optimization is treated as one bounded development search protocol.
- Only the frozen final candidate is counted as a final historical holdout inspection.
- Reusing the final historical holdout to replace the candidate would increment final trial count.
