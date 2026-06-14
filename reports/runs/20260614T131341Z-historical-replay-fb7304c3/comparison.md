# v2 Historical Replay

- Evaluation scope: `exploratory_historical_replay`
- Decision: `challenger_only`
- Selected strategy: `none`
- Challenger: `v1_drawdown_fixed`

| candidate_id | valid_folds | positive_years | spy_outperformance_years | median_annual_sharpe | combined_net_total_return | combined_sharpe | combined_max_drawdown | combined_average_turnover | selection_passed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v1_control | 7 | 6 | 6 | 2.244767443208116 | 3.4459392336292494 | 1.1183031757781552 | -0.3763717327518219 | 0.03862391981775981 | False |
| v1_drawdown_fixed | 7 | 6 | 6 | 2.1627726826733227 | 3.427119000688677 | 1.1248712292612097 | -0.3699800571560353 | 0.03798686366936116 | False |
| v2_core | 6 | 6 | 3 | 1.728650076963107 | 3.1293558581098786 | 1.3358922182419504 | -0.3324042903545352 | 0.08674940232114542 | False |

These results are exploratory historical replay evidence. They do not replace the frozen confirmatory v1 decision.
