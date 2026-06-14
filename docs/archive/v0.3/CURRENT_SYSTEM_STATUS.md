# 当前系统状态

更新日期：`2026-06-13`

## 事实基线

当前 canonical 规则实验为
[`20260613T133519Z-experiment-7ad846cd`](../../../reports/runs/20260613T133519Z-experiment-7ad846cd/)；
其 full child 为
[`20260613T133520Z-backtest-11107a4e`](../../../reports/runs/20260613T133520Z-backtest-11107a4e/)。

| 项目 | 当前值 |
| --- | --- |
| full child 状态 | `valid` |
| 数据截止日 | `2026-06-11` |
| 策略 | `multifactor_balanced_us` |
| 股票池 | `sp500_historical_standard` |
| 模型 | canonical `rule_score` |
| Acceptance | `20260613T144054Z-acceptance-dafe14fb`，15/15 通过 |
| Bias review | `eligible_for_human_review` |
| Paper-trading candidate | 无策略获批 |
| 系统边界 | 研究与人工审批平台；无实盘路由 |

关键证据：

- [规则稳健性矩阵](../../../reports/runs/20260613T133519Z-experiment-7ad846cd/robustness_matrix.csv)
- [规则 bias review](../../../reports/runs/20260613T133520Z-backtest-11107a4e/bias_review.json)
- [Acceptance checks](../../../reports/runs/20260613T144054Z-acceptance-dafe14fb/acceptance_checks.csv)
- [Baseline comparison](../../../reports/research/v0.3_baseline_comparison.md)
- [Candidate selection](../../../reports/research/v0.3_candidate_selection.md)
- [十日运营汇总](../../../reports/operations/ten_day_summary.md)

## 已验证能力

### 规则研究基线

- 13 个生产因子均达到 80% 配置覆盖门槛。
- `debt_to_equity`、`gross_margin`、`operating_margin` 已从生产配置移除，
  决策记录见 [因子覆盖决策](./FACTOR_COVERAGE_DECISION.md)。
- 行业收益归因逐日和全区间均与组合及内部基准收益对账。
- 稳健性矩阵包含 base、两个 subperiod、Top-N `30/50/100` 和调仓日
  `-5/0/+5`，无 skipped 项。
- Top-N 子运行分别保持精确 30、50 和 100 只持仓。

规则 full child 指标：

| 指标 | 结果 |
| --- | ---: |
| CAGR | 20.95% |
| Sharpe | 1.1342 |
| 最大回撤 | -33.60% |
| 净总收益 | 624.96% |
| 平均换手 | 4.83% |

### 统一模型比较

同一数据区间和成本协议下已运行：

- Rule score：`20260613T133519Z-experiment-7ad846cd`
- Ridge：`20260613T112925Z-experiment-029b1c13`
- Elastic Net：`20260613T113548Z-experiment-2880563a`
- LightGBM：`20260613T115415Z-experiment-57b88ea5`
- SEC 文本规则：`20260613T144114Z-experiment-ebea7fb3`

Ridge、Elastic Net 和 LightGBM 的 acceptance 均通过，但净 Sharpe 均低于
规则参考，因此 candidate decision 为 `rejected`。

SEC 文本 PIT 过滤和缓存链路已运行，但十年样本覆盖仅 `0.25%`。文本 full
child 的 acceptance 因 `risk_disclosure_score` 覆盖不足而无效，状态为
`needs_more_data`。

### 审批与运营

- Monthly rebalance 已修复精确持仓基数，生成两份独立 50 只候选包。
- `20260613T145940Z-rebalance-6944e389` 已转为
  `approved_for_candidate` 并生成 approved weights。
- `20260613T150002Z-rebalance-152496dd` 已转为 `rejected`，没有 approved
  export；终态再次转换被拒绝。
- `2026-05-29` 至 `2026-06-11` 连续十个交易日 risk dry run 最终 10/10
  有效、0 告警、10 个独立 registry run。
- 每日日志保留首次目录失败、修复后的浮点假阳性以及最终成功尝试，没有覆盖历史
  run。

## 当前边界与未完成项

数据缺口分级见
[v0.3 数据缺口评估](./v0.3_DATA_GAP_ASSESSMENT.md)。Wikipedia 历史成分、
退市与公司行为仍需人工警示；FRED vintage 和机构级 PIT 数据属于后续升级；
SEC 文本覆盖阻塞文本策略进入 paper trading。

当前没有策略满足全部 candidate 准入条件，详见
[candidate selection](../../../reports/research/v0.3_candidate_selection.md)。

系统当前不是自动实盘交易系统。尚未实现的 P3 项包括机构级数据、外部生产调度器、
broker paper adapter、持仓对账和任何实盘订单路由。
