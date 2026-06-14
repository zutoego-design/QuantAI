# QuantAI 文档

更新日期：`2026-06-14`

## 当前定位

QuantAI 是 S&P 500 月频多因子研究系统。它提供 PIT 数据门禁、因子诊断、
账本回测、统一留出期组合评估、统计证据审查和可复现运行产物，不连接券商，
也不生成自动实盘订单。

旧 v0.3 canonical rule 实验
[`20260613T133519Z-experiment-7ad846cd`](../reports/runs/20260613T133519Z-experiment-7ad846cd/)
及其 full child 现在标记为 `legacy_reference`。其工程产物仍有效，但在重新通过
预注册、冻结快照和独立留出期验证前，不构成确认性研究证据。

新框架的首个真实确认性示例
[`20260614T011051Z-experiment-ec84fa0a`](../reports/runs/20260614T011051Z-experiment-ec84fa0a/)
已完成：产物状态为 `valid`，研究结论为 `rejected`。留出期净 Sharpe 为
`1.4321`，但计入三次实际尝试后的 Deflated Sharpe 概率仅为 `80.38%`，且
预注册因子没有通过方向与 FDR 联合门槛。该结果证明结论门禁已按协议工作，
不代表策略获得支持。汇总见
[留出期基线比较](../reports/research/holdout_baseline_comparison.md)。

## 活跃文档

1. [当前研究状态](./CURRENT_RESEARCH_STATUS.md)
2. [研究方法](./RESEARCH_METHODOLOGY.md)
3. [可信度改造计划](./RESEARCH_CREDIBILITY_PLAN.md)
4. [运行手册](./OPERATIONS.md)

## 当前能力

- 旧 YAML 继续运行，但自动归类为 `exploratory`。
- 确认性实验记录协议、spec hash、数据 snapshot ID 和试验次数。
- 确认性因子证据只使用留出期，且强制四类稳健性矩阵。
- 运行身份包含宏观、Fama-French、依赖环境和脏工作树补丁。
- Rule 与 ML 使用同一优化器、成本模型、账本和留出期指标。
- 研究证据包含 HAC/FDR、区块 Bootstrap、Deflated Sharpe 与
  Fama-French 5 因子加 Momentum 回归。
- 运行产物有效性为 `valid/invalid`；研究结论独立为
  `supported/inconclusive/rejected`。

## 已知限制

- 历史 S&P 500 成分来自 Wikipedia 变更记录重建，并非授权成分主数据。
- Yahoo/Stooq 不能完整证明退市现金流和公司行为；退市触发为零时只表示
  “未观察到”，不表示退市处理已验证。
- SEC 文本历史覆盖仍不足以支持文本策略。
- 第二股票池验证源当前仍为 `disabled`。
- 回测已明确使用执行日收盘成交，但尚未用成交级数据校准停牌、部分成交和冲击。
- Fama-French 官方数据存在发布时间滞后，确认性留出期必须具有至少 95% 覆盖。
- 当前范围仅为可信研究报告；paper trading、broker 和机构数据采购保留在
  [v0.3 归档](./archive/v0.3/README.md)。

## 历史归档

- [v0.3 状态与决策](./archive/v0.3/README.md)
- [v0.2 交付文档](./archive/v0.2/README.md)
