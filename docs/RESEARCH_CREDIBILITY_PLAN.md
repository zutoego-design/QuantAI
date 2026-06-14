# 研究可信度改造计划

更新日期：`2026-06-14`

## 目标

把系统从工程验收完整升级为可复算、可证伪的研究平台。当前范围只覆盖可信
研究报告，不包含 paper trading、broker 或实盘路由。

## 实施状态

- [x] 研究协议、开发期/留出期和标签间隔校验
- [x] 数据 SHA-256 快照与 manifest/registry 身份
- [x] 同一 spec 的快照一致性保护和试验次数记录
- [x] Rule/ML 共用优化器、成本和账本的留出期评估
- [x] HAC 因子检验与 FDR
- [x] 区块 Bootstrap 与 Deflated Sharpe
- [x] Fama-French 5 因子加 Momentum 回归
- [x] `valid/invalid` 与研究证据状态拆分
- [x] Top-N 单变量稳健性和配置 diff
- [x] 退市零触发显式标记
- [x] 完成一次真实确认性示例运行并登记结果

## 验收结果

- 确认性示例：
  [`20260614T011051Z-experiment-ec84fa0a`](../reports/runs/20260614T011051Z-experiment-ec84fa0a/)
- 产物状态：`valid`
- 研究结论：`rejected`
- 留出期：`2024-08-01` 至 `2025-12-31`
- 留出期净 Sharpe：`1.4321`
- 主指标单侧 95% Bootstrap 下界：`0.3070`
- 实际试验次数：`3`
- Deflated Sharpe 概率：`80.38%`
- 风格因子覆盖率：`100%`
- 阻断项：Deflated Sharpe 低于 95%，且预注册因子未通过方向与 FDR 联合检查
- 方法学验收：
  [`20260614T011309Z-acceptance-863dd125`](../reports/runs/20260614T011309Z-acceptance-863dd125/)
  13/13 通过
- 比较报告：[留出期基线比较](../reports/research/holdout_baseline_comparison.md)

## 验收条件

- 同一数据生成相同 snapshot ID，文件变化生成不同 ID。
- 开发标签不能进入留出期，Rule 与 ML 只能比较相同留出区间。
- 所有统计结果可由保存的日收益、协议、快照和固定随机种子复算。
- 稳健性矩阵不存在 `skipped`，Top-N 只改变目标持仓数。
- 活跃 `docs/` 根目录只包含 README、研究方法、计划和运行手册。
- 全量 pytest、Ruff 和 compileall 通过。
