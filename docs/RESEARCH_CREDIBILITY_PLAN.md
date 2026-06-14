# 研究可信度改造计划

更新日期：`2026-06-14`

## 目标

把系统从工程验收完整升级为可复算、可证伪的研究平台。当前范围只覆盖可信
研究报告，不包含 paper trading、broker 或实盘路由。

## 实施状态

- [x] 研究协议、开发期/留出期和标签间隔校验
- [x] 数据 SHA-256 快照与 manifest/registry 身份
- [x] 严格数据截止日交易日门禁
- [x] 确认性因子证据仅使用留出期
- [x] 宏观、Fama-French、依赖环境和脏工作树完整身份
- [x] 确认性四类稳健性矩阵强制门禁
- [x] 同一 spec 的快照一致性保护和试验次数记录
- [x] Rule/ML 共用优化器、成本和账本的留出期评估
- [x] HAC 因子检验与 FDR
- [x] 区块 Bootstrap 与 Deflated Sharpe
- [x] Fama-French 5 因子加 Momentum 回归
- [x] `valid/invalid` 与研究证据状态拆分
- [x] Top-N 单变量稳健性和配置 diff
- [x] 退市零触发显式标记
- [x] 完成一次真实确认性示例运行并登记结果
- [x] 依赖锁文件与 CI 自动复算

## 历史验收结果

以下结果产生于 P0 修复前，只保留为历史参考，不再满足当前确认性验收：

- 旧确认性示例：
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
- 旧方法学验收：
  [`20260614T011309Z-acceptance-863dd125`](../reports/runs/20260614T011309Z-acceptance-863dd125/)
  13/13 通过
- 比较报告：[留出期基线比较](../reports/research/holdout_baseline_comparison.md)

## 验收条件

- 同一数据生成相同 snapshot ID，文件变化生成不同 ID。
- 开发标签不能进入留出期，Rule 与 ML 只能比较相同留出区间。
- 所有统计结果可由保存的日收益、协议、快照和固定随机种子复算。
- 稳健性矩阵不存在 `skipped`，Top-N 只改变目标持仓数。
- 确认性矩阵必须覆盖成本、调仓日、持仓数和子区间。
- 运行必须来自 clean commit，或包含可复核的补丁、未跟踪文件和环境快照。
- 全量 pytest、Ruff 和 compileall 通过。

## 未完成的数据升级

- 授权历史成分、精确退市收益和公司行动账本。
- PIT 基本面修订历史与 ALFRED 宏观 vintage。
- SEC 文本覆盖至少 80%。
- 第二数据源交叉验证。
- 成交级点差、市场冲击、停牌和部分成交校准。
