# 研究方法

## 研究分级

实验分为：

- `exploratory`：用于提出假设、调试因子和选择参数，不得支持确认性结论。
- `confirmatory`：必须预注册开发期、独立留出期、主指标、阈值、原假设、
  试验族和因子方向。

旧格式实验自动归类为 `exploratory`。运行产物是否完整与研究结论是否成立
分别判断。

## 预注册与冻结

确认性协议必须满足：

- 开发期早于留出期。
- 两个区间之间至少间隔一个标签周期，当前默认是 21 个交易日。
- 同一 spec hash 再次运行时必须使用相同 data snapshot ID；数据变化后应建立
  新研究或修改协议，不能冒充原运行。
- 同一 `trial_family` 的参数修改和重复试验都会增加试验次数，并进入
  Deflated Sharpe 修正。

运行会对实际读取的价格、基本面、股票池、SEC 事件、宏观和 Fama-French
文件计算 SHA-256，并把 Python 版本和已安装包版本写入
`data_snapshot.json`。每个 run 还保存：

- `environment.json`：完整依赖环境。
- `workspace_identity.json`：Git commit、clean/dirty 状态和工作区哈希。
- `code.patch`：存在已跟踪修改时保存完整 binary diff。
- `untracked_files/`：存在未跟踪研究代码或配置时保存原文件副本。

每个输入文件还按 SHA-256 归档到 `data/archive/research_inputs/`。同一文件只
归档一次，并优先使用硬链接降低磁盘占用；验收会复核归档文件大小和哈希。

## PIT 与标签

- 股票池使用信号日可见的历史成分快照。
- 基本面按 SEC filing/available date 独立选择，不回填未来修订。
- 缺失因子保持缺失，不使用未来值、0 或横截面中位数伪造覆盖率。
- 标签结束时间必须早于留出期及 embargo 边界。
- ML 只用开发期内且标签已结束的样本训练，只对留出期生成预测。
- 确认性因子 IC、HAC、FDR、分位数组合和衰减诊断只接收留出期因子行；
  价格输入不得晚于留出期结束日。

## 统一组合评估

Rule score 和 ML 预测都转换为日期化截面分数，并经过同一流程：

1. 历史协方差估计。
2. 精确目标持仓数优化。
3. 原有行业、单股、换手和 ADV 约束。
4. 相同执行滞后、执行日收盘成交、佣金、滑点和市场冲击。
5. 同一持仓/现金账本及基准收益。

`backtest.execution_price` 当前只允许 `close`。这是对现有账本行为的明确冻结，
不是 next-open 或 VWAP 模拟。现金利息由
`backtest.cash_interest_annual_rate` 显式配置。

Baseline comparison 只使用留出期净指标。旧全样本结果显示为
`legacy_reference`。

## 统计证据

- 因子 IC 使用 Newey-West/HAC 标准误。
- 同一实验内的因子 p-value 使用 Benjamini-Hochberg FDR 修正。
- 主指标、Sharpe、alpha 和最大回撤使用 21 日循环区块、2,000 次固定种子
  Bootstrap。
- Deflated Sharpe 使用 registry 中同一试验族的实际试验次数。
- 风格归因使用官方 Fama-French 5 因子与 Momentum 日频数据，alpha 标准误
  使用 HAC。

因子覆盖率只是数据质量条件，不能替代方向、IC 和 FDR 证据。

## 结论规则

`supported` 必须同时满足：

- 主指标单侧 95% Bootstrap 下界超过预注册阈值。
- Deflated Sharpe 概率不低于 95%。
- 留出期净总收益为正。
- 预注册因子通过方向和 FDR 检查。
- 风格回归可用且覆盖至少 95% 留出期交易日。
- 不存在其他方法学阻断项。

净收益非正或存在阻断项时为 `rejected`；证据不足但没有硬性阻断时为
`inconclusive`。

## 稳健性

- 确认性实验必须同时注册子区间、成本、Top-N 和调仓日偏移测试。
- 成本至少三个不同场景，Top-N 至少两个目标数，调仓日至少两个非零偏移。
- Top-N 只改变目标持仓数，不关闭优化器、不放宽换手、不切换等权。
- 每个子运行保存配置 diff；出现未允许的字段变化即判为无效。
- 无法满足精确持仓数时测试为 `invalid`，不得跳过。
- 退市敏感性必须报告实际触发次数；零触发标记为 `not_observed`。
