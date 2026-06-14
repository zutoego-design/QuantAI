# v0.3 验收与后续待办

更新日期：`2026-06-13`

P0-P2 已按运行产物完成验收。当前待办只包含明确延后的 P3 项，不将其计入
v0.3 完成标准。

## P0：基线验收

- [x] P0-1 acceptance-check
  - 证据：`20260613T144054Z-acceptance-dafe14fb`
  - 结果：规则 full child 15/15 检查通过。
- [x] P0-2 低覆盖基本面因子处理
  - 证据：`FACTOR_COVERAGE_DECISION.md`
  - 结果：三个弱覆盖因子移出生产配置，保留 13 个因子均通过 80% 门槛。
- [x] P0-3 稳健性矩阵
  - 证据：`20260613T133519Z-experiment-7ad846cd/robustness_matrix.csv`
  - 结果：base、两个 subperiod、Top-N `30/50/100`、平移 `-5/+5` 全部完成，
    0 skipped；base 代表平移 0。
- [x] P0-4 行业收益归因
  - 证据：规则 full child 的 `sector_return_attribution.csv` 和 summary。
  - 结果：组合和内部基准的逐日及 linked contribution 全部对账。

## P1：能力验证

- [x] P1-1 四个 canonical baseline
  - Rule、Ridge、Elastic Net、LightGBM 已按统一协议运行。
  - 三个 ML child 均有非空 `ml_evaluation/`，acceptance 均为 valid。
- [x] P1-2 SEC 文本因子
  - 证据：`20260613T144114Z-experiment-ebea7fb3`
  - 结果：PIT 文本链路有效；覆盖 `0.25%`，按规则标记 `needs_more_data`，
    未把缺失文本当作中性信号。
- [x] P1-3 registry 与 baseline comparison
  - 产物：`reports/research/v0.3_baseline_comparison.md`
  - 结果：规则为 reference；Ridge、Elastic Net、LightGBM 均 rejected；
    文本策略 needs more data。
- [x] P1-4 approve/reject 双路径
  - Approve：`20260613T145940Z-rebalance-6944e389`
  - Reject：`20260613T150002Z-rebalance-152496dd`
  - 两个终态均不可再次变更，registry 状态一致。

## P2：运营与 paper-trading 准备

- [x] P2-1 连续十个交易日 operational dry run
  - 窗口：`2026-05-29` 至 `2026-06-11`
  - 结果：10/10 有效，0 最终告警，10 个独立 risk run。
  - 产物：`reports/operations/daily_log/`、`ten_day_summary.csv` 和
    `ten_day_summary.md`。
- [x] P2-2 数据缺口评估
  - 产物：`v0.3_DATA_GAP_ASSESSMENT.md`
  - 结果：历史成分、退市、公司行为、PIT 基本面、SEC 文本和 FRED vintage
    均已按统一状态分级。
- [x] P2-3 candidate 准入
  - 产物：`reports/research/v0.3_candidate_selection.md`
  - 结果：本轮没有策略获批；LightGBM 和 SEC 文本策略保留为 rejected
    examples。

## P3：延后事项

以下事项不纳入 v0.3 完成标准。

### [ ] P3-1 机构级 PIT、退市与公司行为数据

- 证据：`v0.3_DATA_GAP_ASSESSMENT.md`
- 执行动作：采购并接入带有效日期、稳定标识符和修订历史的数据。
- 验收标准：可对历史成分、退市现金流、公司行为和 PIT 基本面逐项对账。
- 预期产物：机构数据适配器、数据合同、迁移与回归报告。

### [ ] P3-2 FRED vintage

- 证据：当前宏观数据没有 vintage 重建。
- 执行动作：接入 ALFRED 或等价 vintage 数据。
- 验收标准：任一历史信号只读取当时已发布的宏观版本。
- 预期产物：vintage-aware 宏观表和泄漏回归测试。

### [ ] P3-3 外部生产调度器

- 证据：当前为 CLI 与 documented job runner。
- 执行动作：部署 Prefect 或等价调度器，配置重试、告警和凭据隔离。
- 验收标准：月度与日度任务在独立环境连续运行并可审计。
- 预期产物：部署清单、flow 定义、运行手册。

### [ ] P3-4 Broker paper adapter 与对账

- 证据：当前仅生成 approved target weights，不连接 broker。
- 执行动作：设计 sandbox adapter、订单生命周期、持仓对账和漂移检查。
- 验收标准：仅 paper 环境可用，所有订单需人工批准且可回放。
- 预期产物：adapter、contract tests、paper reconciliation 报告。

### [ ] P3-5 实盘路由

- 证据：当前无自动下单路径。
- 执行动作：本阶段不实施。
- 验收标准：需单独治理、风控和法律审批，不得由 v0.3 状态推导。
- 预期产物：无。

## v0.3 完成定义

- [x] P0 全部完成，最新 canonical rule acceptance 通过。
- [x] 四个 canonical baseline 与 SEC 文本实验完成统一比较。
- [x] approve/reject 双路径完成。
- [x] 连续十个交易日 dry run 完成。
- [x] 数据缺口和 candidate selection 文档完成。
- [x] 保留 rejected examples。
- [x] 没有引入自动实盘交易路径。
