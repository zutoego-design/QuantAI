# AI辅助美股量化研究平台：改进版系统建设报告

基于机器学习量化研究报告与前序系统方案的工程化修订版

日期：2026-06-12

---

## 执行摘要

这份改进报告的目标不是重新写一份机器学习量化文献综述，而是把已有研究结论转化为可执行的系统建设方案。结论很明确：系统第一阶段不应追求复杂模型，而应优先建立点时一致、成本敏感、实验可复现、Agent 可审计的量化研究底座。

原研究报告对系统建设最有价值的启示有五点：第一，美股机器学习量化的关键风险来自数据时间错配、幸存者偏差、样本设计和交易成本，而不是模型规模不足；第二，线性模型和树模型应作为 ML 基线，Transformer/GNN/RL/LLM 应作为后续分层增强；第三，LLM 更适合作为文本语义因子抽取与研究自动化工具，而不是直接替代预测模型；第四，RL 更适合仓位、再平衡和执行层，而不是万能 alpha 挖掘器；第五，研究系统必须保存设计选择、数据版本、特征快照、标签定义、成本模型和回测引擎版本。

因此，建议将当前系统定位为 ML-ready、PIT-aware、cost-aware、agent-auditable 的美股量化研究平台。第一版仍然保持轻量：价格数据、基础因子、月频调仓、每日风险监控、实验记录、AI 报告与审查。但底层 schema、目录结构和工具边界必须为 PIT 数据、SEC 文本、树模型、walk-forward、purged validation、组合优化和未来 Agent 工作流预留接口。

## 一、从原报告提炼出的系统设计原则

原报告对美股机器学习量化的核心判断可以概括为：研究系统的质量优先于模型复杂度。对本系统而言，这意味着 Agent 不应被设计成“自动发现赚钱策略”的黑箱，而应被设计成规范化研究流程的编排者、审查者和报告生成者。

系统设计原则如下：

- PIT-first：所有基本面、公告、新闻和事件数据必须保留 as-of 时间，不允许默认使用未来修订数据。
- Baseline-first：先建立规则因子、线性模型和树模型基线，再逐步加入 Transformer、GNN、LLM 与 RL。
- Cost-aware：回测结果必须同时报告 gross 与 net，成本、滑点、换手、容量和借券假设必须参数化。
- Experiment-driven：每次研究必须保存完整配置、数据版本、特征版本、标签定义、模型参数、回测规则和报告。
- Agent-as-orchestrator：Agent 只负责生成配置、调用工具、审查结果和生成报告，真实计算由确定性 Python 工具完成。
- Human-in-the-loop：新因子入库、核心回测逻辑修改、策略批准、月度调仓权重导出和实盘交易接口必须人工审批。

这些原则直接决定了系统架构：它不是一个 notebook 集合，也不是一个全自动交易机器人，而是一个可审计、可复现、可扩展的研究操作系统。

## 二、目标系统定位

建议将项目正式定义为：AI 辅助的美股量化研究工作流平台。

系统第一阶段的目标是服务研究，而不是实盘交易；服务月频/周频横截面选股，而不是高频撮合；服务可解释、可复现的研究过程，而不是追逐单次最优回测收益。

目标能力分为四层：

- 研究层：将自然语言研究想法转化为结构化 strategy spec。
- 计算层：通过确定性工具完成数据加载、因子计算、回测、优化和风险评估。
- 审计层：自动检查数据质量、特征泄漏、标签重叠、交易成本、换手率和样本外稳健性。
- 报告层：生成研究报告、日报、月度调仓报告和策略批准材料。

非目标能力包括：第一阶段不做自动交易、不做高频策略、不做复杂 RL 环境、不做无审查的 Agent 自主改代码、不做未经验证的数据源混用。

## 三、改进后的总体架构

建议采用“确定性量化工具 + Agent 编排层 + 实验审计层”的架构。

总体流程如下：

用户研究请求 -> Orchestrator Agent -> Research Spec Agent -> Data QA Tool/Agent -> Feature & Label Tools -> Backtest Tool -> Cost/Risk Tool -> Critic Agent -> Report Agent -> Human Approval -> Experiment Registry

该架构有三个关键边界：

- Agent 可以生成配置、解释结果、检查风险，但不能绕过工具直接计算绩效。
- Backtest、optimizer、risk metrics、data validation 必须是可测试的 Python 函数。
- 所有写入型动作必须进入实验目录或审批目录，不能覆盖原始数据和历史实验。

建议的核心模块包括：data、features、labels、models、backtest、portfolio、agents、reports、experiments、tests。

## 四、数据层改进：PIT-aware Data Lake

数据层是本系统最重要的底座。即使第一版使用免费数据，也应按未来可替换为 CRSP、Compustat、Sharadar、SEC EDGAR、新闻数据源的方式设计接口。

数据层建议分为 raw、normalized、point_in_time、features 四层：

- raw：保留供应商原始返回，不做破坏性清洗。
- normalized：统一字段名、交易日历、ticker 映射、公司行为和价格格式。
- point_in_time：保存 as-of 可得视角，尤其适用于财报、公告、新闻和指数成分。
- features：保存每次特征计算快照，不直接覆盖。

核心数据表必须预留以下字段：symbol、security_id、cik、vendor、source、event_date、report_date、filing_datetime、effective_datetime、as_of_datetime、revision_datetime、ingestion_datetime、source_version、is_point_in_time。

第一版可使用 yfinance 做价格研究原型，但必须在 data_quality_report 中明确标注：价格数据来源、复权方式、退市处理限制、当前成分股导致的幸存者偏差风险。

## 五、Feature Store 与 Factor Metadata

因子不能只是函数。每个因子都应该有 metadata、依赖字段、时间可得性约束、适用持有期、预期换手、泄漏检查和单元测试。

建议将因子分为六类：价格动量/反转、波动率/流动性、基本面/质量、估值、事件/公司行为、文本/语义。第一版只实现价格动量、低波动、简单质量和简单价值代理，但保留扩展接口。

每个因子 metadata 至少包括：

- name、category、description、inputs、lookback_days、skip_days、higher_is_better。
- point_in_time_requirements：是否依赖 filing date、news timestamp、revision timestamp。
- preprocessing：winsorization、standardization、neutralization。
- expected_horizon：适合 5 日、20 日、60 日还是更长持有期。
- cost_sensitivity：高换手、中换手或低换手。
- leakage_checks：look-ahead、survivorship、timestamp alignment。

Feature Store 应支持按 run_id、as_of_date、universe_id、factor_version 查询，以便回测结果可复现。

## 六、标签与目标定义层

机器学习量化中，标签设计常常比模型更重要。系统应把标签层独立出来，避免在模型代码或 notebook 中临时生成 forward return。

建议第一阶段支持三类标签：

- forward_return：未来 N 个交易日收益，用于回归或排序。
- cross_sectional_rank：未来收益横截面分位数，用于 ranker 或分类。
- event_window_return：事件后窗口收益，未来用于 SEC filing、8-K、新闻事件。

第二阶段再增加 triple-barrier、meta-labeling 和 trend-scanning。

标签必须记录 horizon、label_start_time、label_end_time、overlap、purge_required、embargo_days。只要标签窗口重叠，就不能使用普通随机 K-fold。

## 七、模型路线：先基线，后复杂模型

改进后的模型路线应遵循：规则因子 -> 线性模型 -> 树模型 -> 文本因子 -> Transformer/GNN -> RL overlay。

第一阶段建议只做：规则打分、Ridge/Lasso/Logistic、LightGBM/XGBoost。目标是建立稳定、可解释、净成本后仍有意义的横截面排序基线。

第二阶段将 SEC 10-K/10-Q/8-K 和新闻数据转化为文本语义因子。LLM 的角色不是直接给出买卖建议，而是生成可缓存、可审计、可回测的 embedding、tone score、risk disclosure score、event type 和主题强度。

第三阶段才考虑 Transformer 和 GNN。Transformer 用于多尺度序列与市场状态建模，GNN 用于行业、相关性、供应链、新闻共振和 peer effect。

RL 应放在 allocation、rebalance timing、execution 和 risk overlay，而不是第一阶段 alpha 挖掘。

## 八、回测与评估框架改进

回测层必须从一开始支持成本敏感和稳健性测试。第一版可以是向量化月频回测，但接口必须预留事件驱动、多引擎复核和滑点模型。

基础回测流程：每个调仓日加载 as-of 数据 -> 计算因子和标签不可见性检查 -> 生成 score -> 选股 -> 权重构造或优化 -> 下期持有 -> 计算 gross/net return -> 记录交易、持仓、成本和风险。

建议内置以下稳健性测试：

- subperiod test：按市场阶段拆分，如 2015-2019、2020-2022、2023-至今。
- cost sensitivity：5bps、10bps、25bps、50bps 成本压力测试。
- rebalance day shift：月末、月初、第 5 个交易日调仓对比。
- top_n sensitivity：Top 30、Top 50、Top 100 对比。
- factor weight sensitivity：因子权重扰动后表现是否稳定。
- universe sensitivity：NASDAQ 100、S&P 500、大市值流动性池对比。

评估指标不能只看 Sharpe。必须同时保存 annual_return、volatility、Sharpe、Sortino、max_drawdown、Calmar、turnover、transaction_cost_paid、hit_rate、information_ratio、sector exposure、beta、capacity proxy、IC、ICIR、t-stat、Deflated Sharpe 和 PBO 的预留字段。

## 九、Agent 工作流改进

Agent 层应从“多 Agent 炫技”改为“受控编排 + 审计闭环”。第一版只需要 Orchestrator Agent、Research Spec Agent、Data QA Agent、Critic Agent 和 Report Agent。

推荐工作流如下：

- 1. Orchestrator Agent 判断任务类型：strategy_research、factor_development、daily_monitoring、monthly_rebalance、report_generation。
- 2. Research Spec Agent 将自然语言请求转换为 strategy_config.yaml。
- 3. Data QA Agent 调用数据校验工具，阻止不合格数据进入回测。
- 4. Backtest Tool 执行确定性回测，输出 metrics、holdings、trades、equity_curve。
- 5. Cost/Risk Tool 计算成本、换手、回撤、风格/行业暴露。
- 6. Critic Agent 专门寻找伪 alpha、数据泄漏、过拟合、行业集中、容量不足和成本低估。
- 7. Report Agent 生成研究报告和后续测试建议。
- 8. Human Approval 决定是否进入候选策略库。

Agent 的禁止事项：不得自行计算收益，不得直接修改核心回测引擎，不得覆盖历史实验，不得自动批准策略，不得导出实盘订单。

## 十、实验审计与可复现设计

实验系统应保存的不只是结果，而是完整研究上下文。每个 run_id 都应保存配置、输入、输出、环境和审查记录。

建议实验目录结构：

```text
experiments/runs/{run_id}/
  strategy_config.yaml
  research_plan.md
  data_quality_report.json
  feature_snapshot.parquet
  label_config.yaml
  model_config.yaml
  backtest_config.yaml
  performance_metrics.json
  holdings.parquet
  trades.parquet
  equity_curve.parquet
  bias_review.md
  final_report.md
```

experiment registry 应至少支持按 strategy_id、universe_id、factor_set、model_type、date_range、validation_method、net_sharpe、max_drawdown、turnover 查询。

所有报告中必须区分事实结果、模型解释和 Agent 推断。Agent 推断必须可追溯到工具输出。

## 十一、组合构建与风险控制改进

组合层建议先从简单约束开始，不要一开始引入复杂均值方差或黑箱优化。第一阶段可使用 equal weight、score weight、risk-scaled weight 和带约束的 score maximization。

基础约束建议：long_only=true、fully_invested=true、max_single_name_weight=5%、max_sector_weight=30%、max_turnover=40%、minimum_liquidity、transaction_cost_bps 参数化。

风险监控应每日运行，但不自动交易。日报应报告组合收益、benchmark 对比、回撤、波动率、行业暴露、单名贡献、异常波动和是否触发人工检查。

月度调仓流程必须包含审批：计算候选权重 -> 运行风险检查 -> 生成调仓报告 -> 人工确认 -> 导出 target_weights.csv。

## 十二、推荐技术栈与目录结构

推荐继续采用轻量技术栈：Python、pandas/Polars、DuckDB、Parquet、Pydantic、YAML/TOML、scipy/cvxpy、Prefect、pytest、Markdown/HTML 报告。

改进后的目录结构建议：

```text
quant-ai-research-platform/
  config/
    data_sources.yaml
    universe.yaml
    research_protocol.yaml
    cost_model.yaml
    validation.yaml
  data/
    raw/
    normalized/
    point_in_time/
    features/
    labels/
  src/
    data/
    features/
    labels/
    models/
    backtest/
    portfolio/
    agents/
    reports/
    workflows/
  experiments/
    runs/
    registry.duckdb
  tests/
  docs/
```

第一版只需要实现最少可运行链路，但每个目录的接口要为未来扩展预留。

## 十三、实施路线图

建议按四个阶段实施。每个阶段都必须产出可运行代码、测试和报告，而不是只产出 notebook。

阶段 1：研究底座 MVP。实现价格数据缓存、股票池、基础因子、月频回测、实验记录和基础报告。验收标准是一个策略可以从 config 跑完，并生成可复现结果。

阶段 2：ML-ready 改造。增加 labels 模块、walk-forward、线性/树模型、feature metadata、Data QA Agent、Critic Agent。验收标准是可以训练并回测一个 LightGBM 横截面排序基线。

阶段 3：文本与事件因子。接入 SEC filing metadata、10-K/10-Q/8-K 文本处理、embedding 缓存、事件窗口标签和文本增量 IC 测试。验收标准是文本因子可以独立做 event study 并进入组合回测。

阶段 4：准生产研究工作流。增加每日风险监控、月度调仓报告、审批目录、Prefect 调度、实验 registry 查询和稳健性测试矩阵。验收标准是系统可以支持持续研究和 paper trading 前的审计。

## 十四、MVP 任务清单

如果立即交给 Agent/Codex 开发，建议先拆成以下任务：

- 创建项目骨架和配置系统：pyproject、src、config、tests、experiments。
- 实现 data loader：yfinance price loader、trading calendar、Parquet cache、data quality report。
- 实现 universe loader：NASDAQ 100 或 S&P 500 当前成分作为 MVP，并标注幸存者偏差。
- 实现 factor registry：momentum_12_1、volatility_12m、simple_value_proxy、simple_quality_proxy。
- 实现 monthly backtester：top_n、equal weight、transaction cost、benchmark SPY。
- 实现 metrics：return、volatility、Sharpe、drawdown、turnover、hit rate、benchmark comparison。
- 实现 experiment logger：保存 config、metrics、holdings、trades、equity_curve、report。
- 实现 Agent 最小闭环：Research Spec -> Backtest Tool -> Critic -> Report。
- 实现 pytest：数据完整性、因子无未来函数、回测现金和权重一致性、报告生成。

这组任务完成后，系统已经具备扩展 ML、文本和优化器的基础。

## 十五、验收标准

系统验收不应以某个策略收益率为唯一标准，而应以研究质量和工程可复现性为标准。

MVP 验收标准：

- 任意策略都可以通过 YAML 配置运行，不需要手改代码。
- 每次实验都有唯一 run_id，并保存完整输入输出。
- 因子都有 metadata、依赖字段和单元测试。
- 回测结果同时报告 gross 和 net。
- Data QA 能明确标注数据缺陷和偏差风险。
- Critic Agent 能输出 blocking issues、major concerns 和 required follow-up tests。
- Report Agent 只能引用工具输出，不允许编造指标。
- 月度调仓只导出 target weights，不直接交易。

ML-ready 验收标准：

- 支持 walk-forward 训练与评估。
- 支持 forward return 和 rank label。
- 支持线性模型和 LightGBM/XGBoost 基线。
- 支持成本敏感的组合映射。
- 支持至少三种稳健性测试。

## 十六、主要风险与缓解措施

风险 1：免费数据导致偏差。缓解措施：MVP 明确标注研究用途，不把当前成分结果解释为历史可交易结论；后续替换为 PIT 数据源。

风险 2：Agent 生成看似合理但不可复现的策略。缓解措施：所有 Agent 输出必须落为 schema 和 artifact，真实计算由工具执行。

风险 3：过度追求复杂模型。缓解措施：设置 baseline gate，只有线性/树模型在净成本后稳定，才进入 Transformer/GNN/RL 阶段。

风险 4：交易成本低估。缓解措施：默认报告成本敏感性，并把换手率、ADV 参与率和容量 proxy 纳入核心指标。

风险 5：研究结果不可比较。缓解措施：统一 experiment registry，保存数据版本、特征版本、标签、模型、成本和回测引擎版本。

风险 6：未来函数和时间戳错配。缓解措施：Data QA Agent、Feature QA Agent 和标签层必须强制检查 as-of 逻辑。

## 十七、结论

这份改进报告建议把项目从“AI + 回测器”升级为“AI 辅助、PIT-aware、ML-ready、cost-aware、experiment-driven 的美股量化研究平台”。

短期最优路线不是直接做 Transformer、GNN 或 RL，而是先搭建可靠研究底座：数据缓存、因子注册、标签定义、月频回测、成本建模、实验审计、Data QA、Critic Agent 和自动报告。

一旦这个底座稳定，后续接入 SEC 文本、LightGBM 排序、新闻因子、组合优化、每日监控和月度调仓审批都可以自然扩展，不需要推倒重来。

## 附录 A：Strategy Config 示例

```yaml
strategy_id: low_vol_quality_us_v1
universe: nasdaq100
benchmark: QQQ
start_date: '2015-01-01'
end_date: '2026-06-12'
rebalance_frequency: monthly
factors:
  - name: momentum_12_1
    weight: 0.35
    direction: higher_is_better
  - name: volatility_12m
    weight: 0.35
    direction: lower_is_better
  - name: quality_proxy
    weight: 0.30
    direction: higher_is_better
portfolio:
  selection_method: top_n
  top_n: 50
  weighting: equal_weight
  max_single_name_weight: 0.05
  transaction_cost_bps: 10
risk_controls:
  long_only: true
  max_sector_weight: 0.30
  max_turnover: 0.40
validation:
  method: walk_forward
  cost_sensitivity_bps: [5, 10, 25, 50]
  robustness_tests:
    - subperiod
    - rebalance_day_shift
    - top_n_sensitivity
```

## 附录 B：Data QA Report 示例

```yaml
status: pass_with_warnings
blocking_issues: []
warnings:
  - current_universe_may_have_survivorship_bias
  - delisting_returns_not_available_in_mvp
  - fundamental_data_not_fully_point_in_time
checks:
  price_missing_rate: pass
  corporate_action_adjustment: pass
  timestamp_alignment: not_applicable_for_price_only_mvp
  filing_lag_check: pending
  universe_history_check: warning
recommendation: research_only_do_not_use_for_live_trading
```

## 附录 C：Critic Agent 输出模板

```text
# Bias and Risk Review

## Blocking Issues
- None / or list issues that must stop the workflow.

## Major Concerns
- Survivorship bias may exist if current index constituents are used historically.
- Turnover may make gross alpha non-tradable under higher cost assumptions.
- Sector exposure may explain part of the strategy return.

## Required Follow-up Tests
- Run cost sensitivity at 5/10/25/50 bps.
- Run rebalance-day shift test.
- Run subperiod test.
- Compare equal weight and constrained optimizer.
- Compare with sector-neutral version.

## Recommendation
- Reject / hold for further testing / promote to candidate strategy.
```
