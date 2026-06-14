# 运行手册

更新日期：`2026-06-14`

## 环境检查

```powershell
uv sync --frozen --extra dev
uv run ruff check .
uv run python -m compileall -q src tests
uv run pytest -q
```

现有 `.venv` 也可执行：

```powershell
.\.venv\Scripts\python.exe -m compileall -q src tests
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check .
```

重度 SEC 同步前设置真实联系信息：

```powershell
$env:SEC_USER_AGENT = "Your Name your.email@example.com"
```

## 数据更新

```powershell
.\.venv\Scripts\python.exe -m qss.cli autopilot-start `
  --config configs/default.yaml `
  --start 2023-01-01 `
  --end 2026-06-12

.\.venv\Scripts\python.exe -m qss.cli autopilot-status `
  --config configs/default.yaml
```

单独验证：

```powershell
.\.venv\Scripts\python.exe -m qss.cli validate-data `
  --config configs/default.yaml `
  --start 2023-01-01 `
  --end 2026-06-12
```

截止日门禁按最近应有的美国交易日判断。请求 `2026-06-12` 时，基准行情必须
至少包含 `2026-06-12`；只有请求周末或正常休市日时才回退到上一交易日。

## 风格因子

从 Kenneth French 官方数据仓库缓存日频 FF5 与 Momentum：

```powershell
.\.venv\Scripts\python.exe -m qss.cli ingest-style-factors `
  --config configs/default.yaml
```

使用 `--refresh` 强制更新。确认性研究要求风格因子覆盖至少 95% 留出期交易日。

## 运行实验

探索性实验：

```powershell
.\.venv\Scripts\python.exe -m qss.cli run-experiment `
  --config configs/default.yaml `
  --spec experiments/example.yaml
```

确认性示例：

```powershell
.\.venv\Scripts\python.exe -m qss.cli run-experiment `
  --config configs/default.yaml `
  --spec experiments/confirmatory_rule_score.yaml
```

确认性实验额外生成：

- `research_protocol.json`
- `data_snapshot.json`
- `environment.json`
- `workspace_identity.json`
- `code.patch`（脏工作树）
- `holdout_evaluation/`
- `factor_diagnostics_scope.json`
- `bootstrap_summary.csv`
- `deflated_sharpe.json`
- `style_factor_exposures.csv`
- `factor_evidence.csv`
- `research_decision.json` 和 Markdown 报告

输入数据归档位于 `data/archive/research_inputs/`。该目录不进入 Git，应由
备份或对象存储策略单独保护。

确认性规格必须列出：

```yaml
robustness_tests:
  - subperiod
  - cost_sensitivity
  - top_n_sensitivity
  - rebalance_day_shift
```

## 比较与登记

```powershell
.\.venv\Scripts\python.exe -m qss.cli registry-refresh
.\.venv\Scripts\python.exe -m qss.cli registry-query --strategy-id multifactor_balanced_us
```

Baseline comparison 仅把确认性留出期指标用于结论；旧实验显示为
`legacy_reference`：

```powershell
.\.venv\Scripts\python.exe -m qss.cli baseline-comparison `
  --experiment-run 20260614T011051Z-experiment-ec84fa0a `
  --experiment-run 20260613T133519Z-experiment-7ad846cd `
  --output reports/research/holdout_baseline_comparison.md
```

## 前端与综合报告

```powershell
start_frontend.bat
```

启动脚本会先扫描最新有效研究结果，生成综合 HTML/JSON 报告，再启动
Streamlit。前端默认打开 `Research Brief`，原有 Overview 与 Backtest 页面
读取同一份最新留出期结果。

也可单独生成：

```powershell
.\.venv\Scripts\python.exe -m qss.cli comprehensive-report
```

产物位于 `reports/comprehensive/<生成时间>-<source_run_id>/`，稳定指针为
`reports/comprehensive/latest.json`。综合报告不会改写历史 run 目录。

## 运行边界

当前命令只生成研究与人工审阅产物。审批包不代表策略获得 paper trading 或
实盘资格，本系统不连接 broker。

当前成交模型是执行日 `close`。停牌、部分成交、真实点差/冲击和公司行动对账
尚未达到实盘级要求，详见[当前研究状态](./CURRENT_RESEARCH_STATUS.md)。
