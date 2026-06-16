from __future__ import annotations

import html
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ComprehensiveReportBundle:
    source_run_id: str
    source_run_path: Path
    root: Path
    html_report: Path
    structured_report: Path
    pointer: Path


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return {} if default is None else default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {} if default is None else default


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


def _created_at(manifest: dict[str, Any], path: Path) -> pd.Timestamp:
    value = pd.to_datetime(manifest.get("created_at"), utc=True, errors="coerce")
    if pd.isna(value):
        return pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC")
    return value


def find_latest_research_run(reports_root: str | Path) -> Path:
    reports = Path(reports_root)
    runs_root = reports / "runs"
    manifests: list[tuple[pd.Timestamp, Path, dict[str, Any]]] = []
    child_run_ids: set[str] = set()

    for manifest_path in runs_root.glob("*/manifest.json"):
        manifest = _read_json(manifest_path)
        if not manifest:
            continue
        run_root = manifest_path.parent
        manifests.append((_created_at(manifest, run_root), run_root, manifest))
        if manifest.get("run_type") == "experiment":
            for child in _read_json(run_root / "child_runs.json", []):
                if isinstance(child, dict) and child.get("run_id"):
                    child_run_ids.add(str(child["run_id"]))

    candidates: list[tuple[pd.Timestamp, int, Path]] = []
    for created_at, run_root, manifest in manifests:
        if manifest.get("status") != "valid":
            continue
        run_type = manifest.get("run_type")
        run_id = str(manifest.get("run_id", run_root.name))
        if run_type == "experiment" and (run_root / "research_decision.json").exists():
            candidates.append((created_at, 2, run_root))
        elif (
            run_type == "backtest"
            and run_id not in child_run_ids
            and (manifest.get("quality_gates") or {}).get("artifact_level") != "metrics"
            and (run_root / "metrics.csv").exists()
            and (run_root / "daily_returns.csv").exists()
        ):
            candidates.append((created_at, 1, run_root))

    if not candidates:
        raise ValueError(f"No valid research result found under {runs_root}.")
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def _metric_map(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty or not {"metric", "value"}.issubset(frame.columns):
        return {}
    return {
        str(row.metric): float(row.value)
        for row in frame[["metric", "value"]].itertuples(index=False)
        if pd.notna(row.value)
    }


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def _child_run_root(source_root: Path, reports_root: Path) -> Path | None:
    children = _read_json(source_root / "child_runs.json", [])
    if not children:
        return None
    full = next(
        (
            child
            for child in children
            if isinstance(child, dict) and child.get("period") == "full"
        ),
        children[0],
    )
    if not isinstance(full, dict) or not full.get("run_id"):
        return None
    child_root = reports_root / "runs" / str(full["run_id"])
    return child_root if child_root.exists() else None


def _find_acceptance(reports_root: Path, source_run_id: str) -> tuple[dict[str, Any], pd.DataFrame]:
    matches: list[tuple[pd.Timestamp, dict[str, Any], Path]] = []
    for manifest_path in (reports_root / "runs").glob("*-acceptance-*/manifest.json"):
        manifest = _read_json(manifest_path)
        notes = " ".join(str(note) for note in manifest.get("notes", []))
        if source_run_id not in notes:
            continue
        matches.append((_created_at(manifest, manifest_path.parent), manifest, manifest_path.parent))
    if not matches:
        return {}, pd.DataFrame()
    _, manifest, root = max(matches, key=lambda item: item[0])
    return manifest, _read_csv(root / "acceptance_checks.csv")


def _evaluation_paths(
    source_root: Path,
    reports_root: Path,
    manifest: dict[str, Any],
    decision: dict[str, Any],
) -> tuple[Path, Path | None]:
    if manifest.get("run_type") != "experiment":
        return source_root, None
    model = str(decision.get("selected_model", "rule_score"))
    holdout_root = source_root / "holdout_evaluation" / model
    child_root = _child_run_root(source_root, reports_root)
    return (holdout_root if holdout_root.exists() else child_root or source_root), child_root


def _fmt_pct(value: Any, digits: int = 1) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(number):
        return "N/A"
    return f"{number:.{digits}%}"


def _fmt_num(value: Any, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(number):
        return "N/A"
    return f"{number:,.{digits}f}"


def _metric_card(label: str, value: str, note: str, tone: str = "") -> str:
    return (
        f'<article class="metric {tone}"><div class="metric-label">{html.escape(label)}</div>'
        f'<div class="metric-value">{html.escape(value)}</div>'
        f'<div class="metric-note">{html.escape(note)}</div></article>'
    )


def _svg_path(
    values: pd.Series,
    width: float,
    height: float,
    padding: float,
    low: float | None = None,
    high: float | None = None,
) -> str:
    series = pd.to_numeric(values, errors="coerce").interpolate().bfill().ffill()
    if series.empty:
        return ""
    low = float(series.min()) if low is None else low
    high = float(series.max()) if high is None else high
    span = max(high - low, 1e-12)
    x_span = max(len(series) - 1, 1)
    points = []
    for index, value in enumerate(series):
        x = padding + (index / x_span) * (width - 2 * padding)
        y = padding + (1 - (float(value) - low) / span) * (height - 2 * padding)
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def _equity_chart(daily: pd.DataFrame) -> str:
    if daily.empty or "portfolio_value" not in daily:
        return '<div class="empty">没有可用的留出期净值序列。</div>'
    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame.get("date"), errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date")
    if frame.empty:
        return '<div class="empty">没有可用的留出期净值序列。</div>'
    if "benchmark_value" not in frame:
        benchmark_return = pd.to_numeric(frame.get("benchmark_return"), errors="coerce").fillna(0)
        frame["benchmark_value"] = (1 + benchmark_return).cumprod() * float(
            frame["portfolio_value"].iloc[0]
        )
    width, height, padding = 1200.0, 390.0, 42.0
    combined = pd.concat(
        [
            pd.to_numeric(frame["portfolio_value"], errors="coerce"),
            pd.to_numeric(frame["benchmark_value"], errors="coerce"),
        ],
        ignore_index=True,
    ).dropna()
    low, high = float(combined.min()), float(combined.max())
    portfolio = _svg_path(frame["portfolio_value"], width, height, padding, low, high)
    benchmark = _svg_path(frame["benchmark_value"], width, height, padding, low, high)
    start = frame["date"].iloc[0].strftime("%Y-%m-%d")
    end = frame["date"].iloc[-1].strftime("%Y-%m-%d")
    portfolio_end = float(pd.to_numeric(frame["portfolio_value"], errors="coerce").iloc[-1])
    benchmark_end = float(pd.to_numeric(frame["benchmark_value"], errors="coerce").iloc[-1])
    grid = "".join(
        f'<line x1="{padding}" y1="{padding + row * 76}" x2="{width - padding}" '
        f'y2="{padding + row * 76}" class="chart-grid"/>'
        for row in range(5)
    )
    return f"""
    <div class="chart-legend">
      <span><i class="swatch portfolio"></i>策略净值 {_fmt_num(portfolio_end, 0)}</span>
      <span><i class="swatch benchmark"></i>基准净值 {_fmt_num(benchmark_end, 0)}</span>
    </div>
    <svg class="equity-chart" viewBox="0 0 1200 390" role="img" aria-label="留出期策略与基准净值">
      {grid}
      <polyline points="{benchmark}" class="line benchmark-line"/>
      <polyline points="{portfolio}" class="line portfolio-line"/>
      <text x="{padding}" y="382" class="axis-label">{start}</text>
      <text x="{width - padding}" y="382" text-anchor="end" class="axis-label">{end}</text>
    </svg>
    """


def _drawdown_chart(daily: pd.DataFrame) -> str:
    if daily.empty or "drawdown" not in daily:
        return ""
    width, height, padding = 1200.0, 170.0, 18.0
    path = _svg_path(daily["drawdown"], width, height, padding)
    area = f"{padding},{padding} {path} {width - padding},{padding}"
    return f"""
    <svg class="drawdown-chart" viewBox="0 0 1200 170" role="img" aria-label="留出期回撤">
      <line x1="{padding}" y1="{padding}" x2="{width - padding}" y2="{padding}" class="zero-line"/>
      <polygon points="{area}" class="drawdown-area"/>
      <polyline points="{path}" class="drawdown-line"/>
    </svg>
    """


def _gate(
    label: str,
    actual: str,
    requirement: str,
    passed: bool | None,
    detail: str,
) -> str:
    if passed is True:
        state, state_label = "pass", "通过"
    elif passed is False:
        state, state_label = "fail", "未通过"
    else:
        state, state_label = "unknown", "未评估"
    return f"""
    <article class="gate {state}">
      <div class="gate-mark"><span></span></div>
      <div class="gate-body">
        <div class="gate-top">
          <h3>{html.escape(label)}</h3>
          <span class="gate-status">{state_label}</span>
        </div>
        <div class="gate-reading">
          <strong>{html.escape(actual)}</strong>
          <span>要求 {html.escape(requirement)}</span>
        </div>
        <p>{html.escape(detail)}</p>
      </div>
    </article>
    """


def _table(frame: pd.DataFrame, columns: list[str], labels: dict[str, str]) -> str:
    if frame.empty:
        return '<div class="empty">暂无数据。</div>'
    available = [column for column in columns if column in frame]
    if not available:
        return '<div class="empty">暂无数据。</div>'
    rows = []
    for row in frame[available].itertuples(index=False, name=None):
        cells = "".join(f"<td>{html.escape(str(value))}</td>" for value in row)
        rows.append(f"<tr>{cells}</tr>")
    headers = "".join(f"<th>{html.escape(labels.get(column, column))}</th>" for column in available)
    return f'<div class="table-wrap"><table><thead><tr>{headers}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def _factor_rows(factors: pd.DataFrame) -> str:
    if factors.empty:
        return '<div class="empty">没有因子证据表。</div>'
    rows = []
    max_ic = max(float(pd.to_numeric(factors.get("ic"), errors="coerce").abs().max()), 0.01)
    for row in factors.itertuples(index=False):
        ic = float(getattr(row, "ic", 0.0))
        width = min(abs(ic) / max_ic * 100, 100)
        significant = bool(getattr(row, "fdr_significant", False))
        direction = bool(getattr(row, "direction_matches", False))
        supported = significant and direction
        state = "supported" if supported else ("direction" if direction else "unsupported")
        label = "FDR 支持" if supported else ("方向一致" if direction else "方向不符")
        rows.append(
            f"""
            <div class="factor-row">
              <div class="factor-name">{html.escape(str(getattr(row, "factor_name", "")))}</div>
              <div class="factor-bar"><span class="{state}" style="width:{width:.1f}%"></span></div>
              <div class="factor-ic">{ic:+.3f}</div>
              <div class="factor-q">q={_fmt_num(getattr(row, "fdr_q_value", None), 3)}</div>
              <div class="factor-state {state}">{label}</div>
            </div>
            """
        )
    return "".join(rows)


def _status_copy(status: str) -> tuple[str, str, str]:
    mapping = {
        "supported": ("证据支持", "supported", "统计门槛与方法学门槛均已通过。"),
        "rejected": (
            "证据拒绝",
            "rejected",
            "运行有效且收益为正，但确认性证据未达到预注册标准，当前不能把结果表述为已验证 alpha。",
        ),
        "rejected_final": (
            "最终拒绝",
            "rejected",
            "该确认性研究已经关闭，当前留出期不得继续用于 v1 调参或确认性重跑。",
        ),
        "inconclusive": (
            "证据不足",
            "inconclusive",
            "现有结果不足以支持或拒绝预注册假设，需要更多独立样本或修复方法学阻断项。",
        ),
        "legacy_reference": (
            "历史参考",
            "inconclusive",
            "该结果仅作为历史基线，不属于现行确认性证据。",
        ),
    }
    return mapping.get(
        status,
        ("探索性结果", "inconclusive", "该运行没有确认性研究决策，只能作为探索性参考。"),
    )


def _reason_copy(reason: Any) -> str:
    text = str(reason)
    if text == "The Deflated Sharpe probability is below the required threshold.":
        return "Deflated Sharpe 概率低于预注册要求。"
    if text == "Methodology blockers remain unresolved.":
        return "仍有方法学阻断项未解决。"
    if text.startswith("Preregistered factor evidence did not survive"):
        return "预注册因子均未同时通过预期方向与 FDR 检验。"
    return text


def _render_html(payload: dict[str, Any], daily: pd.DataFrame, factors: pd.DataFrame) -> str:
    metrics = payload["metrics"]
    decision = payload["decision"]
    protocol = payload["protocol"]
    bootstrap = pd.DataFrame(payload["bootstrap"])
    style = payload["style_summary"]
    exposures = pd.DataFrame(payload["style_exposures"])
    acceptance = pd.DataFrame(payload["acceptance_checks"])
    evidence_label, evidence_class, evidence_copy = _status_copy(payload["evidence_status"])
    artifact_valid = payload["artifact_status"] == "valid"
    dirty_warning = ""
    if payload.get("code_dirty"):
        dirty_warning = f"""
  <section class="workspace-warning">
    <strong>Dirty git workspace</strong>
    <span>该确认性报告的代码身份包含 dirty 标记。未来确认性研究默认要求 clean commit；本报告仅保留为可追溯历史产物。Code: {html.escape(str(payload.get("code_version") or "unknown"))}</span>
  </section>
        """

    primary = bootstrap.loc[bootstrap.get("metric", pd.Series(dtype=str)) == protocol.get("primary_metric")]
    primary_lower = (
        float(primary.iloc[0]["one_sided_lower_95"]) if not primary.empty else None
    )
    threshold = float(protocol.get("primary_metric_threshold", 0.0) or 0.0)
    dsr = payload["deflated_sharpe"]
    dsr_probability = dsr.get("probability")
    dsr_required = decision.get("required_deflated_sharpe_probability", 0.95)
    trial_number = payload.get("trial_number")
    trial_budget = payload.get("trial_budget")
    trial_label = str(trial_number or "N/A")
    if trial_budget:
        trial_label = f"{trial_label}/{trial_budget}"
    holdout_inspection_count = payload.get("holdout_inspection_count") or dsr.get(
        "trial_count"
    )
    try:
        inspection_label = str(int(float(holdout_inspection_count)))
    except (TypeError, ValueError):
        inspection_label = "N/A"
    factor_supported = int(
        (
            factors.get("fdr_significant", pd.Series(dtype=bool)).astype(bool)
            & factors.get("direction_matches", pd.Series(dtype=bool)).astype(bool)
        ).sum()
    )
    factor_total = len(factors)
    acceptance_passed = (
        int(acceptance.get("passed", pd.Series(dtype=bool)).astype(bool).sum())
        if not acceptance.empty
        else 0
    )

    gates = [
        _gate(
            "产物可复算",
            f"{acceptance_passed}/{len(acceptance)} 项验收通过" if len(acceptance) else payload["artifact_status"],
            "全部验收检查通过",
            artifact_valid and (acceptance.empty or acceptance_passed == len(acceptance)),
            "数据快照、协议、账本和统计结果必须能够从保存的产物重新计算。",
        ),
        _gate(
            "主指标下界",
            _fmt_num(primary_lower, 3),
            f"> {_fmt_num(threshold, 3)}",
            None if primary_lower is None else primary_lower > threshold,
            "使用 21 交易日区块、2,000 次固定种子 Bootstrap 的单侧 95% 下界。",
        ),
        _gate(
            "Deflated Sharpe",
            _fmt_pct(dsr_probability),
            f">= {_fmt_pct(dsr_required)}",
            (
                None
                if dsr_probability is None
                else float(dsr_probability) >= float(dsr_required)
            ),
            f"试验族实际尝试 {int(float(dsr.get('trial_count', 0) or 0))} 次，修正多重尝试后的过拟合概率。",
        ),
        _gate(
            "净收益",
            _fmt_pct(metrics.get("net_total_return")),
            "> 0.0%",
            metrics.get("net_total_return") is not None
            and float(metrics["net_total_return"]) > 0,
            "留出期、统一优化器和成本账本后的净收益。",
        ),
        _gate(
            "预注册因子证据",
            f"{factor_supported}/{factor_total} 个因子",
            "方向一致且通过 FDR",
            None if factor_total == 0 else factor_supported == factor_total,
            "因子覆盖率不替代有效性；必须同时满足预期方向与 Benjamini-Hochberg FDR。",
        ),
    ]

    bootstrap_display = bootstrap.copy()
    if not bootstrap_display.empty:
        for column in ["estimate", "lower_95", "upper_95", "one_sided_lower_95"]:
            if column in bootstrap_display:
                bootstrap_display[column] = bootstrap_display[column].map(
                    lambda value: _fmt_num(value, 3)
                )
    exposure_display = exposures.copy()
    if not exposure_display.empty:
        for column in ["coefficient", "annualized_coefficient", "t_stat", "p_value"]:
            if column in exposure_display:
                exposure_display[column] = exposure_display[column].map(
                    lambda value: _fmt_num(value, 3)
                )

    blockers = decision.get("blocking_issues", [])
    reasons = decision.get("reasons", [])
    if not reasons and protocol.get("stage") != "confirmatory":
        reasons = [
            "该运行未生成确认性 Bootstrap、Deflated Sharpe 和预注册因子证据。"
        ]
    reason_items = "".join(
        f"<li>{html.escape(_reason_copy(reason))}</li>" for reason in [*reasons, *blockers]
    )
    snapshot_id = str(payload.get("data_snapshot_id") or "N/A")
    spec_hash = str(payload.get("spec_hash") or "N/A")
    study_id = str(protocol.get("study_id") or "未预注册")
    report_links = payload["links"]
    generated_label = pd.Timestamp(payload["generated_at"]).strftime("%Y-%m-%d %H:%M UTC")

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>QuantAI 综合研究报告 - {html.escape(payload["source_run_id"])}</title>
  <style>
    :root {{
      --ink: #17213c;
      --muted: #647089;
      --paper: #f4f7fb;
      --surface: #ffffff;
      --line: #d9e1ec;
      --indigo: #263f91;
      --indigo-soft: #e8edff;
      --cyan: #1e8c91;
      --cyan-soft: #e5f4f3;
      --amber: #bd761b;
      --amber-soft: #fff3dc;
      --red: #b74848;
      --red-soft: #fbe8e8;
      --shadow: 0 18px 55px rgba(28, 46, 91, 0.11);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background:
        linear-gradient(rgba(38, 63, 145, 0.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(38, 63, 145, 0.035) 1px, transparent 1px),
        var(--paper);
      background-size: 28px 28px;
      font-family: "Microsoft YaHei UI", "PingFang SC", "Segoe UI", sans-serif;
      line-height: 1.55;
    }}
    .page {{ max-width: 1320px; margin: 0 auto; padding: 28px 28px 64px; }}
    .masthead {{
      display: flex; align-items: center; justify-content: space-between;
      padding: 8px 2px 22px; color: var(--muted); font-size: 12px;
      letter-spacing: .08em; text-transform: uppercase;
    }}
    .brand {{ color: var(--indigo); font: 700 16px "Cascadia Mono", monospace; }}
    .hero {{
      position: relative; overflow: hidden; display: grid; grid-template-columns: 1.45fr .75fr;
      gap: 34px; padding: 48px; border-radius: 26px; color: white;
      background:
        radial-gradient(circle at 90% 10%, rgba(117, 203, 200, .28), transparent 32%),
        linear-gradient(132deg, #1c2a60 0%, #304e9e 58%, #277d85 120%);
      box-shadow: var(--shadow);
    }}
    .hero::after {{
      content: ""; position: absolute; right: -70px; bottom: -120px; width: 360px; height: 360px;
      border: 1px solid rgba(255,255,255,.2); border-radius: 50%;
      box-shadow: 0 0 0 35px rgba(255,255,255,.045), 0 0 0 70px rgba(255,255,255,.025);
    }}
    .eyebrow {{ font: 700 12px "Cascadia Mono", monospace; letter-spacing: .16em; opacity: .72; }}
    h1 {{ margin: 16px 0 14px; font: 700 clamp(34px, 4.2vw, 50px)/1.08 "Bahnschrift SemiCondensed", "Microsoft YaHei UI", sans-serif; letter-spacing: -.035em; }}
    .hero p {{ margin: 0; max-width: 760px; color: #dfe8ff; font-size: 17px; }}
    .verdict {{
      position: relative; z-index: 1; align-self: stretch; display: flex; flex-direction: column;
      justify-content: center; padding: 26px; border: 1px solid rgba(255,255,255,.2);
      border-radius: 19px; background: rgba(9, 19, 50, .28); backdrop-filter: blur(8px);
    }}
    .verdict-label {{ font-size: 12px; opacity: .7; letter-spacing: .12em; }}
    .verdict strong {{ margin: 8px 0; font-size: 34px; letter-spacing: -.03em; }}
    .verdict.rejected strong {{ color: #ffd0c3; }}
    .verdict.supported strong {{ color: #bff4dd; }}
    .verdict.inconclusive strong {{ color: #ffe0a3; }}
    .verdict small {{ color: #dce6ff; }}
    .workspace-warning {{
      display: grid; grid-template-columns: 210px 1fr; gap: 16px; align-items: center;
      margin: 18px 0; padding: 15px 18px; border-radius: 12px;
      color: #6f3b00; background: #fff2ce; border: 1px solid #f0c56d;
    }}
    .workspace-warning strong {{ font-size: 14px; text-transform: uppercase; letter-spacing: .08em; }}
    .workspace-warning span {{ font-size: 12px; line-height: 1.45; overflow-wrap: anywhere; }}
    .meta-strip {{
      display: grid; grid-template-columns: repeat(6, 1fr); margin: 18px 0 34px;
      background: var(--surface); border: 1px solid var(--line); border-radius: 16px;
      box-shadow: 0 8px 28px rgba(28, 46, 91, .06); overflow: hidden;
    }}
    .meta {{ padding: 18px 20px; border-right: 1px solid var(--line); min-width: 0; }}
    .meta:last-child {{ border-right: 0; }}
    .meta span {{ display: block; color: var(--muted); font-size: 11px; letter-spacing: .08em; text-transform: uppercase; }}
    .meta strong {{ display: block; margin-top: 5px; font-size: 14px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    .section {{ margin-top: 38px; }}
    .section-head {{ display: flex; align-items: end; justify-content: space-between; gap: 24px; margin-bottom: 16px; }}
    h2 {{ margin: 0; font: 700 26px "Bahnschrift SemiCondensed", "Microsoft YaHei UI", sans-serif; letter-spacing: -.02em; }}
    .section-head p {{ margin: 0; color: var(--muted); font-size: 13px; max-width: 660px; text-align: right; }}
    .metrics {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px; }}
    .metric {{ padding: 20px; background: var(--surface); border: 1px solid var(--line); border-radius: 15px; }}
    .metric-label {{ color: var(--muted); font-size: 12px; }}
    .metric-value {{ margin: 8px 0 5px; font: 700 27px "Bahnschrift SemiCondensed", "Cascadia Mono", sans-serif; letter-spacing: -.02em; }}
    .metric-note {{ color: var(--muted); font-size: 11px; }}
    .metric.negative .metric-value {{ color: var(--red); }}
    .chart-card, .card {{
      padding: 24px; background: var(--surface); border: 1px solid var(--line);
      border-radius: 18px; box-shadow: 0 10px 34px rgba(28, 46, 91, .055);
    }}
    .chart-legend {{ display: flex; justify-content: flex-end; gap: 22px; color: var(--muted); font-size: 12px; }}
    .swatch {{ display: inline-block; width: 18px; height: 3px; margin-right: 7px; vertical-align: middle; border-radius: 2px; }}
    .swatch.portfolio {{ background: var(--indigo); }}
    .swatch.benchmark {{ background: #9aa6bb; }}
    svg {{ width: 100%; height: auto; overflow: visible; }}
    .chart-grid {{ stroke: #e9eef5; stroke-width: 1; }}
    .line {{ fill: none; stroke-linecap: round; stroke-linejoin: round; stroke-width: 4; }}
    .portfolio-line {{ stroke: var(--indigo); }}
    .benchmark-line {{ stroke: #a5afbf; stroke-width: 2.5; }}
    .axis-label {{ fill: var(--muted); font: 12px "Cascadia Mono", monospace; }}
    .drawdown-chart {{ margin-top: -10px; }}
    .zero-line {{ stroke: #d3dbe8; }}
    .drawdown-area {{ fill: rgba(183, 72, 72, .10); }}
    .drawdown-line {{ fill: none; stroke: var(--red); stroke-width: 2; }}
    .evidence-grid {{ display: grid; grid-template-columns: minmax(0, 1.45fr) minmax(300px, .75fr); gap: 18px; }}
    .evidence-rail {{ position: relative; padding: 6px 0; }}
    .evidence-rail::before {{ content: ""; position: absolute; left: 17px; top: 25px; bottom: 25px; width: 2px; background: var(--line); }}
    .gate {{ position: relative; display: grid; grid-template-columns: 36px 1fr; gap: 15px; padding: 12px 0; }}
    .gate-mark {{ position: relative; z-index: 1; display: grid; place-items: center; }}
    .gate-mark span {{ width: 15px; height: 15px; border: 3px solid white; border-radius: 50%; box-shadow: 0 0 0 1px var(--line); background: #aeb8c8; }}
    .gate.pass .gate-mark span {{ background: var(--cyan); }}
    .gate.fail .gate-mark span {{ background: var(--red); }}
    .gate-body {{ padding: 16px 18px; border: 1px solid var(--line); border-radius: 14px; background: var(--surface); }}
    .gate-top {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; }}
    .gate h3 {{ margin: 0; font-size: 15px; }}
    .gate-status {{ padding: 3px 8px; border-radius: 999px; font-size: 11px; background: #eef1f5; color: var(--muted); }}
    .gate.pass .gate-status {{ color: var(--cyan); background: var(--cyan-soft); }}
    .gate.fail .gate-status {{ color: var(--red); background: var(--red-soft); }}
    .gate-reading {{ display: flex; align-items: baseline; gap: 12px; margin: 8px 0 4px; }}
    .gate-reading strong {{ font: 700 22px "Cascadia Mono", monospace; }}
    .gate-reading span, .gate p {{ color: var(--muted); font-size: 12px; }}
    .gate p {{ margin: 0; }}
    .decision-card {{ padding: 25px; border-radius: 18px; background: var(--red-soft); border: 1px solid #f1caca; }}
    .decision-card h3 {{ margin: 0 0 10px; color: var(--red); font-size: 19px; }}
    .decision-card p {{ margin: 0; color: #6f4444; }}
    .decision-card ul {{ padding-left: 19px; margin: 16px 0 0; color: #6f4444; font-size: 13px; }}
    .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
    .factor-row {{ display: grid; grid-template-columns: 175px 1fr 65px 80px 90px; gap: 12px; align-items: center; padding: 9px 0; border-bottom: 1px solid #edf1f6; font-size: 12px; }}
    .factor-row:last-child {{ border-bottom: 0; }}
    .factor-name {{ font-family: "Cascadia Mono", monospace; font-size: 11px; }}
    .factor-bar {{ height: 7px; border-radius: 5px; background: #edf1f6; overflow: hidden; }}
    .factor-bar span {{ display: block; height: 100%; border-radius: inherit; background: #aeb8c8; }}
    .factor-bar span.direction {{ background: var(--amber); }}
    .factor-bar span.supported {{ background: var(--cyan); }}
    .factor-ic, .factor-q {{ font-family: "Cascadia Mono", monospace; color: var(--muted); }}
    .factor-state {{ text-align: right; font-size: 11px; color: var(--red); }}
    .factor-state.direction {{ color: var(--amber); }}
    .factor-state.supported {{ color: var(--cyan); }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    th {{ padding: 10px; color: var(--muted); font-weight: 600; text-align: left; border-bottom: 1px solid var(--line); white-space: nowrap; }}
    td {{ padding: 10px; border-bottom: 1px solid #edf1f6; font-family: "Cascadia Mono", monospace; white-space: nowrap; }}
    .provenance {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; }}
    .hash {{ padding: 15px; border-radius: 12px; background: #eef2f8; overflow-wrap: anywhere; }}
    .hash span {{ display: block; color: var(--muted); font-size: 11px; }}
    .hash code {{ font-size: 11px; color: var(--indigo); }}
    .links {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 18px; }}
    .links a {{ color: var(--indigo); text-decoration: none; padding: 8px 11px; border: 1px solid #cfd8ec; border-radius: 9px; background: white; font-size: 12px; }}
    .empty {{ color: var(--muted); padding: 20px; text-align: center; }}
    footer {{ margin-top: 44px; padding-top: 18px; border-top: 1px solid var(--line); display: flex; justify-content: space-between; color: var(--muted); font-size: 11px; }}
    @media (max-width: 980px) {{
      .hero, .evidence-grid, .two-col {{ grid-template-columns: 1fr; }}
      .metrics {{ grid-template-columns: repeat(3, 1fr); }}
      .meta-strip {{ grid-template-columns: repeat(2, 1fr); }}
      .meta {{ border-bottom: 1px solid var(--line); }}
      .factor-row {{ grid-template-columns: 130px 1fr 55px; }}
      .factor-q, .factor-state {{ display: none; }}
    }}
    @media (max-width: 620px) {{
      .page {{ padding: 16px 14px 40px; }}
      .hero {{ padding: 28px 22px; border-radius: 18px; }}
      .metrics {{ grid-template-columns: repeat(2, 1fr); }}
      .meta-strip {{ grid-template-columns: 1fr; }}
      .meta {{ border-right: 0; }}
      .section-head {{ align-items: start; flex-direction: column; }}
      .section-head p {{ text-align: left; }}
      .provenance {{ grid-template-columns: 1fr; }}
      footer {{ flex-direction: column; gap: 8px; }}
    }}
    @media print {{
      body {{ background: white; }}
      .page {{ max-width: none; padding: 0; }}
      .hero, .card, .chart-card, .metric {{ box-shadow: none; break-inside: avoid; }}
      .links {{ display: none; }}
    }}
  </style>
</head>
<body>
<main class="page">
  <header class="masthead">
    <div class="brand">QSS / RESEARCH DOSSIER</div>
    <div>生成时间 {html.escape(generated_label)}</div>
  </header>

  <section class="hero">
    <div>
      <div class="eyebrow">确认性量化研究综合报告</div>
      <h1>收益出现了，证据还没有过关。</h1>
      <p>{html.escape(evidence_copy)}</p>
    </div>
    <div class="verdict {evidence_class}">
      <span class="verdict-label">研究结论</span>
      <strong>{evidence_label}</strong>
      <small>产物状态：{"有效" if artifact_valid else "无效"} · 试验 {html.escape(trial_label)} · Holdout 查看 {html.escape(inspection_label)} 次 · 有效产物不等于研究支持或交易批准</small>
    </div>
  </section>

  {dirty_warning}

  <section class="meta-strip">
    <div class="meta"><span>Study</span><strong title="{html.escape(study_id)}">{html.escape(study_id)}</strong></div>
    <div class="meta"><span>Source run</span><strong title="{html.escape(payload["source_run_id"])}">{html.escape(payload["source_run_id"])}</strong></div>
    <div class="meta"><span>Holdout</span><strong>{html.escape(str(protocol.get("holdout_start", "N/A")))} 至 {html.escape(str(protocol.get("holdout_end", "N/A")))}</strong></div>
    <div class="meta"><span>Data cutoff</span><strong>{html.escape(str(payload.get("data_cutoff") or "N/A"))}</strong></div>
    <div class="meta"><span>Model</span><strong>{html.escape(str(decision.get("selected_model") or payload.get("model_type") or "rule_score"))}</strong></div>
    <div class="meta"><span>Trials / Budget</span><strong>{html.escape(trial_label)}</strong></div>
  </section>

  <section class="section">
    <div class="section-head"><h2>留出期表现</h2><p>以下数字来自统一优化器、成本模型和账本后的净指标，不混入开发期。</p></div>
    <div class="metrics">
      {_metric_card("净总收益", _fmt_pct(metrics.get("net_total_return")), f"基准 {_fmt_pct(metrics.get('benchmark_total_return'))}")}
      {_metric_card("年化收益", _fmt_pct(metrics.get("cagr")), "Holdout CAGR")}
      {_metric_card("Sharpe", _fmt_num(metrics.get("sharpe_ratio")), "净收益口径")}
      {_metric_card("最大回撤", _fmt_pct(metrics.get("max_drawdown")), "峰值至谷底", "negative")}
      {_metric_card("平均换手", _fmt_pct(metrics.get("average_turnover")), "每次调仓")}
      {_metric_card("成本拖累", _fmt_pct(metrics.get("cost_drag"), 2), "累计净值影响")}
    </div>
  </section>

  <section class="section chart-card">
    <div class="section-head"><h2>净值与回撤</h2><p>报告仅展示冻结留出期，曲线不承担统计结论。</p></div>
    {_equity_chart(daily)}
    {_drawdown_chart(daily)}
  </section>

  <section class="section">
    <div class="section-head"><h2>证据刻度尺</h2><p>确认性结论必须同时通过所有门槛；单个漂亮指标不能覆盖失败项。</p></div>
    <div class="evidence-grid">
      <div class="evidence-rail">{"".join(gates)}</div>
      <aside class="decision-card">
        <h3>{evidence_label}</h3>
        <p>{html.escape(evidence_copy)}</p>
        <ul>{reason_items or "<li>没有记录额外原因。</li>"}</ul>
      </aside>
    </div>
  </section>

  <section class="section two-col">
    <article class="card">
      <div class="section-head"><h2>Bootstrap 区间</h2></div>
      {_table(bootstrap_display, ["metric", "estimate", "lower_95", "upper_95", "one_sided_lower_95"], {"metric": "指标", "estimate": "估计值", "lower_95": "双侧下界", "upper_95": "双侧上界", "one_sided_lower_95": "单侧下界"})}
    </article>
    <article class="card">
      <div class="section-head"><h2>风格回归</h2><p>覆盖率 {_fmt_pct(style.get("coverage"))} · R² {_fmt_num(style.get("r_squared"), 3)}</p></div>
      {_table(exposure_display, ["factor", "annualized_coefficient", "t_stat", "p_value"], {"factor": "因子", "annualized_coefficient": "系数", "t_stat": "HAC t", "p_value": "p-value"})}
    </article>
  </section>

  <section class="section card">
    <div class="section-head"><h2>预注册因子证据</h2><p>{factor_supported}/{factor_total} 个因子同时满足方向与 FDR。条形长度表示绝对 IC，不代表显著性。</p></div>
    {_factor_rows(factors)}
  </section>

  <section class="section two-col">
    <article class="card">
      <div class="section-head"><h2>验收检查</h2><p>{acceptance_passed}/{len(acceptance)} 通过</p></div>
      {_table(acceptance, ["check", "passed", "details"], {"check": "检查项", "passed": "通过", "details": "说明"})}
    </article>
    <article class="card">
      <div class="section-head"><h2>可复现身份</h2><p>复现实验必须匹配协议、spec 与数据快照。</p></div>
      <div class="provenance">
        <div class="hash"><span>Data snapshot</span><code>{html.escape(snapshot_id)}</code></div>
        <div class="hash"><span>Spec hash</span><code>{html.escape(spec_hash)}</code></div>
      </div>
      <div class="links">
        <a href="{html.escape(report_links["source_manifest"])}">源 Manifest</a>
        <a href="{html.escape(report_links["research_decision"])}">研究决策 JSON</a>
        <a href="{html.escape(report_links["holdout_metrics"])}">留出期指标 CSV</a>
        <a href="{html.escape(report_links["acceptance_checks"])}">验收检查 CSV</a>
      </div>
    </article>
  </section>

  <footer>
    <span>QuantAI Research System · Artifact-backed report</span>
    <span>不构成投资建议 · 结论以保存的研究协议和统计产物为准</span>
  </footer>
</main>
</body>
</html>
"""


def _relative_link(report_root: Path, target: Path) -> str:
    return Path(os.path.relpath(target, report_root)).as_posix()


def generate_comprehensive_report(
    reports_root: str | Path = "reports",
    run_path: str | Path | None = None,
    output_root: str | Path | None = None,
) -> ComprehensiveReportBundle:
    reports = Path(reports_root).resolve()
    source_root = Path(run_path).resolve() if run_path else find_latest_research_run(reports)
    manifest = _read_json(source_root / "manifest.json")
    if not manifest:
        raise ValueError(f"Missing or invalid manifest: {source_root / 'manifest.json'}")

    source_run_id = str(manifest.get("run_id", source_root.name))
    decision = _read_json(source_root / "research_decision.json")
    protocol = _read_json(source_root / "research_protocol.json")
    if not protocol:
        protocol = manifest.get("research_protocol") or {}
    evaluation_root, child_root = _evaluation_paths(source_root, reports, manifest, decision)
    metrics_frame = _read_csv(evaluation_root / "metrics.csv")
    daily = _read_csv(evaluation_root / "daily_returns.csv")
    factors = _read_csv(source_root / "factor_evidence.csv")
    if factors.empty and child_root is not None:
        factors = _read_csv(child_root / "factor_diagnostics.csv")
    bootstrap = _read_csv(source_root / "bootstrap_summary.csv")
    deflated_sharpe = _read_json(source_root / "deflated_sharpe.json")
    style_summary = _read_json(source_root / "style_factor_summary.json")
    style_exposures = _read_csv(source_root / "style_factor_exposures.csv")
    robustness = _read_csv(source_root / "robustness_matrix.csv")
    acceptance_manifest, acceptance_checks = _find_acceptance(reports, source_run_id)

    metrics = _metric_map(metrics_frame)
    if "benchmark_total_return" not in metrics and not daily.empty and "benchmark_value" in daily:
        benchmark = pd.to_numeric(daily["benchmark_value"], errors="coerce").dropna()
        if len(benchmark) > 1:
            metrics["benchmark_total_return"] = float(benchmark.iloc[-1] / benchmark.iloc[0] - 1)

    generated_at = datetime.now(timezone.utc)
    output_base = Path(output_root).resolve() if output_root else reports / "comprehensive"
    report_id = f"{generated_at:%Y%m%dT%H%M%SZ}-{source_run_id}"
    root = output_base / report_id
    suffix = 1
    while root.exists():
        root = output_base / f"{report_id}-{suffix}"
        suffix += 1
    root.mkdir(parents=True, exist_ok=False)

    acceptance_root = (
        reports / "runs" / str(acceptance_manifest.get("run_id"))
        if acceptance_manifest
        else source_root
    )
    payload = {
        "schema_version": "1.0",
        "generated_at": generated_at.isoformat(),
        "source_run_id": source_run_id,
        "source_run_path": str(source_root),
        "run_type": manifest.get("run_type"),
        "artifact_status": manifest.get("status", "unknown"),
        "code_dirty": manifest.get("code_dirty"),
        "code_version": manifest.get("code_version"),
        "evidence_status": decision.get(
            "status",
            manifest.get("evidence_status") or "exploratory",
        ),
        "study_status": decision.get(
            "study_status",
            manifest.get("study_status") or protocol.get("study_status"),
        ),
        "trial_number": manifest.get("trial_number"),
        "trial_budget": decision.get("trial_budget")
        or manifest.get("trial_budget")
        or protocol.get("trial_budget"),
        "holdout_inspection_count": deflated_sharpe.get("trial_count")
        or manifest.get("trial_number"),
        "data_cutoff": manifest.get("data_cutoff"),
        "data_snapshot_id": manifest.get("data_snapshot_id"),
        "spec_hash": manifest.get("spec_hash"),
        "model_type": decision.get("selected_model"),
        "protocol": protocol,
        "decision": decision,
        "metrics": metrics,
        "bootstrap": _records(bootstrap),
        "deflated_sharpe": deflated_sharpe,
        "style_summary": style_summary,
        "style_exposures": _records(style_exposures),
        "factor_evidence": _records(factors),
        "robustness": _records(robustness),
        "acceptance_manifest": acceptance_manifest,
        "acceptance_checks": _records(acceptance_checks),
        "links": {
            "source_manifest": _relative_link(root, source_root / "manifest.json"),
            "research_decision": _relative_link(root, source_root / "research_decision.json"),
            "holdout_metrics": _relative_link(root, evaluation_root / "metrics.csv"),
            "acceptance_checks": _relative_link(
                root,
                acceptance_root / "acceptance_checks.csv",
            ),
        },
    }

    structured_report = root / "report.json"
    html_report = root / "report.html"
    structured_report.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    html_report.write_text(_render_html(payload, daily, factors), encoding="utf-8")

    pointer = output_base / "latest.json"
    pointer.write_text(
        json.dumps(
            {
                "generated_at": payload["generated_at"],
                "source_run_id": source_run_id,
                "source_run_path": str(source_root),
                "path": str(root),
                "html_report": str(html_report),
                "structured_report": str(structured_report),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return ComprehensiveReportBundle(
        source_run_id=source_run_id,
        source_run_path=source_root,
        root=root,
        html_report=html_report,
        structured_report=structured_report,
        pointer=pointer,
    )


def ensure_comprehensive_report(
    reports_root: str | Path = "reports",
) -> ComprehensiveReportBundle:
    reports = Path(reports_root).resolve()
    source_root = find_latest_research_run(reports)
    source_manifest = _read_json(source_root / "manifest.json")
    source_run_id = str(source_manifest.get("run_id", source_root.name))
    pointer = reports / "comprehensive" / "latest.json"
    current = _read_json(pointer)
    if current.get("source_run_id") == source_run_id:
        html_report = Path(str(current.get("html_report", "")))
        structured_report = Path(str(current.get("structured_report", "")))
        root = Path(str(current.get("path", "")))
        if html_report.exists() and structured_report.exists() and root.exists():
            return ComprehensiveReportBundle(
                source_run_id=source_run_id,
                source_run_path=source_root,
                root=root,
                html_report=html_report,
                structured_report=structured_report,
                pointer=pointer,
            )
    return generate_comprehensive_report(reports, source_root)
