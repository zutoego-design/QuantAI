from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import time
from collections import deque
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
import plotly.express as px
import streamlit as st
import yaml

from qss.config.loader import get_config

st.set_page_config(
    page_title="QSS Control Deck",
    page_icon="Q",
    layout="wide",
    initial_sidebar_state="expanded",
)


REPORTS = ROOT / "reports"
DATA = ROOT / "data"
RAW = DATA / "raw"
SILVER = DATA / "silver"
GOLD = DATA / "gold"
CONFIGS = ROOT / "configs"
DEFAULT_CONFIG_PATH = "configs/default.yaml"

DEFAULT_CONFIG_FILE = CONFIGS / "default.yaml"
STRATEGY_CONFIG_FILE = CONFIGS / "strategy_multifactor_balanced.yaml"
OPTIMIZER_CONFIG_FILE = CONFIGS / "optimizer.yaml"
RISK_CONFIG_FILE = CONFIGS / "risk_limits.yaml"
UNIVERSE_CONFIG_FILE = CONFIGS / "universe_us_large_mid.yaml"

CLI_REQUIRED_MODULES = {
    "cvxpy": "cvxpy",
    "duckdb": "duckdb",
    "typer": "typer",
}


def _apply_style() -> None:
    st.markdown(
        """
        <style>
        :root {
            --canvas: #f3f6fa;
            --canvas-warm: #f8f4ee;
            --surface: #ffffff;
            --surface-soft: #f8fafc;
            --navy: #172a46;
            --navy-soft: #243b5d;
            --ink: #172033;
            --muted: #5f6d82;
            --line: #dce3ec;
            --line-strong: #c9d3e1;
            --accent: #e75f43;
            --accent-deep: #c8472d;
            --good: #18794e;
            --warn: #9a650f;
            --bad: #b23a3a;
            --focus: rgba(231, 95, 67, 0.22);
            --shadow-sm: 0 8px 24px rgba(23, 42, 70, 0.07);
            --shadow-lg: 0 22px 55px rgba(23, 42, 70, 0.12);
        }

        html, body, [class*="css"] {
            font-family: Inter, "Segoe UI", Arial, sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at 92% 2%, rgba(231, 95, 67, 0.10), transparent 26rem),
                linear-gradient(135deg, var(--canvas) 0%, var(--canvas-warm) 100%);
            color: var(--ink);
        }

        [data-testid="stHeader"] {
            background: rgba(243, 246, 250, 0.88);
            border-bottom: 1px solid rgba(220, 227, 236, 0.72);
            backdrop-filter: blur(14px);
        }

        [data-testid="stToolbar"] {
            color: var(--ink);
        }

        [data-testid="stSidebar"] {
            background:
                radial-gradient(circle at 20% 0%, rgba(231, 95, 67, 0.18), transparent 15rem),
                linear-gradient(180deg, #14253e 0%, #1c3150 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] label p {
            color: #ffffff !important;
        }

        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            color: #cbd6e5 !important;
        }

        [data-testid="stSidebar"] div[data-baseweb="input"],
        [data-testid="stSidebar"] div[data-baseweb="base-input"] {
            background: rgba(255, 255, 255, 0.10) !important;
            border-color: rgba(255, 255, 255, 0.18) !important;
        }

        [data-testid="stSidebar"] input {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
        }

        [data-testid="stSidebar"] pre {
            background: rgba(8, 18, 34, 0.50) !important;
            border: 1px solid rgba(255, 255, 255, 0.10);
        }

        [data-testid="stSidebar"] pre code {
            color: #e8eef7 !important;
        }

        [data-testid="stSidebarCollapseButton"] button {
            background: var(--navy);
            color: #ffffff;
            border: 1px solid rgba(255, 255, 255, 0.12);
        }

        .block-container {
            padding-top: 1.35rem;
            padding-bottom: 3rem;
            max-width: 1500px;
        }

        .hero-shell {
            background:
                radial-gradient(circle at 88% 15%, rgba(231, 95, 67, 0.28), transparent 15rem),
                linear-gradient(135deg, #162a47 0%, #29496f 100%);
            color: #ffffff;
            border-radius: 22px;
            padding: 1.7rem 1.8rem 1.55rem;
            box-shadow: var(--shadow-lg);
            border: 1px solid rgba(255, 255, 255, 0.10);
            margin-bottom: 1.15rem;
            overflow: hidden;
        }

        .hero-shell h1 {
            letter-spacing: -0.035em;
            margin: 0 0 0.45rem 0;
            font-size: clamp(1.85rem, 3vw, 2.55rem);
            color: #ffffff !important;
        }

        .hero-shell p {
            margin: 0;
            max-width: 980px;
            font-size: 1.02rem;
            line-height: 1.6;
            color: #dce7f4 !important;
        }

        .page-label {
            text-transform: uppercase;
            letter-spacing: 0.18em;
            font-size: 0.72rem;
            color: #b9cce3;
            margin-bottom: 0.4rem;
            font-weight: 700;
        }

        .hero-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 1rem;
        }

        .hero-meta span {
            padding: 0.38rem 0.7rem;
            border-radius: 999px;
            color: #f7fbff;
            background: rgba(255, 255, 255, 0.10);
            border: 1px solid rgba(255, 255, 255, 0.14);
            font-size: 0.78rem;
            font-weight: 650;
        }

        .metric-card {
            background: var(--surface);
            border-radius: 16px;
            padding: 1rem 1.05rem;
            border: 1px solid var(--line);
            box-shadow: var(--shadow-sm);
            min-height: 128px;
        }

        .metric-kicker {
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.72rem;
            color: var(--muted);
            font-weight: 600;
            margin-bottom: 0.55rem;
        }

        .metric-value {
            font-size: 1.85rem;
            font-weight: 700;
            line-height: 1.05;
            color: var(--ink);
            margin-bottom: 0.35rem;
        }

        .metric-note {
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.45;
        }

        .panel-card {
            background: var(--surface);
            border-radius: 16px;
            border: 1px solid var(--line);
            box-shadow: var(--shadow-sm);
            padding: 1.15rem 1.25rem;
            margin-bottom: 1rem;
        }

        .panel-card h3 {
            margin: 0 0 0.4rem 0;
            color: var(--ink);
            font-size: 1.25rem;
        }

        .panel-card p {
            margin: 0;
            color: var(--muted);
            line-height: 1.55;
        }

        .subtle-label {
            text-transform: uppercase;
            letter-spacing: 0.16em;
            font-size: 0.72rem;
            color: var(--muted);
            font-weight: 600;
            margin-bottom: 0.35rem;
        }

        .status-pill {
            display: inline-block;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 600;
            margin: 0.1rem 0.35rem 0.1rem 0;
        }

        .status-good {
            background: rgba(31, 122, 82, 0.12);
            color: var(--good);
        }

        .status-warn {
            background: rgba(167, 106, 18, 0.12);
            color: var(--warn);
        }

        .status-bad {
            background: rgba(161, 58, 58, 0.12);
            color: var(--bad);
        }

        .status-neutral {
            background: #e9eef5;
            color: var(--ink);
        }

        [data-testid="stSidebar"] .status-good {
            color: #9ce7c5 !important;
            background: rgba(72, 187, 132, 0.16);
        }

        [data-testid="stSidebar"] .status-warn {
            color: #f6d88d !important;
            background: rgba(232, 176, 63, 0.16);
        }

        [data-testid="stSidebar"] .status-bad {
            color: #ffb4b4 !important;
            background: rgba(235, 87, 87, 0.16);
        }

        [data-testid="stSidebar"] .status-neutral {
            color: #e8eef7 !important;
            background: rgba(255, 255, 255, 0.10);
        }

        h1, h2, h3, h4, h5, h6,
        [data-testid="stMarkdownContainer"] p,
        label p {
            color: var(--ink);
        }

        [data-testid="stCaptionContainer"] {
            color: var(--muted);
        }

        div[role="radiogroup"] {
            display: flex;
            flex-wrap: wrap;
            gap: 0.3rem;
            padding: 0.35rem;
            margin: 0 0 1rem;
            border: 1px solid var(--line);
            border-radius: 15px;
            background: rgba(255, 255, 255, 0.92);
            box-shadow: var(--shadow-sm);
        }

        div[role="radiogroup"] label[data-baseweb="radio"] {
            border-radius: 10px;
            padding: 0.58rem 0.78rem;
            transition: background 140ms ease, color 140ms ease, transform 140ms ease;
        }

        div[role="radiogroup"] label[data-baseweb="radio"] > div:first-child {
            display: none;
        }

        div[role="radiogroup"] label[data-baseweb="radio"] p {
            color: var(--ink) !important;
            font-weight: 650;
        }

        div[role="radiogroup"] label[data-baseweb="radio"]:has(input:checked) {
            background: var(--navy);
            box-shadow: 0 5px 14px rgba(23, 42, 70, 0.18);
        }

        div[role="radiogroup"] label[data-baseweb="radio"]:has(input:checked) p {
            color: #ffffff !important;
        }

        div[role="radiogroup"] label[data-baseweb="radio"]:hover {
            background: #edf2f7;
        }

        div[role="radiogroup"] label[data-baseweb="radio"]:has(input:checked):hover {
            background: var(--navy-soft);
        }

        div[data-baseweb="tab-list"] {
            gap: 0.35rem;
            border-bottom: 1px solid var(--line);
            padding-bottom: 0.45rem;
        }

        div[data-baseweb="tab"] {
            border-radius: 10px;
            padding: 0.35rem 0.85rem;
            background: #edf2f7;
            color: var(--ink);
        }

        div[data-baseweb="tab"][aria-selected="true"] {
            background: var(--navy);
            color: #ffffff;
        }

        div[data-baseweb="input"],
        div[data-baseweb="select"] > div,
        textarea {
            background: var(--surface) !important;
            border-color: var(--line-strong) !important;
            color: var(--ink) !important;
            border-radius: 10px !important;
        }

        input, textarea {
            color: var(--ink) !important;
            -webkit-text-fill-color: var(--ink) !important;
        }

        input:focus, textarea:focus {
            box-shadow: 0 0 0 3px var(--focus) !important;
        }

        .stButton > button,
        .stDownloadButton > button {
            min-height: 2.75rem;
            border-radius: 10px;
            padding: 0.58rem 1.05rem;
            border: 1px solid var(--line-strong);
            background: var(--surface);
            color: var(--ink) !important;
            font-weight: 650;
            box-shadow: 0 3px 10px rgba(23, 42, 70, 0.05);
            transition: border-color 140ms ease, transform 140ms ease, box-shadow 140ms ease;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            border-color: var(--navy);
            color: var(--navy) !important;
            transform: translateY(-1px);
            box-shadow: 0 7px 16px rgba(23, 42, 70, 0.10);
        }

        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, var(--accent), var(--accent-deep));
            color: #ffffff !important;
            border-color: transparent;
            box-shadow: 0 8px 18px rgba(200, 71, 45, 0.22);
        }

        .stButton > button[kind="primary"]:hover {
            color: #ffffff !important;
            border-color: transparent;
            box-shadow: 0 11px 22px rgba(200, 71, 45, 0.30);
        }

        button[data-testid="stBaseButton-primary"],
        button[data-testid="stBaseButton-primary"] p {
            color: #ffffff !important;
        }

        button[data-testid="stBaseButton-secondary"] p {
            color: var(--ink) !important;
        }

        .stTextInput label,
        .stDateInput label,
        .stNumberInput label,
        .stSelectbox label,
        .stTextArea label,
        .stMultiSelect label {
            font-weight: 600;
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(255, 255, 255, 0.78);
            border-color: var(--line) !important;
            border-radius: 16px !important;
            box-shadow: var(--shadow-sm);
        }

        [data-testid="stAlert"] {
            border-radius: 12px;
            border: 1px solid var(--line);
        }

        [data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 12px;
            overflow: hidden;
            background: var(--surface);
        }

        [data-testid="stStatusWidget"] {
            border: 1px solid var(--line);
            border-radius: 12px;
            background: var(--surface);
        }

        pre {
            border-radius: 12px !important;
        }

        .mono-small {
            font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
            font-size: 0.84rem;
            color: var(--muted);
            overflow-wrap: anywhere;
        }

        @media (max-width: 900px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .hero-shell {
                padding: 1.35rem;
            }

            div[role="radiogroup"] label[data-baseweb="radio"] {
                padding: 0.5rem 0.62rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _safe_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _safe_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _fmt_pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.2%}"


def _fmt_number(value: float | int | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):,.{digits}f}"


def _fmt_billions(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"${float(value) / 1_000_000_000:.1f}B"


def _to_date(value: Any, fallback: date) -> date:
    if value is None or pd.isna(value):
        return fallback
    return pd.Timestamp(value).date()


def _status_class(status: str) -> str:
    mapping = {
        "success": "status-good",
        "warning": "status-warn",
        "error": "status-bad",
        "info": "status-neutral",
    }
    return mapping.get(status, "status-neutral")


def _status_pill(text: str, status: str = "info") -> str:
    return f'<span class="status-pill {_status_class(status)}">{text}</span>'


def _metric_card(title: str, value: str, note: str) -> str:
    return f"""
    <div class="metric-card">
      <div class="metric-kicker">{title}</div>
      <div class="metric-value">{value}</div>
      <div class="metric-note">{note}</div>
    </div>
    """


def _panel_header(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="panel-card">
          <h3>{title}</h3>
          <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _read_yaml_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _first_non_null(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, float) and pd.isna(value):
            continue
        if pd.isna(value):
            continue
        return value
    return None


def _backup_and_write_yaml(path: Path, payload: dict[str, Any]) -> None:
    backup_dir = CONFIGS / ".backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{path.stem}_{timestamp}.yaml"
    shutil.copy2(path, backup_path)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


@st.cache_data(ttl=5, show_spinner=False)
def _load_artifacts() -> dict[str, Any]:
    quality_files = sorted((REPORTS / "data_quality").glob("data_quality_*.csv"))
    risk_files = sorted(
        (REPORTS / "runs").glob("*risk*/alerts.csv"),
        key=lambda path: path.stat().st_mtime,
    )
    rebalance_weight_files = sorted(
        (REPORTS / "runs").glob("*rebalance*/target_weights.csv"),
        key=lambda path: path.stat().st_mtime,
    )

    latest_run_file = REPORTS / "latest_run.json"
    latest_run_root = None
    if latest_run_file.exists():
        try:
            latest_run_root = Path(json.loads(latest_run_file.read_text(encoding="utf-8"))["path"])
        except (KeyError, ValueError, OSError):
            latest_run_root = None
    backtest_daily = (
        _safe_csv(latest_run_root / "daily_returns.csv")
        if latest_run_root
        else pd.DataFrame()
    )
    backtest_metrics = (
        _safe_csv(latest_run_root / "metrics.csv")
        if latest_run_root
        else pd.DataFrame()
    )
    portfolio = _safe_parquet(GOLD / "portfolios" / "portfolio_weights.parquet")
    scores = _safe_parquet(GOLD / "scores" / "alpha_scores.parquet")
    factors = _safe_parquet(GOLD / "factors" / "factor_values.parquet")
    universe = _safe_parquet(SILVER / "universe" / "universe_membership.parquet")
    macro = _safe_parquet(GOLD / "macro" / "macro_regime.parquet")
    validation_checks = sorted(
        (REPORTS / "runs").glob("*data-validation*/checks.csv"),
        key=lambda path: path.stat().st_mtime,
    )
    quality = (
        _safe_csv(validation_checks[-1])
        if validation_checks
        else (_safe_csv(quality_files[-1]) if quality_files else pd.DataFrame())
    )
    risk_alerts = _safe_csv(risk_files[-1]) if risk_files else pd.DataFrame()
    target_weights_csv = _safe_csv(rebalance_weight_files[-1]) if rebalance_weight_files else pd.DataFrame()

    return {
        "backtest_daily": backtest_daily,
        "backtest_metrics": backtest_metrics,
        "portfolio": portfolio,
        "scores": scores,
        "factors": factors,
        "universe": universe,
        "macro": macro,
        "quality": quality,
        "risk_alerts": risk_alerts,
        "target_weights_csv": target_weights_csv,
    }


def _try_load_config(config_path: str) -> tuple[Any | None, str | None]:
    try:
        return get_config([config_path]), None
    except Exception as exc:  # pragma: no cover - UI path
        return None, str(exc)


def _artifact_snapshot(artifacts: dict[str, Any]) -> list[dict[str, str]]:
    files = {
        "Legacy Demo Metrics": REPORTS / "backtest" / "backtest_metrics.csv",
        "Legacy Demo Daily Returns": REPORTS / "backtest" / "daily_returns.csv",
        "Portfolio Weights": GOLD / "portfolios" / "portfolio_weights.parquet",
        "Alpha Scores": GOLD / "scores" / "alpha_scores.parquet",
        "Universe Membership": SILVER / "universe" / "universe_membership.parquet",
        "Macro Regime": GOLD / "macro" / "macro_regime.parquet",
        "Risk Alerts": max(sorted((REPORTS / "risk").glob("risk_alerts_*.csv")), default=REPORTS / "risk" / "missing.csv"),
        "Data Quality": max(sorted((REPORTS / "data_quality").glob("data_quality_*.csv")), default=REPORTS / "data_quality" / "missing.csv"),
    }
    rows: list[dict[str, str]] = []
    for label, path in files.items():
        if path.exists():
            modified = pd.Timestamp(path.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M:%S")
            size_kb = f"{path.stat().st_size / 1024:.1f} KB"
            rows.append({"artifact": label, "updated": modified, "size": size_kb, "path": str(path)})
        else:
            rows.append({"artifact": label, "updated": "missing", "size": "-", "path": str(path)})
    return rows


def _latest_report_dates(artifacts: dict[str, Any]) -> dict[str, date]:
    today = pd.Timestamp.today().date()
    portfolio = artifacts["portfolio"]
    backtest_daily = artifacts["backtest_daily"]
    universe = artifacts["universe"]

    latest_portfolio = portfolio["date"].max() if not portfolio.empty and "date" in portfolio else None
    latest_backtest = backtest_daily["date"].max() if not backtest_daily.empty and "date" in backtest_daily else None
    latest_universe = universe["date"].max() if not universe.empty and "date" in universe else None
    return {
        "pipeline_date": _to_date(_first_non_null(latest_portfolio, latest_universe), today),
        "risk_date": _to_date(_first_non_null(latest_portfolio, latest_backtest), today),
        "backtest_end": _to_date(_first_non_null(latest_backtest, latest_portfolio), today),
    }


def _init_state(artifacts: dict[str, Any]) -> None:
    config_obj, _ = _try_load_config(st.session_state.get("config_path", DEFAULT_CONFIG_PATH))
    defaults = _latest_report_dates(artifacts)
    today = pd.Timestamp.today().date()

    if "config_path" not in st.session_state:
        st.session_state["config_path"] = DEFAULT_CONFIG_PATH
    if "pipeline_date" not in st.session_state:
        st.session_state["pipeline_date"] = defaults["pipeline_date"]
    if "risk_date" not in st.session_state:
        st.session_state["risk_date"] = defaults["risk_date"]
    if "price_start_date" not in st.session_state:
        if config_obj is not None:
            st.session_state["price_start_date"] = _to_date(config_obj.backtest.start_date, today)
        else:
            st.session_state["price_start_date"] = date(2015, 1, 1)
    if "backtest_start_date" not in st.session_state:
        if config_obj is not None:
            st.session_state["backtest_start_date"] = _to_date(config_obj.backtest.start_date, today)
        else:
            st.session_state["backtest_start_date"] = date(2015, 1, 1)
    if "backtest_end_date" not in st.session_state:
        st.session_state["backtest_end_date"] = defaults["backtest_end"]
    if "run_history" not in st.session_state:
        st.session_state["run_history"] = []
    if "page" not in st.session_state:
        st.session_state["page"] = "Command Deck"


def _remember_run(label: str, command: list[str], returncode: int, duration: float, output: str) -> None:
    entry = {
        "label": label,
        "command": " ".join(command),
        "returncode": returncode,
        "duration": round(duration, 2),
        "output": output,
        "finished_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    history = st.session_state.get("run_history", [])
    history.insert(0, entry)
    st.session_state["run_history"] = history[:8]


def _missing_cli_modules() -> list[str]:
    return [
        distribution
        for import_name, distribution in CLI_REQUIRED_MODULES.items()
        if importlib.util.find_spec(import_name) is None
    ]


def _run_cli_task(label: str, cli_args: list[str]) -> None:
    command = [sys.executable, "-m", "qss.cli", *cli_args]
    missing = _missing_cli_modules()
    if missing:
        details = (
            f"Cannot start {label}. The active Python runtime is missing: {', '.join(missing)}.\n"
            f"Interpreter: {sys.executable}\n"
            f"Repair command: \"{sys.executable}\" -m pip install -e ."
        )
        _remember_run(label, command, 2, 0.0, details)
        st.error(details)
        return

    output_lines: deque[str] = deque(maxlen=240)
    log_box = st.empty()
    started = time.perf_counter()
    status = st.status(f"Running {label}", expanded=True)
    status.write(f"Command: {' '.join(command)}")

    returncode = 1
    try:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            clean = line.rstrip()
            output_lines.append(clean)
            log_box.code("\n".join(output_lines), language="bash")
        returncode = process.wait()
    except OSError as exc:
        output_lines.append(f"Failed to launch task: {exc}")
    finally:
        duration = time.perf_counter() - started

    output = "\n".join(output_lines)
    _remember_run(label, command, returncode, duration, output)
    if returncode == 0:
        status.update(label=f"{label} completed in {duration:.1f}s", state="complete", expanded=False)
        st.success(f"{label} completed.")
    else:
        status.update(label=f"{label} failed after {duration:.1f}s", state="error", expanded=True)
        st.error(f"{label} failed. See log tail below.")
    st.cache_data.clear()
    st.rerun()


def _render_last_run() -> None:
    history = st.session_state.get("run_history", [])
    if not history:
        return
    latest = history[0]
    state = "success" if latest["returncode"] == 0 else "error"
    st.markdown(
        f"""
        <div class="panel-card">
          <div class="subtle-label">Latest Run</div>
          <h3>{latest['label']}</h3>
          <p>Finished at {latest['finished_at']} in {latest['duration']}s</p>
          <div style="margin-top:0.55rem;">{_status_pill('success' if state == 'success' else 'failed', state)}</div>
          <div class="mono-small" style="margin-top:0.8rem;">{latest['command']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if latest["output"]:
        st.code(latest["output"], language="bash")


def _save_parameter_updates() -> None:
    default_cfg = _read_yaml_file(DEFAULT_CONFIG_FILE)
    strategy_cfg = _read_yaml_file(STRATEGY_CONFIG_FILE)
    optimizer_cfg = _read_yaml_file(OPTIMIZER_CONFIG_FILE)
    risk_cfg = _read_yaml_file(RISK_CONFIG_FILE)
    universe_cfg = _read_yaml_file(UNIVERSE_CONFIG_FILE)

    default_cfg.setdefault("backtest", {})
    default_cfg["backtest"]["start_date"] = str(st.session_state["cfg_backtest_start"])
    default_cfg["backtest"]["initial_capital"] = float(st.session_state["cfg_initial_capital"])
    default_cfg["backtest"]["rebalance_execution_lag_days"] = int(st.session_state["cfg_exec_lag"])
    default_cfg["backtest"].setdefault("transaction_cost", {})
    default_cfg["backtest"]["transaction_cost"]["commission_bps"] = float(st.session_state["cfg_commission_bps"])
    default_cfg["backtest"]["transaction_cost"]["slippage_bps"] = float(st.session_state["cfg_slippage_bps"])

    strategy_cfg.setdefault("strategy", {})
    strategy_cfg["strategy"]["benchmark"] = st.session_state["cfg_benchmark"].strip().upper()

    optimizer_cfg.setdefault("optimizer", {})
    optimizer_cfg["optimizer"].setdefault("objective", {})
    optimizer_cfg["optimizer"].setdefault("constraints", {})
    optimizer_cfg["optimizer"].setdefault("fallback", {})
    optimizer_cfg["optimizer"]["objective"]["risk_aversion"] = float(st.session_state["cfg_risk_aversion"])
    optimizer_cfg["optimizer"]["objective"]["turnover_penalty"] = float(st.session_state["cfg_turnover_penalty"])
    optimizer_cfg["optimizer"]["constraints"]["max_weight"] = float(st.session_state["cfg_max_weight"])
    optimizer_cfg["optimizer"]["constraints"]["target_num_holdings"] = int(st.session_state["cfg_target_holdings"])
    optimizer_cfg["optimizer"]["constraints"]["max_sector_weight"] = float(st.session_state["cfg_max_sector_weight"])
    optimizer_cfg["optimizer"]["constraints"]["max_turnover_per_rebalance"] = float(st.session_state["cfg_max_turnover"])
    tracking_error_limit = float(st.session_state["cfg_tracking_error_limit"])
    optimizer_cfg["optimizer"]["constraints"]["tracking_error_limit"] = (
        tracking_error_limit if tracking_error_limit > 0 else None
    )
    optimizer_cfg["optimizer"]["fallback"]["top_n"] = int(st.session_state["cfg_fallback_top_n"])

    risk_cfg.setdefault("risk_limits", {})
    risk_cfg["risk_limits"].setdefault("portfolio", {})
    risk_cfg["risk_limits"]["portfolio"]["max_daily_loss"] = float(st.session_state["cfg_max_daily_loss"])
    risk_cfg["risk_limits"]["portfolio"]["max_drawdown_alert"] = float(st.session_state["cfg_max_drawdown"])
    risk_cfg["risk_limits"]["portfolio"]["max_realized_vol_annualized"] = float(st.session_state["cfg_max_realized_vol"])
    risk_cfg["risk_limits"]["portfolio"]["min_beta_to_benchmark"] = float(st.session_state["cfg_beta_min"])
    risk_cfg["risk_limits"]["portfolio"]["max_beta_to_benchmark"] = float(st.session_state["cfg_beta_max"])
    risk_cfg["risk_limits"]["portfolio"]["max_tracking_error"] = float(st.session_state["cfg_risk_tracking_error"])

    universe_cfg.setdefault("universe", {})
    universe_cfg["universe"].setdefault("filters", {})
    universe_cfg["universe"].setdefault("exclude", {})
    universe_cfg["universe"]["filters"]["min_market_cap"] = float(st.session_state["cfg_min_market_cap"])
    universe_cfg["universe"]["filters"]["min_price"] = float(st.session_state["cfg_min_price"])
    universe_cfg["universe"]["filters"]["min_adv_20d"] = float(st.session_state["cfg_min_adv_20d"])
    universe_cfg["universe"]["exclude"]["sectors"] = [value.strip() for value in st.session_state["cfg_exclude_sectors"].split(",") if value.strip()]

    _backup_and_write_yaml(DEFAULT_CONFIG_FILE, default_cfg)
    _backup_and_write_yaml(STRATEGY_CONFIG_FILE, strategy_cfg)
    _backup_and_write_yaml(OPTIMIZER_CONFIG_FILE, optimizer_cfg)
    _backup_and_write_yaml(RISK_CONFIG_FILE, risk_cfg)
    _backup_and_write_yaml(UNIVERSE_CONFIG_FILE, universe_cfg)
    st.cache_data.clear()
    st.success("Configuration files updated. Backups were written to configs/.backups.")


def _render_sidebar(artifacts: dict[str, Any], config_error: str | None) -> None:
    st.sidebar.markdown("## QSS Control Deck")
    st.sidebar.caption("Run research jobs, tune config, and inspect artifacts from one surface.")
    st.sidebar.text_input("Config entrypoint", key="config_path")

    st.sidebar.markdown("### Runtime")
    missing = _missing_cli_modules()
    if missing:
        st.sidebar.markdown(
            _status_pill(f"missing {', '.join(missing)}", "error"),
            unsafe_allow_html=True,
        )
        st.sidebar.caption(f"Python: {sys.executable}")
    else:
        st.sidebar.markdown(_status_pill("pipeline ready", "success"), unsafe_allow_html=True)
        st.sidebar.caption(f"Python {sys.version_info.major}.{sys.version_info.minor} | project environment")

    latest_portfolio = artifacts["portfolio"]["date"].max() if not artifacts["portfolio"].empty else None
    latest_backtest = artifacts["backtest_daily"]["date"].max() if not artifacts["backtest_daily"].empty else None
    latest_macro = artifacts["macro"]["date"].max() if not artifacts["macro"].empty else None

    st.sidebar.markdown("### Freshness")
    st.sidebar.markdown(_status_pill(f"portfolio {str(latest_portfolio)[:10] if latest_portfolio is not None else 'missing'}", "info"), unsafe_allow_html=True)
    st.sidebar.markdown(_status_pill(f"backtest {str(latest_backtest)[:10] if latest_backtest is not None else 'missing'}", "info"), unsafe_allow_html=True)
    st.sidebar.markdown(_status_pill(f"macro {str(latest_macro)[:10] if latest_macro is not None else 'missing'}", "info"), unsafe_allow_html=True)

    if config_error:
        st.sidebar.error(f"Config load failed: {config_error}")
    else:
        st.sidebar.success("Config profile parsed successfully.")

    history = st.session_state.get("run_history", [])
    if history:
        st.sidebar.markdown("### Recent Task")
        last = history[0]
        tone = "success" if last["returncode"] == 0 else "error"
        st.sidebar.markdown(_status_pill(last["label"], tone), unsafe_allow_html=True)
        st.sidebar.caption(f"{last['finished_at']} | {last['duration']}s")

    st.sidebar.markdown("### Quick Start")
    st.sidebar.code(
        "1. Save strategy parameters\n"
        "2. Run Monthly Pipeline\n"
        "3. Run Backtest\n"
        "4. Run Risk Monitor",
        language="text",
    )


def _render_hero(title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="hero-shell">
          <div class="page-label">Quant Stock Selection System</div>
          <h1>{title}</h1>
          <p>{description}</p>
          <div class="hero-meta">
            <span>Local research console</span>
            <span>Config-driven workflow</span>
            <span>Artifact-backed monitoring</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_overview(artifacts: dict[str, Any]) -> None:
    backtest_daily = artifacts["backtest_daily"]
    backtest_metrics = artifacts["backtest_metrics"]
    portfolio = artifacts["portfolio"]
    risk_alerts = artifacts["risk_alerts"]

    latest_backtest = backtest_daily.tail(1)
    latest_portfolio_date = portfolio["date"].max() if not portfolio.empty else None
    if {"metric", "value"}.issubset(backtest_metrics.columns):
        annual_return = backtest_metrics.loc[backtest_metrics["metric"] == "annualized_return", "value"]
        sharpe = backtest_metrics.loc[backtest_metrics["metric"] == "sharpe_ratio", "value"]
        max_dd = backtest_metrics.loc[backtest_metrics["metric"] == "max_drawdown", "value"]
    else:
        annual_return = pd.Series(dtype="float64")
        sharpe = pd.Series(dtype="float64")
        max_dd = pd.Series(dtype="float64")

    cols = st.columns(4)
    cols[0].markdown(
        _metric_card(
            "Latest Rebalance",
            str(latest_portfolio_date)[:10] if latest_portfolio_date is not None else "N/A",
            "Most recent saved portfolio date.",
        ),
        unsafe_allow_html=True,
    )
    cols[1].markdown(
        _metric_card(
            "Portfolio Return",
            _fmt_pct(latest_backtest["portfolio_return"].iloc[-1]) if not latest_backtest.empty else "N/A",
            "Latest saved daily return from the backtest artifact.",
        ),
        unsafe_allow_html=True,
    )
    cols[2].markdown(
        _metric_card(
            "Annualized Return",
            _fmt_pct(annual_return.iloc[0]) if not annual_return.empty else "N/A",
            "Computed from saved daily returns.",
        ),
        unsafe_allow_html=True,
    )
    cols[3].markdown(
        _metric_card(
            "Sharpe / Max DD",
            f"{_fmt_number(sharpe.iloc[0], 2) if not sharpe.empty else 'N/A'} / {_fmt_pct(max_dd.iloc[0]) if not max_dd.empty else 'N/A'}",
            "Quick performance read on the latest report set.",
        ),
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.45, 1.0], gap="large")
    with left:
        _panel_header("System Posture", "This front end is now both the run console and the analysis surface. The charts below always reflect saved artifacts on disk.")
        if not backtest_daily.empty:
            frame = backtest_daily.copy()
            frame["benchmark_value"] = (1 + frame["benchmark_return"].fillna(0.0)).cumprod() * 1_000_000
            fig = px.line(
                frame,
                x="date",
                y=["portfolio_value", "benchmark_value"],
                labels={"value": "Value", "variable": "Series"},
                title="Portfolio vs Benchmark",
                template="plotly_white",
            )
            fig.update_layout(height=430, legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)
    with right:
        _panel_header("Attention Queue", "Use these alerts and file timestamps to decide whether you need a fresh pipeline run, a backtest rerun, or only a risk refresh.")
        if risk_alerts.empty:
            st.markdown(_status_pill("No current risk alerts", "success"), unsafe_allow_html=True)
        else:
            for row in risk_alerts.itertuples(index=False):
                st.markdown(_status_pill(f"{row.rule}: {row.value:.4f}", "warning"), unsafe_allow_html=True)
        freshness = pd.DataFrame(_artifact_snapshot(artifacts))
        st.dataframe(freshness[["artifact", "updated", "size"]], use_container_width=True, hide_index=True)


def _render_command_deck() -> None:
    _panel_header("Run Controls", "Adjust runtime dates here, then launch the exact CLI task you want. Long-running jobs stream their log tail into the page.")
    controls_left, controls_right = st.columns([1.1, 1.1], gap="large")

    with controls_left:
        with st.container(border=True):
            st.markdown("#### Monthly Pipeline")
            st.caption("Refresh data, rebuild the research stack, and publish the latest portfolio artifacts.")
            st.date_input("Pipeline date", key="pipeline_date")
            st.date_input("Price ingestion start", key="price_start_date")
            if st.button("Run Monthly Pipeline", type="primary", use_container_width=True):
                _run_cli_task(
                    "Monthly Pipeline",
                    [
                        "run-monthly-pipeline",
                        "--config",
                        st.session_state["config_path"],
                        "--date",
                        str(st.session_state["pipeline_date"]),
                        "--start",
                        str(st.session_state["price_start_date"]),
                    ],
                )

        with st.container(border=True):
            st.markdown("#### Ingestion Only")
            st.caption("Refresh one source family without running downstream research stages.")
            ingest_cols = st.columns(3)
            if ingest_cols[0].button("Prices", use_container_width=True):
                _run_cli_task(
                    "Ingest Prices",
                    [
                        "ingest-prices",
                        "--config",
                        st.session_state["config_path"],
                        "--start",
                        str(st.session_state["price_start_date"]),
                    ],
                )
            if ingest_cols[1].button("Fundamentals", use_container_width=True):
                _run_cli_task("Ingest Fundamentals", ["ingest-fundamentals", "--config", st.session_state["config_path"]])
            if ingest_cols[2].button("Macro", use_container_width=True):
                _run_cli_task("Ingest Macro", ["ingest-macro", "--config", st.session_state["config_path"]])

    with controls_right:
        with st.container(border=True):
            st.markdown("#### Backtest")
            st.caption("Rebuild performance, benchmark, drawdown, and transaction-cost reports.")
            st.date_input("Backtest start", key="backtest_start_date")
            st.date_input("Backtest end", key="backtest_end_date")
            if st.button("Run Backtest", type="primary", use_container_width=True):
                _run_cli_task(
                    "Backtest",
                    [
                        "backtest",
                        "--config",
                        st.session_state["config_path"],
                        "--start",
                        str(st.session_state["backtest_start_date"]),
                        "--end",
                        str(st.session_state["backtest_end_date"]),
                    ],
                )

        with st.container(border=True):
            st.markdown("#### Risk And Stages")
            st.caption("Run a focused risk refresh or resume from a specific monthly pipeline stage.")
            st.date_input("Risk monitor date", key="risk_date")
            if st.button("Run Risk Monitor", type="primary", use_container_width=True):
                _run_cli_task(
                    "Risk Monitor",
                    [
                        "risk-monitor",
                        "--config",
                        st.session_state["config_path"],
                        "--date",
                        str(st.session_state["risk_date"]),
                    ],
                )

            st.markdown("##### Individual stages")
            stage_cols = st.columns(4)
            actions = [
                ("Universe", ["build-universe", "--config", st.session_state["config_path"], "--date", str(st.session_state["pipeline_date"])]),
                ("Factors", ["compute-factors", "--config", st.session_state["config_path"], "--date", str(st.session_state["pipeline_date"])]),
                ("Score", ["score", "--config", st.session_state["config_path"], "--date", str(st.session_state["pipeline_date"])]),
                ("Rebalance", ["rebalance", "--config", st.session_state["config_path"], "--date", str(st.session_state["pipeline_date"])]),
            ]
            for col, (label, args) in zip(stage_cols, actions, strict=False):
                if col.button(label, use_container_width=True):
                    _run_cli_task(label, args)

    _render_last_run()


def _render_configuration(config_obj: Any | None) -> None:
    _panel_header("Configuration Surface", "Edit the bundled research profile directly from the front end. Changes persist to YAML and apply to subsequent runs.")
    if config_obj is None:
        st.error("The selected config entrypoint could not be parsed. Fix that first, or switch back to configs/default.yaml.")
        return
    editor_enabled = st.session_state["config_path"] == DEFAULT_CONFIG_PATH
    if not editor_enabled:
        st.warning("The built-in editor only writes to configs/default.yaml and its included YAML files. Switch the config entrypoint back to configs/default.yaml to enable Save Configuration.")

    if "cfg_backtest_start" not in st.session_state:
        st.session_state["cfg_backtest_start"] = _to_date(config_obj.backtest.start_date, date(2015, 1, 1))
        st.session_state["cfg_initial_capital"] = float(config_obj.backtest.initial_capital)
        st.session_state["cfg_exec_lag"] = int(config_obj.backtest.rebalance_execution_lag_days)
        st.session_state["cfg_commission_bps"] = float(config_obj.backtest.transaction_cost.commission_bps)
        st.session_state["cfg_slippage_bps"] = float(config_obj.backtest.transaction_cost.slippage_bps)
        st.session_state["cfg_benchmark"] = config_obj.strategy.benchmark
        st.session_state["cfg_risk_aversion"] = float(config_obj.optimizer.objective.risk_aversion)
        st.session_state["cfg_turnover_penalty"] = float(config_obj.optimizer.objective.turnover_penalty)
        st.session_state["cfg_max_weight"] = float(config_obj.optimizer.constraints.max_weight)
        st.session_state["cfg_target_holdings"] = int(config_obj.optimizer.constraints.target_num_holdings)
        st.session_state["cfg_max_sector_weight"] = float(config_obj.optimizer.constraints.max_sector_weight)
        st.session_state["cfg_max_turnover"] = float(config_obj.optimizer.constraints.max_turnover_per_rebalance)
        st.session_state["cfg_tracking_error_limit"] = float(
            config_obj.optimizer.constraints.tracking_error_limit or 0.0
        )
        st.session_state["cfg_fallback_top_n"] = int(config_obj.optimizer.fallback.top_n)
        st.session_state["cfg_max_daily_loss"] = float(config_obj.risk_limits.portfolio.max_daily_loss)
        st.session_state["cfg_max_drawdown"] = float(config_obj.risk_limits.portfolio.max_drawdown_alert)
        st.session_state["cfg_max_realized_vol"] = float(config_obj.risk_limits.portfolio.max_realized_vol_annualized)
        st.session_state["cfg_beta_min"] = float(config_obj.risk_limits.portfolio.min_beta_to_benchmark)
        st.session_state["cfg_beta_max"] = float(config_obj.risk_limits.portfolio.max_beta_to_benchmark)
        st.session_state["cfg_risk_tracking_error"] = float(config_obj.risk_limits.portfolio.max_tracking_error)
        st.session_state["cfg_min_market_cap"] = float(config_obj.universe.filters.min_market_cap)
        st.session_state["cfg_min_price"] = float(config_obj.universe.filters.min_price)
        st.session_state["cfg_min_adv_20d"] = float(config_obj.universe.filters.min_adv_20d)
        st.session_state["cfg_exclude_sectors"] = ", ".join(config_obj.universe.exclude.sectors)

    tabs = st.tabs(["Backtest & Costs", "Universe", "Optimizer", "Risk Limits", "Snapshot"])

    with tabs[0]:
        cols = st.columns(3)
        cols[0].date_input("Backtest start", key="cfg_backtest_start")
        cols[1].number_input("Initial capital", min_value=10000.0, step=100000.0, key="cfg_initial_capital")
        cols[2].number_input("Execution lag days", min_value=0, step=1, key="cfg_exec_lag")
        cols = st.columns(3)
        cols[0].text_input("Benchmark", key="cfg_benchmark")
        cols[1].number_input("Commission (bps)", min_value=0.0, step=0.5, key="cfg_commission_bps")
        cols[2].number_input("Slippage (bps)", min_value=0.0, step=0.5, key="cfg_slippage_bps")

    with tabs[1]:
        cols = st.columns(3)
        cols[0].number_input("Min market cap", min_value=0.0, step=100000000.0, key="cfg_min_market_cap")
        cols[1].number_input("Min price", min_value=0.0, step=0.5, key="cfg_min_price")
        cols[2].number_input("Min ADV 20d", min_value=0.0, step=100000.0, key="cfg_min_adv_20d")
        st.text_area("Excluded sectors (comma separated)", key="cfg_exclude_sectors", height=90)

    with tabs[2]:
        cols = st.columns(4)
        cols[0].number_input("Risk aversion", min_value=0.0, step=0.5, key="cfg_risk_aversion")
        cols[1].number_input("Turnover penalty", min_value=0.0, step=0.25, key="cfg_turnover_penalty")
        cols[2].number_input("Max weight", min_value=0.0, max_value=1.0, step=0.01, key="cfg_max_weight", format="%.4f")
        cols[3].number_input("Target holdings", min_value=1, step=1, key="cfg_target_holdings")
        cols = st.columns(3)
        cols[0].number_input("Max sector weight", min_value=0.0, max_value=1.0, step=0.01, key="cfg_max_sector_weight", format="%.4f")
        cols[1].number_input("Max turnover per rebalance", min_value=0.0, max_value=1.0, step=0.01, key="cfg_max_turnover", format="%.4f")
        cols[2].number_input("Tracking error limit", min_value=0.0, max_value=1.0, step=0.01, key="cfg_tracking_error_limit", format="%.4f")
        st.number_input("Fallback top N", min_value=1, step=1, key="cfg_fallback_top_n")

    with tabs[3]:
        cols = st.columns(3)
        cols[0].number_input("Max daily loss", min_value=0.0, max_value=1.0, step=0.01, key="cfg_max_daily_loss", format="%.4f")
        cols[1].number_input("Max drawdown alert", min_value=0.0, max_value=1.0, step=0.01, key="cfg_max_drawdown", format="%.4f")
        cols[2].number_input("Max realized vol", min_value=0.0, max_value=2.0, step=0.01, key="cfg_max_realized_vol", format="%.4f")
        cols = st.columns(3)
        cols[0].number_input("Beta min", min_value=0.0, max_value=3.0, step=0.05, key="cfg_beta_min", format="%.4f")
        cols[1].number_input("Beta max", min_value=0.0, max_value=3.0, step=0.05, key="cfg_beta_max", format="%.4f")
        cols[2].number_input("Risk tracking error", min_value=0.0, max_value=1.0, step=0.01, key="cfg_risk_tracking_error", format="%.4f")

    with tabs[4]:
        snapshot = {
            "benchmark": config_obj.strategy.benchmark,
            "rebalance_frequency": config_obj.strategy.rebalance_frequency,
            "risk_monitor_frequency": config_obj.strategy.risk_monitor_frequency,
            "universe_name": config_obj.universe.name,
            "optimizer_method": config_obj.optimizer.method,
            "config_entrypoint": st.session_state["config_path"],
        }
        st.json(snapshot)

    action_cols = st.columns([1, 1.4, 1.2])
    if action_cols[0].button("Save Configuration", type="primary", use_container_width=True, disabled=not editor_enabled):
        _save_parameter_updates()
    if action_cols[1].button("Reload From Disk", use_container_width=True):
        keys = [key for key in st.session_state if key.startswith("cfg_")]
        for key in keys:
            del st.session_state[key]
        st.cache_data.clear()
        st.rerun()
    action_cols[2].caption("Backups are written automatically before YAML overwrite.")


def _render_universe(artifacts: dict[str, Any]) -> None:
    universe = artifacts["universe"]
    if universe.empty:
        st.info("No universe artifact found yet. Run the monthly pipeline or build-universe first.")
        return

    size = universe.groupby("date")["included"].sum().reset_index(name="universe_size")
    latest_date = universe["date"].max()
    latest = universe.loc[universe["date"] == latest_date].copy()

    col1, col2 = st.columns([1.15, 1.0], gap="large")
    with col1:
        fig = px.area(size, x="date", y="universe_size", title="Universe Size Through Time", template="plotly_white")
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        included = latest.loc[latest["included"]]
        sector_counts = included.groupby("sector", as_index=False)["symbol"].count().rename(columns={"symbol": "count"})
        fig = px.bar(sector_counts, x="sector", y="count", title=f"Sector Mix on {str(latest_date)[:10]}", template="plotly_white")
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)

    filter_col, table_col = st.columns([0.8, 1.4], gap="large")
    with filter_col:
        st.markdown("#### Filters")
        selected_date = st.selectbox(
            "Snapshot date",
            options=sorted(universe["date"].astype(str).unique(), reverse=True),
            index=0,
        )
        include_flag = st.selectbox("Membership", options=["All", "Included", "Excluded"], index=1)
        symbol_query = st.text_input("Symbol contains")
    with table_col:
        view = universe.loc[universe["date"].astype(str) == selected_date].copy()
        if include_flag == "Included":
            view = view.loc[view["included"]]
        elif include_flag == "Excluded":
            view = view.loc[~view["included"]]
        if symbol_query:
            view = view.loc[view["symbol"].str.contains(symbol_query.upper(), na=False)]
        st.dataframe(
            view.sort_values(["included", "symbol"], ascending=[False, True]),
            use_container_width=True,
            hide_index=True,
        )


def _render_factor_lab(artifacts: dict[str, Any]) -> None:
    scores = artifacts["scores"]
    factors = artifacts["factors"]
    if scores.empty or factors.empty:
        st.info("Factor artifacts are not ready yet. Run the pipeline through factor computation and scoring first.")
        return

    available_dates = sorted(scores["date"].astype(str).unique(), reverse=True)
    selected_date = st.selectbox("Factor snapshot date", options=available_dates, index=0)
    selected_scores = scores.loc[scores["date"].astype(str) == selected_date].copy().sort_values("rank")
    selected_factors = factors.loc[factors["date"].astype(str) == selected_date].copy()

    top_symbol = st.selectbox("Symbol drilldown", options=selected_scores["symbol"].tolist(), index=0)
    score_cols = ["value_score", "quality_score", "momentum_score", "low_volatility_score"]

    left, right = st.columns([1.05, 1.15], gap="large")
    with left:
        fig = px.bar(
            selected_scores.head(20),
            x="symbol",
            y="total_score",
            color="sector",
            title=f"Top 20 Alpha Ranks | {selected_date}",
            template="plotly_white",
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(selected_scores.head(30), use_container_width=True, hide_index=True)
    with right:
        mean_group = selected_factors.groupby("factor_group", as_index=False)["processed_value"].mean()
        fig = px.bar(mean_group, x="factor_group", y="processed_value", title="Average Processed Factor by Group", template="plotly_white")
        fig.update_layout(height=260)
        st.plotly_chart(fig, use_container_width=True)

        symbol_scores = selected_scores.loc[selected_scores["symbol"] == top_symbol, score_cols].T.reset_index()
        symbol_scores.columns = ["factor_group", "score"]
        fig = px.bar(symbol_scores, x="factor_group", y="score", title=f"{top_symbol} Group Score Profile", template="plotly_white")
        fig.update_layout(height=260)
        st.plotly_chart(fig, use_container_width=True)

        symbol_factors = selected_factors.loc[selected_factors["symbol"] == top_symbol, ["factor_name", "processed_value", "raw_value", "factor_group"]]
        st.dataframe(symbol_factors.sort_values(["factor_group", "factor_name"]), use_container_width=True, hide_index=True)


def _render_portfolio(artifacts: dict[str, Any]) -> None:
    portfolio = artifacts["portfolio"]
    if portfolio.empty:
        st.info("No portfolio weights are saved yet. Run rebalance or the monthly pipeline first.")
        return

    latest_date = portfolio["date"].max()
    latest = portfolio.loc[portfolio["date"] == latest_date].copy().sort_values("target_weight", ascending=False)
    turnover = latest["trade_weight"].abs().sum()

    cols = st.columns(4)
    cols[0].markdown(_metric_card("Portfolio Date", str(latest_date)[:10], "Latest target portfolio snapshot."), unsafe_allow_html=True)
    cols[1].markdown(_metric_card("Holdings", str(len(latest)), "Target positions in the saved portfolio."), unsafe_allow_html=True)
    cols[2].markdown(_metric_card("Turnover", _fmt_pct(turnover), "Sum of absolute trade weights."), unsafe_allow_html=True)
    cols[3].markdown(_metric_card("Largest Weight", _fmt_pct(latest["target_weight"].max()), "Single-name concentration check."), unsafe_allow_html=True)

    left, right = st.columns([1.15, 1.0], gap="large")
    with left:
        fig = px.bar(
            latest.head(20),
            x="symbol",
            y="target_weight",
            color="sector",
            title=f"Top Holdings | {str(latest_date)[:10]}",
            template="plotly_white",
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(latest, use_container_width=True, hide_index=True)
    with right:
        sector_exposure = latest.groupby("sector", as_index=False)["target_weight"].sum()
        fig = px.bar(sector_exposure, x="sector", y="target_weight", title="Sector Exposure", template="plotly_white")
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)

        trades = latest[["symbol", "previous_weight", "target_weight", "trade_weight"]].copy()
        trades["trade_direction"] = trades["trade_weight"].apply(lambda x: "Buy" if x > 0 else "Sell" if x < 0 else "Hold")
        st.dataframe(trades.sort_values("trade_weight", ascending=False), use_container_width=True, hide_index=True)


def _render_backtest(artifacts: dict[str, Any]) -> None:
    backtest_daily = artifacts["backtest_daily"]
    backtest_metrics = artifacts["backtest_metrics"]
    control, summary = st.columns([0.9, 1.35], gap="large")
    with control:
        _panel_header("Backtest Run", "Adjust the historical window here and run a fresh backtest without leaving the UI.")
        st.date_input("Backtest start", key="backtest_start_date")
        st.date_input("Backtest end", key="backtest_end_date")
        if st.button("Run Backtest", type="primary", use_container_width=True, key="backtest_run_button_page"):
            _run_cli_task(
                "Backtest",
                [
                    "backtest",
                    "--config",
                    st.session_state["config_path"],
                    "--start",
                    str(st.session_state["backtest_start_date"]),
                    "--end",
                    str(st.session_state["backtest_end_date"]),
                ],
            )
        if not backtest_metrics.empty:
            st.dataframe(backtest_metrics, use_container_width=True, hide_index=True)
        else:
            st.info("No backtest report is saved yet.")

    with summary:
        if backtest_daily.empty:
            st.info("Run a backtest to populate equity curve, drawdown, and monthly return views.")
        else:
            frame = backtest_daily.copy()
            frame["benchmark_value"] = (1 + frame["benchmark_return"].fillna(0.0)).cumprod() * 1_000_000
            fig = px.line(frame, x="date", y=["portfolio_value", "benchmark_value"], title="Equity Curve", template="plotly_white")
            fig.update_layout(height=360, legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)

            chart_cols = st.columns(2)
            monthly = (
                frame.assign(month=lambda df: pd.to_datetime(df["date"]).dt.to_period("M").astype(str))
                .groupby("month")["portfolio_return"]
                .agg(lambda values: (1.0 + values).prod() - 1.0)
                .reset_index()
            )
            drawdown_fig = px.line(frame, x="date", y="drawdown", title="Drawdown", template="plotly_white")
            drawdown_fig.update_layout(height=280)
            chart_cols[0].plotly_chart(drawdown_fig, use_container_width=True)
            monthly_fig = px.bar(monthly.tail(24), x="month", y="portfolio_return", title="Last 24 Monthly Returns", template="plotly_white")
            monthly_fig.update_layout(height=280)
            chart_cols[1].plotly_chart(monthly_fig, use_container_width=True)


def _render_risk(artifacts: dict[str, Any]) -> None:
    risk_alerts = artifacts["risk_alerts"]
    portfolio = artifacts["portfolio"]
    backtest_daily = artifacts["backtest_daily"]

    left, right = st.columns([0.9, 1.3], gap="large")
    with left:
        _panel_header("Risk Monitor Run", "Use this panel when you only need a fresh risk pass after an existing portfolio or backtest is already on disk.")
        st.date_input("Risk monitor date", key="risk_date")
        if st.button("Run Risk Monitor", type="primary", use_container_width=True, key="risk_run_button_page"):
            _run_cli_task(
                "Risk Monitor",
                [
                    "risk-monitor",
                    "--config",
                    st.session_state["config_path"],
                    "--date",
                    str(st.session_state["risk_date"]),
                ],
            )
        if risk_alerts.empty:
            st.success("No saved alert rows.")
        else:
            st.dataframe(risk_alerts, use_container_width=True, hide_index=True)

    with right:
        if not portfolio.empty:
            latest_date = portfolio["date"].max()
            latest = portfolio.loc[portfolio["date"] == latest_date].copy()
            exposure = latest.groupby("sector", as_index=False)["target_weight"].sum()
            fig = px.bar(exposure, x="sector", y="target_weight", title="Latest Sector Risk Footprint", template="plotly_white")
            fig.update_layout(height=280)
            st.plotly_chart(fig, use_container_width=True)
        if not backtest_daily.empty:
            rolling = backtest_daily.copy()
            rolling["rolling_vol"] = rolling["portfolio_return"].rolling(60).std() * (252**0.5)
            rolling["rolling_tracking_error"] = (rolling["portfolio_return"] - rolling["benchmark_return"]).rolling(60).std() * (252**0.5)
            fig = px.line(
                rolling,
                x="date",
                y=["rolling_vol", "rolling_tracking_error"],
                title="Rolling Volatility vs Tracking Error",
                template="plotly_white",
            )
            fig.update_layout(height=320, legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)


def _render_macro_data(artifacts: dict[str, Any]) -> None:
    macro = artifacts["macro"]
    quality = artifacts["quality"]
    snapshot = pd.DataFrame(_artifact_snapshot(artifacts))

    top, bottom = st.columns([1.0, 1.1], gap="large")
    with top:
        _panel_header("Macro Regime", "Macro remains explanatory in v0.2: it informs risk context and reporting, but does not change portfolio weights.")
        if macro.empty:
            st.info("No macro regime artifact exists yet.")
        else:
            latest = macro.sort_values("date", ascending=False).head(1)
            row = latest.iloc[0]
            st.markdown(
                _status_pill(f"inflation: {row['inflation_regime']}", "warning" if row["inflation_regime"] == "high" else "info")
                + _status_pill(f"rates: {row['rates_regime']}", "info")
                + _status_pill(f"curve: {row['curve_regime']}", "warning" if row["curve_regime"] == "inverted" else "success")
                + _status_pill(f"credit: {row['credit_regime']}", "warning" if row["credit_regime"] != "calm" else "success"),
                unsafe_allow_html=True,
            )
            st.write(row["risk_summary"])
            st.dataframe(macro.sort_values("date", ascending=False), use_container_width=True, hide_index=True)

    with bottom:
        _panel_header("Data Quality And File Freshness", "This is the operational hygiene view: stale inputs, duplicate checks, and artifact update timestamps live here.")
        st.dataframe(quality if not quality.empty else pd.DataFrame({"message": ["No data quality report generated yet."]}), use_container_width=True, hide_index=True)
        st.dataframe(snapshot[["artifact", "updated", "size"]], use_container_width=True, hide_index=True)


def main() -> None:
    _apply_style()
    artifacts = _load_artifacts()
    _init_state(artifacts)
    config_obj, config_error = _try_load_config(st.session_state["config_path"])

    _render_sidebar(artifacts, config_error)
    _render_hero(
        "Research Workbench",
        "Operate the monthly pipeline, backtest, and risk monitor from the front end, then inspect the resulting universe, factors, portfolio, and control outputs without switching surfaces.",
    )

    pages = [
        "Command Deck",
        "Overview",
        "Configuration",
        "Universe",
        "Factor Lab",
        "Portfolio",
        "Backtest",
        "Risk",
        "Macro & Data",
    ]
    page = st.radio("Navigation", pages, horizontal=True, key="page", label_visibility="collapsed")

    if page == "Command Deck":
        _render_command_deck()
    elif page == "Overview":
        _render_overview(artifacts)
    elif page == "Configuration":
        _render_configuration(config_obj)
    elif page == "Universe":
        _render_universe(artifacts)
    elif page == "Factor Lab":
        _render_factor_lab(artifacts)
    elif page == "Portfolio":
        _render_portfolio(artifacts)
    elif page == "Backtest":
        _render_backtest(artifacts)
    elif page == "Risk":
        _render_risk(artifacts)
    elif page == "Macro & Data":
        _render_macro_data(artifacts)


if __name__ == "__main__":
    main()
