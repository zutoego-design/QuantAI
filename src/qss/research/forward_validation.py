from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qss.backtest.engine import (
    BacktestRunSpec,
    _attach_reference_benchmarks,
    _prepare_ledger_market_data,
    _simulate_ledger,
    _target_schedule,
    load_backtest_data,
)
from qss.backtest.metrics import compute_backtest_metrics
from qss.config.schema import AppConfig
from qss.data.storage import resolve_path, write_csv
from qss.research.snapshot import build_data_snapshot, write_data_snapshot
from qss.runs.manifest import code_version, config_hash

FORWARD_SCOPE = "six_month_forward_validation"


@dataclass
class ForwardValidationResult:
    study_id: str
    root: Path
    status: str


def _implementation_hash() -> str:
    root = resolve_path(".")
    paths = sorted((root / "src" / "qss").rglob("*.py"))
    paths.extend(
        path
        for path in [root / "pyproject.toml"]
        if path.exists()
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _ledger_path(root: Path) -> Path:
    return root / "ledger.json"


def _load_ledger(root: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved = resolve_path(root)
    path = _ledger_path(resolved)
    if not path.exists():
        raise ValueError(f"Forward validation ledger does not exist: {path}")
    return resolved, json.loads(path.read_text(encoding="utf-8"))


def _strategy_config(root: Path, strategy_id: str) -> AppConfig:
    path = root / "frozen_configs" / f"{strategy_id}.json"
    if not path.exists():
        raise ValueError(f"Frozen strategy config is missing: {path}")
    return AppConfig.model_validate_json(path.read_text(encoding="utf-8"))


def _write_status(root: Path, status: dict[str, Any]) -> None:
    (root / "status.json").write_text(
        json.dumps(status, indent=2, default=str),
        encoding="utf-8",
    )
    lines = [
        "# Six-Month Forward Validation",
        "",
        f"- State: `{status['state']}`",
        f"- As of: `{status.get('as_of_date') or 'not started'}`",
        f"- Hard failure: `{status['hard_failure']}`",
        f"- Continue conditions met: `{status['continue_conditions_met']}/4`",
        f"- Invalid data days: `{status['invalid_data_days']}`",
        "",
    ]
    for reason in status.get("reasons", []):
        lines.append(f"- {reason}")
    (root / "status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def initialize_forward_validation(
    base_config: AppConfig,
    replay_run: str | Path,
    *,
    start_date: str = "2026-06-15",
    end_date: str = "2026-12-15",
    study_id: str | None = None,
) -> ForwardValidationResult:
    replay_root = resolve_path(replay_run)
    selection_path = replay_root / "selection.json"
    if not selection_path.exists():
        raise ValueError(f"Historical replay selection is missing: {selection_path}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    forward_strategy = selection.get("forward_strategy_id")
    if not forward_strategy:
        raise ValueError("Historical replay did not produce a forward challenger.")
    control_strategy = str(selection.get("control_strategy_id", "v1_control"))
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if end <= start:
        raise ValueError("Forward validation end_date must be after start_date.")

    identifier = study_id or f"multifactor-v2-forward-{start:%Y%m%d}"
    root = resolve_path(base_config.paths.reports) / "forward_validation" / identifier
    root.mkdir(parents=True, exist_ok=True)
    frozen_root = root / "frozen_configs"
    frozen_root.mkdir(parents=True, exist_ok=True)
    strategies = []
    for strategy_id, role in [
        (control_strategy, "control"),
        (str(forward_strategy), "challenger"),
    ]:
        source = replay_root / "candidates" / strategy_id / "resolved_config.json"
        if not source.exists():
            raise ValueError(f"Replay candidate config is missing: {source}")
        config = AppConfig.model_validate_json(source.read_text(encoding="utf-8"))
        target = frozen_root / f"{strategy_id}.json"
        content = config.model_dump_json(indent=2)
        if target.exists() and target.read_text(encoding="utf-8") != content:
            raise ValueError(f"Frozen config already exists with different content: {target}")
        target.write_text(content, encoding="utf-8")
        strategies.append(
            {
                "strategy_id": strategy_id,
                "role": role,
                "config_hash": config_hash(config),
                "frozen_config": str(target),
            }
        )

    ledger = {
        "schema_version": "1.0",
        "study_id": identifier,
        "evaluation_scope": FORWARD_SCOPE,
        "source_replay_run_id": replay_root.name,
        "source_replay_path": str(replay_root),
        "source_replay_decision": selection.get("decision"),
        "start_date": str(start.date()),
        "end_date": str(end.date()),
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "code_version": code_version(),
        "implementation_hash": _implementation_hash(),
        "immutable": True,
        "strategies": strategies,
        "benchmark": "SPY",
        "decision_policy": {
            "maximum_drawdown": -0.20,
            "minimum_cumulative_return": -0.15,
            "maximum_invalid_data_days": 5,
            "minimum_continue_conditions": 2,
        },
    }
    ledger_path = _ledger_path(root)
    serialized = json.dumps(ledger, indent=2)
    if ledger_path.exists():
        existing = json.loads(ledger_path.read_text(encoding="utf-8"))
        comparable_keys = {
            "study_id",
            "source_replay_run_id",
            "start_date",
            "end_date",
            "strategies",
        }
        if any(existing.get(key) != ledger.get(key) for key in comparable_keys):
            raise ValueError("Forward validation ledger already exists with different identity.")
    else:
        ledger_path.write_text(serialized, encoding="utf-8")

    snapshot = build_data_snapshot(base_config)
    write_data_snapshot(snapshot, root / "initial_data_snapshot.json")
    checklist = [
        "# Forward Validation Initialization",
        "",
        f"- Study: `{identifier}`",
        f"- Window: `{start.date()}` to `{end.date()}`",
        f"- Control: `{control_strategy}`",
        f"- Challenger: `{forward_strategy}`",
        "- Benchmark: `SPY`",
        "- Broker connectivity: `disabled`",
        "- Frozen config integrity: `required`",
        "- Daily records: `idempotent by date and strategy`",
        "- Six-month outcome: `continue`, `eliminated`, or `inconclusive`",
        "",
        "The six-month outcome is not a DSR/FDR confirmation or live-trading approval.",
        "",
    ]
    (root / "initialization_checklist.md").write_text(
        "\n".join(checklist),
        encoding="utf-8",
    )
    status = evaluate_forward_validation(root)
    return ForwardValidationResult(identifier, root, status["state"])


def _data_status(cache, as_of_date: pd.Timestamp, benchmark: str) -> tuple[bool, str, str]:
    prices = cache.prices.copy()
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    benchmark_dates = prices.loc[prices["symbol"] == benchmark, "date"]
    latest = benchmark_dates.max() if not benchmark_dates.empty else pd.NaT
    if pd.isna(latest):
        return False, "benchmark_price_history_missing", ""
    if as_of_date not in set(benchmark_dates):
        return False, f"benchmark_data_missing_for_{as_of_date.date()}", str(latest.date())
    membership = cache.membership.copy()
    if membership.empty or "date" not in membership:
        return False, "point_in_time_membership_missing", str(latest.date())
    membership["date"] = pd.to_datetime(membership["date"]).dt.normalize()
    available = membership.loc[membership["date"] <= as_of_date, "date"]
    if available.empty or (as_of_date - available.max()).days > 35:
        return False, "point_in_time_membership_stale", str(latest.date())
    if "source" in prices and prices["source"].astype(str).str.contains(
        "synthetic",
        case=False,
    ).any():
        return False, "synthetic_prices_detected", str(latest.date())
    return True, "", str(latest.date())


def _cash_evaluation(
    cache,
    config: AppConfig,
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prices = cache.prices.copy()
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    dates = sorted(
        prices.loc[
            (prices["symbol"] == config.backtest.secondary_benchmark)
            & pd.to_datetime(prices["date"]).between(
                pd.Timestamp(start_date),
                pd.Timestamp(end_date),
            ),
            "date",
        ].unique()
    )
    spy = (
        prices.loc[
            (prices["symbol"] == config.backtest.secondary_benchmark)
            & prices["date"].isin(dates),
            ["date", "return_1d"],
        ]
        .drop_duplicates("date")
        .set_index("date")["return_1d"]
    )
    daily = pd.DataFrame(
        {
            "date": dates,
            "portfolio_return": 0.0,
            "gross_return": 0.0,
            "benchmark_return": spy.reindex(dates).fillna(0.0).to_numpy(),
            "secondary_benchmark_return": spy.reindex(dates).fillna(0.0).to_numpy(),
            "portfolio_value": config.backtest.initial_capital,
            "transaction_cost": 0.0,
        }
    )
    return daily, pd.DataFrame(), pd.DataFrame()


def _evaluate_strategy(
    cache,
    config: AppConfig,
    start_date: str,
    as_of_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if cache.market_data is None:
        cache.market_data = _prepare_ledger_market_data(cache.prices)
    namespace = cache.factor_snapshots.setdefault(config_hash(config), {})
    targets = _target_schedule(
        start_date,
        as_of_date,
        config,
        cache.prices,
        cache.fundamentals,
        cache.security_master,
        cache.membership,
        filings=cache.filings,
        factor_snapshots=namespace,
        exact_target_count=False,
        market_data=cache.market_data,
    )
    if not targets:
        return _cash_evaluation(cache, config, start_date, as_of_date)
    daily, rebalances, holdings, _, _ = _simulate_ledger(
        BacktestRunSpec(
            start_date=start_date,
            end_date=as_of_date,
            initial_capital=config.backtest.initial_capital,
            execution_lag_days=config.backtest.rebalance_execution_lag_days,
            delisting_return=0.0,
        ),
        config,
        cache.prices,
        targets,
        config.backtest.primary_benchmark,
        market_data=cache.market_data,
    )
    daily = _attach_reference_benchmarks(
        daily,
        cache.prices,
        targets,
        config.backtest.secondary_benchmark,
    )
    daily["benchmark_return"] = daily["secondary_benchmark_return"]
    return daily, rebalances, holdings


def _record_frame(
    strategy_id: str,
    role: str,
    config: AppConfig,
    daily: pd.DataFrame,
    rebalances: pd.DataFrame,
    as_of_date: pd.Timestamp,
    data_valid: bool,
    integrity_valid: bool,
    source_data_date: str,
    invalid_reason: str,
) -> dict[str, Any]:
    metrics = compute_backtest_metrics(daily, rebalances)
    values = {
        str(row.metric): float(row.value)
        for row in metrics.itertuples(index=False)
    }
    last = daily.sort_values("date").iloc[-1] if not daily.empty else None
    return {
        "date": str(as_of_date.date()),
        "strategy_id": strategy_id,
        "role": role,
        "config_hash": config_hash(config),
        "data_valid": data_valid,
        "integrity_valid": integrity_valid,
        "source_data_date": source_data_date,
        "invalid_reason": invalid_reason,
        "daily_return": float(last["portfolio_return"]) if last is not None else np.nan,
        "nav": float(last["portfolio_value"]) if last is not None else np.nan,
        "cumulative_return": values.get("net_total_return"),
        "sharpe_ratio": values.get("sharpe_ratio"),
        "max_drawdown": values.get("max_drawdown"),
        "average_turnover": values.get("average_turnover"),
    }


def _spy_record(
    cache,
    start_date: str,
    as_of_date: pd.Timestamp,
    data_valid: bool,
    source_data_date: str,
    invalid_reason: str,
) -> dict[str, Any]:
    prices = cache.prices.copy()
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    returns = (
        prices.loc[
            (prices["symbol"] == "SPY")
            & prices["date"].between(pd.Timestamp(start_date), as_of_date),
            ["date", "return_1d"],
        ]
        .drop_duplicates("date")
        .sort_values("date")
    )
    values = pd.to_numeric(returns["return_1d"], errors="coerce").fillna(0.0)
    cumulative = float((1.0 + values).prod() - 1.0) if not values.empty else 0.0
    wealth = (1.0 + values).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0 if not wealth.empty else pd.Series([0.0])
    sharpe = (
        float(values.mean() / values.std(ddof=0) * np.sqrt(252))
        if not values.empty and values.std(ddof=0)
        else 0.0
    )
    return {
        "date": str(as_of_date.date()),
        "strategy_id": "SPY",
        "role": "benchmark",
        "config_hash": "",
        "data_valid": data_valid,
        "integrity_valid": True,
        "source_data_date": source_data_date,
        "invalid_reason": invalid_reason,
        "daily_return": float(values.iloc[-1]) if not values.empty else np.nan,
        "nav": 1_000_000.0 * (1.0 + cumulative),
        "cumulative_return": cumulative,
        "sharpe_ratio": sharpe,
        "max_drawdown": float(drawdown.min()),
        "average_turnover": 0.0,
    }


def record_forward_day(
    forward_root: str | Path,
    as_of_date: str | pd.Timestamp,
) -> ForwardValidationResult:
    root, ledger = _load_ledger(forward_root)
    date = pd.Timestamp(as_of_date).normalize()
    start = pd.Timestamp(ledger["start_date"])
    end = pd.Timestamp(ledger["end_date"])
    if date < start or date > end:
        raise ValueError(f"Forward record date must be between {start.date()} and {end.date()}.")
    strategies = ledger["strategies"]
    configs = {
        item["strategy_id"]: _strategy_config(root, item["strategy_id"])
        for item in strategies
    }
    config_integrity = all(
        config_hash(configs[item["strategy_id"]]) == item["config_hash"]
        for item in strategies
    )
    implementation_integrity = (
        _implementation_hash() == ledger["implementation_hash"]
        and code_version() == ledger["code_version"]
    )
    integrity_valid = config_integrity and implementation_integrity
    cache = load_backtest_data(next(iter(configs.values())))
    data_valid, invalid_reason, source_data_date = _data_status(cache, date, "SPY")

    records: list[dict[str, Any]] = []
    if data_valid:
        for item in strategies:
            strategy_id = item["strategy_id"]
            config = configs[strategy_id]
            try:
                daily, rebalances, holdings = _evaluate_strategy(
                    cache,
                    config,
                    ledger["start_date"],
                    str(date.date()),
                )
                record = _record_frame(
                    strategy_id,
                    item["role"],
                    config,
                    daily,
                    rebalances,
                    date,
                    True,
                    integrity_valid,
                    source_data_date,
                    "" if integrity_valid else "frozen_version_integrity_failed",
                )
                latest_holdings = pd.DataFrame()
                if not holdings.empty:
                    holding_dates = pd.to_datetime(holdings["date"]).dt.normalize()
                    latest_holdings = holdings.loc[
                        holding_dates == holding_dates.max()
                    ].copy()
                holding_path = root / "holdings" / str(date.date()) / f"{strategy_id}.csv"
                write_csv(latest_holdings, holding_path)
            except (ValueError, RuntimeError) as exc:
                record = _record_frame(
                    strategy_id,
                    item["role"],
                    config,
                    pd.DataFrame(),
                    pd.DataFrame(),
                    date,
                    False,
                    integrity_valid,
                    source_data_date,
                    str(exc),
                )
            records.append(record)
    else:
        for item in strategies:
            strategy_id = item["strategy_id"]
            records.append(
                _record_frame(
                    strategy_id,
                    item["role"],
                    configs[strategy_id],
                    pd.DataFrame(),
                    pd.DataFrame(),
                    date,
                    False,
                    integrity_valid,
                    source_data_date,
                    invalid_reason,
                )
            )
    records.append(
        _spy_record(
            cache,
            ledger["start_date"],
            date,
            data_valid,
            source_data_date,
            invalid_reason,
        )
    )
    new = pd.DataFrame(records)
    record_path = root / "daily_records.csv"
    existing = (
        pd.read_csv(record_path)
        if record_path.exists()
        else pd.DataFrame(columns=new.columns)
    )
    if not existing.empty:
        existing = existing.loc[
            pd.to_datetime(existing["date"]).dt.normalize() != date
        ].copy()
    combined = pd.concat([existing, new], ignore_index=True)
    combined = combined.sort_values(["date", "role", "strategy_id"])
    write_csv(combined, record_path)
    status = evaluate_forward_validation(root)
    return ForwardValidationResult(ledger["study_id"], root, status["state"])


def evaluate_forward_validation(forward_root: str | Path) -> dict[str, Any]:
    root, ledger = _load_ledger(forward_root)
    record_path = root / "daily_records.csv"
    records = pd.read_csv(record_path) if record_path.exists() else pd.DataFrame()
    policy = ledger["decision_policy"]
    reasons: list[str] = []
    if records.empty:
        status = {
            "state": "initialized",
            "as_of_date": None,
            "hard_failure": False,
            "continue_conditions_met": 0,
            "invalid_data_days": 0,
            "reasons": [],
        }
        _write_status(root, status)
        return status

    records["date"] = pd.to_datetime(records["date"]).dt.normalize()
    as_of = records["date"].max()
    invalid_days = int(
        records.loc[records["role"] != "benchmark"]
        .groupby("date")["data_valid"]
        .apply(lambda values: not bool(values.astype(bool).all()))
        .sum()
    )
    integrity_failed = not bool(
        records.loc[records["role"] != "benchmark", "integrity_valid"]
        .fillna(False)
        .astype(bool)
        .all()
    )
    latest = records.loc[records["date"] == as_of].set_index("role")
    challenger = latest.loc["challenger"] if "challenger" in latest.index else None
    control = latest.loc["control"] if "control" in latest.index else None
    benchmark = latest.loc["benchmark"] if "benchmark" in latest.index else None
    hard_failure = False
    if integrity_failed:
        hard_failure = True
        reasons.append("Frozen configuration or implementation integrity failed.")
    if invalid_days > int(policy["maximum_invalid_data_days"]):
        hard_failure = True
        reasons.append("Invalid data days exceeded the allowed maximum.")
    if challenger is not None and pd.notna(challenger["max_drawdown"]):
        if float(challenger["max_drawdown"]) < float(policy["maximum_drawdown"]):
            hard_failure = True
            reasons.append("Challenger drawdown exceeded 20%.")
    if challenger is not None and pd.notna(challenger["cumulative_return"]):
        if float(challenger["cumulative_return"]) < float(
            policy["minimum_cumulative_return"]
        ):
            hard_failure = True
            reasons.append("Challenger cumulative loss exceeded 15%.")

    conditions = {
        "positive_return": bool(
            challenger is not None
            and pd.notna(challenger["cumulative_return"])
            and float(challenger["cumulative_return"]) > 0
        ),
        "outperformed_spy": bool(
            challenger is not None
            and benchmark is not None
            and pd.notna(challenger["cumulative_return"])
            and pd.notna(benchmark["cumulative_return"])
            and float(challenger["cumulative_return"])
            > float(benchmark["cumulative_return"])
        ),
        "sharpe_above_v1": bool(
            challenger is not None
            and control is not None
            and pd.notna(challenger["sharpe_ratio"])
            and pd.notna(control["sharpe_ratio"])
            and float(challenger["sharpe_ratio"]) > float(control["sharpe_ratio"])
        ),
        "drawdown_below_v1": bool(
            challenger is not None
            and control is not None
            and pd.notna(challenger["max_drawdown"])
            and pd.notna(control["max_drawdown"])
            and float(challenger["max_drawdown"]) > float(control["max_drawdown"])
        ),
    }
    met = sum(conditions.values())
    completed = as_of >= pd.Timestamp(ledger["end_date"])
    if hard_failure:
        state = "eliminated"
    elif completed and met >= int(policy["minimum_continue_conditions"]):
        state = "continue_validation"
    elif completed:
        state = "inconclusive"
    else:
        state = "monitoring"
    status = {
        "state": state,
        "as_of_date": str(as_of.date()),
        "hard_failure": hard_failure,
        "continue_conditions_met": met,
        "continue_conditions": conditions,
        "invalid_data_days": invalid_days,
        "reasons": reasons,
        "completed": completed,
    }
    _write_status(root, status)
    return status
