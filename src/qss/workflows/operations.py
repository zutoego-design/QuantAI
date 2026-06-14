from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from qss.config.schema import AppConfig
from qss.data.storage import resolve_path
from qss.experiments.registry import register_run_path
from qss.risk.monitor import run_daily_risk_monitor


@dataclass
class OperationalDay:
    date: str
    status: str
    started_at: str
    finished_at: str
    data_update_status: str
    data_staleness_days: int
    risk_run_id: str
    alert_count: int
    registry_status: str
    retry_count: int
    error: str
    manual_action: str


def _trading_days(
    config: AppConfig,
    end_date: str | pd.Timestamp,
    count: int,
) -> list[pd.Timestamp]:
    prices = pd.read_parquet(
        Path(config.paths.silver_data) / "prices" / "prices_daily.parquet"
    )
    prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None)
    benchmark = prices.loc[
        prices["symbol"] == config.backtest.secondary_benchmark,
        "date",
    ]
    dates = sorted(
        pd.Timestamp(value).normalize()
        for value in benchmark.loc[
            benchmark <= pd.Timestamp(end_date).normalize()
        ].unique()
    )
    if len(dates) < count:
        raise ValueError(
            f"Only {len(dates)} benchmark trading days exist on or before "
            f"{pd.Timestamp(end_date).date()}; {count} are required."
        )
    return dates[-count:]


def _append_day_log(root: Path, result: OperationalDay) -> Path:
    path = root / "daily_log" / f"{result.date}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"## Attempt {result.started_at}",
        "",
        f"- Status: `{result.status}`",
        f"- Data update check: `{result.data_update_status}`",
        f"- Data staleness: `{result.data_staleness_days}` days",
        f"- Risk run: `{result.risk_run_id or 'none'}`",
        f"- Alerts: `{result.alert_count}`",
        f"- Registry refresh: `{result.registry_status}`",
        f"- Retry count: `{result.retry_count}`",
        f"- Error: `{result.error or 'none'}`",
        f"- Manual action: `{result.manual_action}`",
        f"- Finished at: `{result.finished_at}`",
        "",
    ]
    if path.exists():
        prior = path.read_text(encoding="utf-8").rstrip()
        content = prior + "\n\n" + "\n".join(lines)
    else:
        content = f"# Operations Log: {result.date}\n\n" + "\n".join(lines)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")
    return path


def run_operations_dry_run(
    config: AppConfig,
    *,
    end_date: str | pd.Timestamp,
    trading_days: int = 10,
    max_attempts: int = 2,
) -> tuple[pd.DataFrame, Path]:
    if trading_days < 1:
        raise ValueError("trading_days must be positive.")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive.")
    dates = _trading_days(config, end_date, trading_days)
    prices = pd.read_parquet(
        Path(config.paths.silver_data) / "prices" / "prices_daily.parquet"
    )
    prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None)
    root = resolve_path(config.paths.reports) / "operations"
    rows: list[OperationalDay] = []

    for date in dates:
        started = pd.Timestamp.now(tz="UTC")
        market_date = prices.loc[prices["date"] <= date, "date"].max()
        staleness = (
            int((date - pd.Timestamp(market_date)).days)
            if pd.notna(market_date)
            else -1
        )
        data_status = "valid" if staleness == 0 else "stale"
        risk_run_id = ""
        alerts = 0
        registry_status = "not_run"
        error = ""
        retries = 0
        status = "invalid"
        manual_action = "investigate"

        for attempt in range(1, max_attempts + 1):
            try:
                if data_status != "valid":
                    raise ValueError(
                        f"Cached market data is stale by {staleness} days."
                    )
                risk = run_daily_risk_monitor(date, config)
                risk_run_id = risk.run_id
                alerts = len(risk.alerts)
                registered = register_run_path(config, risk.run_path)
                registry_status = "valid" if registered else "invalid"
                if not registered:
                    raise ValueError("Incremental registry refresh failed.")
                status = "valid"
                manual_action = "none"
                break
            except Exception as exc:
                error = str(exc)
                retries = attempt - 1
                if attempt == max_attempts:
                    break
        finished = pd.Timestamp.now(tz="UTC")
        result = OperationalDay(
            date=str(date.date()),
            status=status,
            started_at=started.isoformat(),
            finished_at=finished.isoformat(),
            data_update_status=data_status,
            data_staleness_days=staleness,
            risk_run_id=risk_run_id,
            alert_count=alerts,
            registry_status=registry_status,
            retry_count=retries,
            error=error,
            manual_action=manual_action,
        )
        rows.append(result)
        _append_day_log(root, result)

    summary = pd.DataFrame([asdict(row) for row in rows])
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "ten_day_summary.csv"
    summary.to_csv(csv_path, index=False)
    markdown_path = root / "ten_day_summary.md"
    markdown_path.write_text(
        "\n".join(
            [
                "# Ten-Day Operations Dry Run",
                "",
                f"- Window: `{summary['date'].min()}` to `{summary['date'].max()}`",
                f"- Trading days: `{len(summary)}`",
                f"- Valid days: `{int(summary['status'].eq('valid').sum())}`",
                f"- Total alerts: `{int(summary['alert_count'].sum())}`",
                f"- Failed days: `{int(summary['status'].ne('valid').sum())}`",
                "",
                "Each daily log preserves prior attempts by appending a new section.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return summary, markdown_path
