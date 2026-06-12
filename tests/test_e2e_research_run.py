import numpy as np
import pandas as pd

from qss.acceptance import run_acceptance_checks
from qss.backtest.engine import run_backtest
from qss.config.loader import get_config


def _write_fixture(tmp_path):
    config = get_config(["configs/default.yaml"])
    config.paths.raw_data = str(tmp_path / "raw")
    config.paths.silver_data = str(tmp_path / "silver")
    config.paths.gold_data = str(tmp_path / "gold")
    config.paths.reports = str(tmp_path / "reports")
    config.universe.filters.min_market_cap = 1
    config.universe.filters.min_price = 1
    config.universe.filters.min_adv_20d = 1
    config.universe.filters.min_history_days = 20
    config.universe.filters.min_price_data_completeness = 0.8
    config.optimizer.constraints.target_num_holdings = 2
    config.optimizer.constraints.max_weight = 0.6
    config.optimizer.constraints.max_sector_weight = 0.8
    config.optimizer.constraints.max_turnover_per_rebalance = 2.0
    config.optimizer.fallback.top_n = 2
    config.backtest.initial_capital = 100_000
    config.backtest.transaction_cost.market_impact_coefficient = 0

    silver = tmp_path / "silver"
    (silver / "prices").mkdir(parents=True)
    (silver / "fundamentals").mkdir(parents=True)
    (silver / "universe").mkdir(parents=True)

    dates = pd.bdate_range("2023-12-01", "2025-04-04")
    price_rows = []
    for index, symbol in enumerate(["AAA", "BBB", "CCC", "^IXIC", "QQQ"], start=1):
        trend = 0.00015 * index
        returns = trend + 0.002 * np.sin(np.arange(len(dates)) / (7 + index))
        closes = 50 * np.cumprod(1 + returns)
        for offset, date in enumerate(dates):
            price_rows.append(
                {
                    "symbol": symbol,
                    "date": date,
                    "open": closes[offset],
                    "high": closes[offset] * 1.01,
                    "low": closes[offset] * 0.99,
                    "close": closes[offset],
                    "adj_close": closes[offset],
                    "volume": 5_000_000,
                    "return_1d": np.nan if offset == 0 else returns[offset],
                    "source": "fixture_live",
                    "quality_status": "live",
                    "ingestion_time": pd.Timestamp("2025-04-05"),
                }
            )
    pd.DataFrame(price_rows).to_parquet(
        silver / "prices" / "prices_daily.parquet", index=False
    )

    metrics = {
        "revenue": 1_000_000_000,
        "gross_profit": 600_000_000,
        "operating_income": 200_000_000,
        "net_income": 150_000_000,
        "total_assets": 2_000_000_000,
        "total_liabilities": 800_000_000,
        "shareholders_equity": 1_200_000_000,
        "operating_cash_flow": 180_000_000,
        "capital_expenditure": 30_000_000,
        "shares_outstanding": 100_000_000,
    }
    observation_rows = []
    for index, symbol in enumerate(["AAA", "BBB", "CCC"], start=1):
        for metric, value in metrics.items():
            observation_rows.append(
                {
                    "symbol": symbol,
                    "metric": metric,
                    "value": value * (1 + index * 0.05),
                    "unit": "shares" if metric == "shares_outstanding" else "USD",
                    "period_end_date": pd.Timestamp("2024-09-30"),
                    "filing_date": pd.Timestamp("2024-11-01"),
                    "available_date": pd.Timestamp("2024-11-01"),
                    "fiscal_year": 2024,
                    "fiscal_period": "Q3",
                    "form": "10-Q",
                    "accession": f"fixture-{symbol}-{metric}",
                    "source": "sec_edgar",
                    "quality_status": "live",
                    "ingestion_time": pd.Timestamp("2025-04-05"),
                }
            )
    pd.DataFrame(observation_rows).to_parquet(
        silver / "fundamentals" / "fundamental_observations.parquet", index=False
    )

    master = pd.DataFrame(
        {
            "security_id": ["sec_a", "sec_b", "sec_c"],
            "symbol": ["AAA", "BBB", "CCC"],
            "name": ["AAA Inc", "BBB Inc", "CCC Inc"],
            "exchange": "XNAS",
            "security_type": "Common Stock",
            "sector": ["Technology", "Health Care", "Industrials"],
            "source": "fixture",
        }
    )
    master.to_parquet(silver / "universe" / "security_master.parquet", index=False)
    membership_rows = []
    for date in pd.to_datetime(
        ["2024-12-31", "2025-01-31", "2025-02-28", "2025-03-31"]
    ):
        for row in master.itertuples(index=False):
            membership_rows.append(
                {
                    "date": date,
                    "security_id": row.security_id,
                    "symbol": row.symbol,
                    "security_type": row.security_type,
                    "included": True,
                    "exclusion_reason": "",
                    "source": "fixture",
                }
            )
    pd.DataFrame(membership_rows).to_parquet(
        silver / "universe" / "universe_membership.parquet", index=False
    )
    return config


def test_deterministic_end_to_end_research_run(tmp_path):
    config = _write_fixture(tmp_path)
    first = run_backtest(
        "2025-01-01", "2025-03-31", config, enforce_data_gate=False
    )
    second = run_backtest(
        "2025-01-01", "2025-03-31", config, enforce_data_gate=False
    )
    assert first.run_id != second.run_id
    pd.testing.assert_frame_equal(
        first.daily_returns.reset_index(drop=True),
        second.daily_returns.reset_index(drop=True),
    )
    assert set(first.rebalances["holding_count"]) == {2}
    assert (first.run_path / "report.html").exists()
    assert (first.run_path / "report.json").exists()
    assert (first.run_path / "factor_diagnostics.csv").exists()
    assert (first.run_path / "data_diagnostics.csv").exists()
    checks, acceptance_context = run_acceptance_checks(config, first.run_path)
    assert bool(checks["passed"].all())
    assert acceptance_context.manifest.status == "valid"
