import numpy as np
import pandas as pd
import pytest

from qss.backtest.engine import BacktestRunSpec, _simulate_ledger
from qss.backtest.metrics import compounded_monthly_returns, drawdown_episodes
from qss.config.loader import get_config
from qss.data.calendar import next_trading_day
from qss.portfolio.optimizer import build_equal_weight_portfolio


def _prices(a_returns):
    dates = pd.date_range("2025-01-01", periods=len(a_returns), freq="D")
    frames = []
    for symbol, returns in {
        "A": a_returns,
        "B": [np.nan] + [0.0] * (len(dates) - 1),
        "^IXIC": [np.nan] + [0.01] * (len(dates) - 1),
    }.items():
        close = pd.Series(100.0, index=dates) * pd.Series(
            [1.0, *np.cumprod(1 + pd.Series(returns[1:]).fillna(0.0)).tolist()],
            index=dates,
        )
        frames.append(
            pd.DataFrame(
                {
                    "symbol": symbol,
                    "date": dates,
                    "adj_close": close.values,
                    "volume": 10_000_000,
                    "return_1d": returns,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _targets():
    weights = pd.Series({"A": 0.5, "B": 0.5})
    universe = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "market_cap": [1e9, 1e9],
            "sector": ["Tech", "Health"],
            "included": [True, True],
        }
    )
    return {
        pd.Timestamp("2025-01-01"): {
            "signal_date": pd.Timestamp("2024-12-31"),
            "weights": weights,
            "sectors": pd.Series({"A": "Tech", "B": "Health"}),
            "optimizer_status": "fixture",
            "warning": None,
            "factors": pd.DataFrame(),
            "universe": universe,
        }
    }


def test_ledger_weights_drift_with_market_returns():
    config = get_config(["configs/default.yaml"])
    config.runtime.research_mode = False
    config.backtest.transaction_cost.commission_bps = 0
    config.backtest.transaction_cost.slippage_bps = 0
    config.backtest.transaction_cost.market_impact_coefficient = 0
    spec = BacktestRunSpec(
        start_date="2025-01-01",
        end_date="2025-01-03",
        initial_capital=1000,
        delisting_return=0,
    )
    daily, _, holdings, trades, attribution = _simulate_ledger(
        spec, config, _prices([np.nan, 0.10, 0.0]), _targets(), "^IXIC"
    )
    day_two = holdings.loc[holdings["date"] == pd.Timestamp("2025-01-02")].set_index("symbol")
    assert day_two.loc["A", "weight"] > 0.5
    assert len(trades) == 2
    assert set(trades["execution_price_model"]) == {"close"}
    assert set(daily["execution_price_model"]) == {"close"}
    attributed = (
        attribution.groupby("date")["portfolio_contribution"]
        .sum()
        .reindex(pd.to_datetime(daily["date"]), fill_value=0.0)
        .to_numpy()
    )
    assert np.allclose(attributed, daily["portfolio_return"])


def test_research_mode_rejects_missing_intermediate_return():
    config = get_config(["configs/default.yaml"])
    config.backtest.transaction_cost.max_adv_participation = float("inf")
    prices = _prices([np.nan, np.nan, 0.01])
    prices.loc[
        (prices["symbol"] == "^IXIC")
        & (prices["date"] == pd.Timestamp("2025-01-01")),
        "return_1d",
    ] = 0.0
    spec = BacktestRunSpec(
        start_date="2025-01-01",
        end_date="2025-01-03",
        initial_capital=1000,
    )
    with pytest.raises(ValueError, match="Missing return"):
        _simulate_ledger(spec, config, prices, _targets(), "^IXIC")


def test_quickstart_mode_fills_missing_intermediate_return_with_zero():
    config = get_config(["configs/default.yaml"])
    config.runtime.research_mode = False
    prices = _prices([np.nan, np.nan, 0.01])
    spec = BacktestRunSpec(
        start_date="2025-01-01",
        end_date="2025-01-03",
        initial_capital=1000,
    )
    daily, _, _, _, _ = _simulate_ledger(
        spec,
        config,
        prices,
        _targets(),
        "^IXIC",
    )
    assert daily["missing_return_fills"].sum() == 1


def test_monthly_returns_are_compounded_and_drawdowns_have_episodes():
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02", "2025-01-03", "2025-02-03"]),
            "portfolio_return": [0.10, -0.10, 0.05],
            "benchmark_return": [0.0, 0.0, 0.0],
        }
    )
    monthly = compounded_monthly_returns(daily).set_index("month")
    assert monthly.loc["2025-01", "portfolio_return"] == pytest.approx(-0.01)
    episodes = drawdown_episodes(daily["portfolio_return"], daily["date"])
    assert not episodes.empty
    assert episodes.iloc[0]["max_drawdown"] < 0


def test_execution_lag_zero_uses_signal_day():
    dates = pd.DatetimeIndex(pd.to_datetime(["2025-01-02", "2025-01-03"]))
    assert next_trading_day(pd.Timestamp("2025-01-02"), dates, 0) == pd.Timestamp(
        "2025-01-02"
    )


def test_cash_interest_is_accrued_explicitly():
    config = get_config(["configs/default.yaml"])
    config.runtime.research_mode = False
    config.backtest.cash_interest_annual_rate = 0.05
    config.backtest.transaction_cost.commission_bps = 0
    config.backtest.transaction_cost.slippage_bps = 0
    config.backtest.transaction_cost.market_impact_coefficient = 0
    prices = _prices([np.nan, 0.0])
    spec = BacktestRunSpec(
        start_date="2025-01-01",
        end_date="2025-01-02",
        initial_capital=1000,
    )

    daily, _, _, _, _ = _simulate_ledger(
        spec,
        config,
        prices,
        {},
        "^IXIC",
    )

    assert daily["cash_interest"].gt(0).all()
    assert daily.iloc[-1]["portfolio_value"] > 1000


def test_equal_weight_top_n_respects_sector_limit():
    scores = pd.DataFrame(
        {
            "symbol": [f"A{i}" for i in range(8)] + [f"B{i}" for i in range(4)],
            "sector": ["A"] * 8 + ["B"] * 4,
            "total_score": list(range(12, 0, -1)),
        }
    )

    portfolio = build_equal_weight_portfolio(
        scores,
        8,
        max_sector_weight=0.50,
    )

    assert len(portfolio) == 8
    assert portfolio.groupby("sector")["target_weight"].sum().max() <= 0.50
