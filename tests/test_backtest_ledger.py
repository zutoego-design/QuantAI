import numpy as np
import pandas as pd
import pytest

from qss.backtest.engine import BacktestRunSpec, _simulate_ledger
from qss.backtest.metrics import compounded_monthly_returns, drawdown_episodes
from qss.config.loader import get_config
from qss.data.calendar import next_trading_day


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
    _, _, holdings, trades = _simulate_ledger(
        spec, config, _prices([np.nan, 0.10, 0.0]), _targets(), "^IXIC"
    )
    day_two = holdings.loc[holdings["date"] == pd.Timestamp("2025-01-02")].set_index("symbol")
    assert day_two.loc["A", "weight"] > 0.5
    assert len(trades) == 2


def test_missing_intermediate_return_is_not_filled_with_zero():
    config = get_config(["configs/default.yaml"])
    config.runtime.research_mode = False
    prices = _prices([np.nan, np.nan, 0.01])
    spec = BacktestRunSpec(
        start_date="2025-01-01",
        end_date="2025-01-03",
        initial_capital=1000,
    )
    with pytest.raises(ValueError, match="Missing return"):
        _simulate_ledger(spec, config, prices, _targets(), "^IXIC")


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
