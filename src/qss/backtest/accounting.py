from __future__ import annotations

import pandas as pd


def compute_drawdown(returns: pd.Series) -> pd.Series:
    wealth = (1 + returns.fillna(0.0)).cumprod()
    return wealth / wealth.cummax() - 1


def compute_portfolio_value(returns: pd.Series, initial_capital: float) -> pd.Series:
    wealth = (1 + returns.fillna(0.0)).cumprod()
    return wealth * initial_capital
