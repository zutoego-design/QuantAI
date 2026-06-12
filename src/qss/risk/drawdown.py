from __future__ import annotations

import pandas as pd


def current_drawdown(returns: pd.Series) -> float:
    wealth = (1 + returns.fillna(0.0)).cumprod()
    if wealth.empty:
        return 0.0
    drawdown = wealth / wealth.cummax() - 1
    return float(drawdown.iloc[-1])
