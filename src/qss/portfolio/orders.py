from __future__ import annotations

import pandas as pd


def build_orders(portfolio_weights: pd.DataFrame) -> pd.DataFrame:
    orders = portfolio_weights.copy()
    orders["action"] = orders["trade_weight"].apply(lambda x: "BUY" if x > 0 else "SELL" if x < 0 else "HOLD")
    return orders[["date", "strategy_name", "symbol", "previous_weight", "target_weight", "trade_weight", "action"]]
