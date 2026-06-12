from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def estimate_transaction_cost(turnover: float, commission_bps: float, slippage_bps: float) -> float:
    return float(turnover * (commission_bps + slippage_bps) / 10_000)


@dataclass
class TransactionCostEstimate:
    commission: float
    slippage: float
    market_impact: float
    total: float
    adv_participation: float


def estimate_trade_cost(
    traded_notional: float,
    portfolio_value: float,
    adv_20d: float | None,
    annualized_volatility: float | None,
    commission_bps: float,
    slippage_bps: float,
    market_impact_coefficient: float,
) -> TransactionCostEstimate:
    if traded_notional <= 0 or portfolio_value <= 0:
        return TransactionCostEstimate(0.0, 0.0, 0.0, 0.0, 0.0)
    commission = traded_notional * commission_bps / 10_000
    slippage = traded_notional * slippage_bps / 10_000
    participation = traded_notional / adv_20d if adv_20d and adv_20d > 0 else np.inf
    volatility = max(float(annualized_volatility or 0.0), 0.0)
    if market_impact_coefficient <= 0:
        impact_rate = 0.0
    elif np.isfinite(participation):
        impact_rate = market_impact_coefficient * volatility * np.sqrt(
            max(participation, 0.0)
        )
    else:
        impact_rate = 1.0
    market_impact = traded_notional * min(float(impact_rate), 1.0)
    total = commission + slippage + market_impact
    return TransactionCostEstimate(
        commission=float(commission),
        slippage=float(slippage),
        market_impact=float(market_impact),
        total=float(total),
        adv_participation=float(participation),
    )
