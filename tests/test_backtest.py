from qss.backtest.transaction_costs import estimate_transaction_cost


def test_transaction_cost_estimation():
    cost = estimate_transaction_cost(turnover=0.25, commission_bps=1.0, slippage_bps=5.0)
    assert abs(cost - 0.00015) < 1e-9
