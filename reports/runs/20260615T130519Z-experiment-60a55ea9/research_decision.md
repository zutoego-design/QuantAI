# Research Evidence Decision

- Status: `rejected`
- Stage: `confirmatory`
- Primary metric: `sharpe_ratio`
- Preregistered threshold: `0.0`

## Blocking Issues
- Preregistered factor evidence did not survive direction and FDR checks: ['accruals', 'beta_to_spy', 'book_to_market', 'earnings_yield', 'free_cash_flow_yield', 'max_drawdown_252d', 'momentum_12_1', 'momentum_3m', 'momentum_6m', 'realized_vol_252d', 'realized_vol_60d', 'roe', 'sales_yield'].
- Required robustness tests are invalid: [{'test': 'top_n_sensitivity', 'setting': '100'}].

## Reasons
- The Deflated Sharpe probability is below the required threshold.
- Methodology blockers remain unresolved.
