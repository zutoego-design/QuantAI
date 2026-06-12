from __future__ import annotations

import pandas as pd


def render_rebalance_report(
    as_of_date: pd.Timestamp,
    portfolio: pd.DataFrame,
    optimizer_status: str,
    warning: str | None,
    universe_size: int,
) -> str:
    sector_exposure = portfolio.groupby("sector", as_index=False)["target_weight"].sum()
    turnover = portfolio["trade_weight"].abs().sum()
    top_alpha = portfolio.sort_values("alpha_score", ascending=False).head(10)
    return f"""
    <html>
      <head><title>Rebalance Report {as_of_date:%Y-%m-%d}</title></head>
      <body>
        <h1>Rebalance Report {as_of_date:%Y-%m-%d}</h1>
        <p>Universe size: {universe_size}</p>
        <p>Optimizer status: {optimizer_status}</p>
        <p>Warning: {warning or 'None'}</p>
        <p>Turnover: {turnover:.4f}</p>
        <h2>Target Weights</h2>
        {portfolio.to_html(index=False)}
        <h2>Sector Exposure</h2>
        {sector_exposure.to_html(index=False)}
        <h2>Top Alpha Contributors</h2>
        {top_alpha.to_html(index=False)}
      </body>
    </html>
    """
