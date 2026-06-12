from __future__ import annotations

import pandas as pd


def render_risk_report(
    as_of_date: pd.Timestamp,
    metrics: dict[str, float],
    portfolio: pd.DataFrame,
    sector_df: pd.DataFrame,
    alerts: pd.DataFrame,
    macro_summary: pd.DataFrame,
) -> str:
    metrics_df = pd.DataFrame([metrics])
    return f"""
    <html>
      <head><title>Risk Report {as_of_date:%Y-%m-%d}</title></head>
      <body>
        <h1>Risk Report {as_of_date:%Y-%m-%d}</h1>
        <h2>Current Risk Metrics</h2>
        {metrics_df.to_html(index=False)}
        <h2>Top Holdings</h2>
        {portfolio.sort_values('target_weight', ascending=False).head(10).to_html(index=False)}
        <h2>Sector Exposure</h2>
        {sector_df.to_html(index=False)}
        <h2>Alerts</h2>
        {alerts.to_html(index=False) if not alerts.empty else '<p>No alerts triggered.</p>'}
        <h2>Macro Summary</h2>
        {macro_summary.to_html(index=False)}
      </body>
    </html>
    """
