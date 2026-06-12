from __future__ import annotations

import html

import pandas as pd
import plotly.express as px

from qss.backtest.metrics import compounded_monthly_returns


def render_backtest_report(
    daily_returns: pd.DataFrame,
    metrics: pd.DataFrame,
    rebalances: pd.DataFrame,
    drawdowns: pd.DataFrame | None = None,
    holdings: pd.DataFrame | None = None,
    factor_diagnostics: pd.DataFrame | None = None,
    data_diagnostics: pd.DataFrame | None = None,
    delisting_sensitivity: pd.DataFrame | None = None,
    manifest: dict | None = None,
) -> str:
    frame = daily_returns.copy()
    equity_fig = px.line(
        frame,
        x="date",
        y=[column for column in ["portfolio_value", "benchmark_value"] if column in frame],
        title="Equity Curve",
    )
    drawdown_fig = px.line(frame, x="date", y="drawdown", title="Underwater Curve")
    monthly = compounded_monthly_returns(frame)
    monthly_fig = px.bar(
        monthly,
        x="month",
        y=["portfolio_return", "benchmark_return"],
        barmode="group",
        title="Compounded Monthly Returns",
    )
    warning = (
        "Free-data research track: long history is approximate and must not be described "
        "as fully survivorship-bias free."
    )
    manifest_html = (
        f"<pre>{html.escape(str(manifest))}</pre>" if manifest is not None else ""
    )
    return f"""
    <html>
      <head>
        <title>QuantAI Research Report</title>
        <meta charset="utf-8"/>
        <style>
          body {{ font-family: Arial, sans-serif; margin: 32px; color: #172033; }}
          table {{ border-collapse: collapse; width: 100%; margin-bottom: 28px; }}
          th, td {{ border: 1px solid #dce3ec; padding: 7px; text-align: right; }}
          th:first-child, td:first-child {{ text-align: left; }}
          .warning {{ background: #fff4db; padding: 12px; border-left: 4px solid #9a650f; }}
        </style>
      </head>
      <body>
        <h1>QuantAI Backtest Research Report</h1>
        <p class="warning">{warning}</p>
        {manifest_html}
        {equity_fig.to_html(full_html=False, include_plotlyjs='cdn')}
        {drawdown_fig.to_html(full_html=False, include_plotlyjs=False)}
        {monthly_fig.to_html(full_html=False, include_plotlyjs=False)}
        <h2>Metrics</h2>
        {metrics.to_html(index=False)}
        <h2>Drawdown Episodes</h2>
        {(drawdowns if drawdowns is not None else pd.DataFrame()).to_html(index=False)}
        <h2>Rebalances And Trading Costs</h2>
        {rebalances.to_html(index=False)}
        <h2>Portfolio Concentration</h2>
        {(holdings if holdings is not None else pd.DataFrame()).to_html(index=False)}
        <h2>Factor Diagnostics</h2>
        {(factor_diagnostics if factor_diagnostics is not None else pd.DataFrame()).to_html(index=False)}
        <h2>Data Quality And Coverage</h2>
        {(data_diagnostics if data_diagnostics is not None else pd.DataFrame()).to_html(index=False)}
        <h2>Delisting Sensitivity</h2>
        {(delisting_sensitivity if delisting_sensitivity is not None else pd.DataFrame()).to_html(index=False)}
      </body>
    </html>
    """
