from __future__ import annotations

from pathlib import Path

import pandas as pd

from qss.config.schema import AppConfig
from qss.data.storage import append_or_replace_parquet
from qss.macro.etf_proxy import compute_etf_proxy_performance
from qss.macro.indicators import latest_series_value, year_over_year_change, zscore_recent


def compute_macro_regime(
    as_of_date: pd.Timestamp,
    macro_observations: pd.DataFrame,
    prices: pd.DataFrame,
    config: AppConfig,
) -> pd.DataFrame:
    inflation_yoy = year_over_year_change(macro_observations, config.macro.fred_series["cpi"], as_of_date)
    fed_funds = latest_series_value(macro_observations, config.macro.fred_series["fed_funds"], as_of_date)
    two_year = latest_series_value(macro_observations, config.macro.fred_series["two_year_treasury"], as_of_date)
    ten_year = latest_series_value(macro_observations, config.macro.fred_series["ten_year_treasury"], as_of_date)
    unemployment = latest_series_value(macro_observations, config.macro.fred_series["unemployment"], as_of_date)
    baa_zscore = zscore_recent(macro_observations, config.macro.fred_series["baa_spread"], as_of_date)
    curve = ten_year - two_year

    inflation_regime = "high" if inflation_yoy >= config.macro.regime_rules["inflation"]["high_threshold_yoy"] else "normal"
    if pd.isna(inflation_yoy):
        inflation_regime = "low"
    rates_regime = "rising" if fed_funds > 3 else "falling" if fed_funds < 1 else "stable"
    curve_regime = "inverted" if curve <= config.macro.regime_rules["rates"]["curve_inversion_threshold"] else "normal"
    credit_regime = "stressed" if baa_zscore > 2 else "widening" if baa_zscore > config.macro.regime_rules["credit"]["spread_widening_zscore"] else "calm"

    summary = f"Inflation={inflation_regime}; Rates={rates_regime}; Curve={curve_regime}; Credit={credit_regime}; Unemployment={unemployment:.2f}"
    df = pd.DataFrame(
        [
            {
                "date": pd.Timestamp(as_of_date).normalize(),
                "inflation_regime": inflation_regime,
                "rates_regime": rates_regime,
                "curve_regime": curve_regime,
                "credit_regime": credit_regime,
                "risk_summary": summary,
            }
        ]
    )
    etf_perf = compute_etf_proxy_performance(prices, config.macro.etf_proxies, as_of_date)
    if not etf_perf.empty:
        merged_summary = "; ".join(f"{row.symbol} 3m={row.return_3m:.2%}" for row in etf_perf.itertuples(index=False))
        df["risk_summary"] = df["risk_summary"] + "; ETF=" + merged_summary
    append_or_replace_parquet(df, Path(config.paths.gold_data) / "macro" / "macro_regime.parquet", ["date"])
    return df
