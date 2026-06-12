from __future__ import annotations

import numpy as np
import pandas as pd

from qss.config.schema import AppConfig


def winsorize_cross_section(
    df: pd.DataFrame,
    value_col: str,
    lower_q: float,
    upper_q: float,
) -> pd.DataFrame:
    out = df.copy()
    series = out[value_col]
    if series.dropna().empty:
        return out
    lower = series.quantile(lower_q)
    upper = series.quantile(upper_q)
    out[value_col] = series.clip(lower=lower, upper=upper)
    return out


def zscore_cross_section(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    out = df.copy()
    series = out[value_col]
    std = series.std(ddof=0)
    if pd.isna(std) or std == 0:
        out[value_col] = 0.0
        return out
    out[value_col] = (series - series.mean()) / std
    return out


def neutralize_factor(
    df: pd.DataFrame,
    factor_col: str,
    sector_col: str,
    market_cap_col: str,
    neutralize_sector: bool,
    neutralize_market_cap: bool,
) -> pd.DataFrame:
    out = df.copy()
    valid = out[factor_col].notna()
    if valid.sum() < 3:
        return out

    x_parts: list[pd.DataFrame] = []
    if neutralize_sector:
        x_parts.append(pd.get_dummies(out.loc[valid, sector_col].fillna("Unknown"), dtype=float))
    if neutralize_market_cap:
        x_parts.append(pd.DataFrame({"log_market_cap": np.log(out.loc[valid, market_cap_col].clip(lower=1.0))}))

    if not x_parts:
        return out
    x = pd.concat(x_parts, axis=1)
    x.insert(0, "intercept", 1.0)
    y = out.loc[valid, factor_col].astype(float).values
    x_values = x.astype(float).values
    try:
        beta, *_ = np.linalg.lstsq(x_values, y, rcond=None)
        fitted = x_values @ beta
        residual = y - fitted
        out.loc[valid, factor_col] = residual
    except np.linalg.LinAlgError:
        return out
    return out


def process_factor_values(factor_values: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for (date, factor_name), group in factor_values.groupby(["date", "factor_name"], sort=False):
        frame = group.copy()
        if config.factor_processing.winsorize.enabled:
            frame = winsorize_cross_section(
                frame,
                "raw_value",
                config.factor_processing.winsorize.lower_quantile,
                config.factor_processing.winsorize.upper_quantile,
            )
        frame = zscore_cross_section(frame, "raw_value")
        frame = neutralize_factor(
            frame,
            factor_col="raw_value",
            sector_col="sector",
            market_cap_col="market_cap",
            neutralize_sector=config.factor_processing.neutralization.sector,
            neutralize_market_cap=config.factor_processing.neutralization.market_cap,
        )
        frame["processed_value"] = frame["raw_value"] * frame["direction"]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else factor_values.copy()
