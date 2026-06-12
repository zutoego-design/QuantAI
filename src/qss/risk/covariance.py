from __future__ import annotations

import numpy as np
import pandas as pd

from qss.config.schema import CovarianceConfig


def estimate_covariance(returns: pd.DataFrame, config: CovarianceConfig) -> pd.DataFrame:
    window = returns.tail(config.lookback_days)
    min_periods = max(20, min(config.lookback_days // 2, len(window)))
    sample = window.cov(min_periods=min_periods)
    variances = window.var(ddof=1).fillna(0.0)
    for symbol in sample.index:
        if pd.isna(sample.loc[symbol, symbol]):
            sample.loc[symbol, symbol] = variances.get(symbol, 0.0)
    sample = sample.fillna(0.0)
    diagonal = pd.DataFrame(np.diag(np.diag(sample.values)), index=sample.index, columns=sample.columns)
    sigma = (1 - config.shrinkage_intensity) * sample + config.shrinkage_intensity * diagonal
    sigma = sigma + np.eye(len(sigma)) * 1e-6
    return sigma


def estimate_covariance_from_prices(
    prices: pd.DataFrame,
    symbols: list[str],
    as_of_date: pd.Timestamp,
    config: CovarianceConfig,
) -> pd.DataFrame:
    subset = prices.loc[(prices["date"] <= pd.Timestamp(as_of_date)) & (prices["symbol"].isin(symbols))]
    returns = subset.pivot(index="date", columns="symbol", values="return_1d").sort_index()
    covariance = estimate_covariance(returns, config)
    return covariance.reindex(index=symbols, columns=symbols).fillna(0.0)
