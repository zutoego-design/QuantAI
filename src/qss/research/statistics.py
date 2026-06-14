from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from qss.backtest.accounting import compute_drawdown
from qss.utils import annualize_return


@dataclass(frozen=True)
class MeanTest:
    mean: float
    standard_error: float
    t_stat: float
    p_value: float


def newey_west_mean_test(
    values: pd.Series,
    max_lags: int | None = None,
) -> MeanTest:
    series = pd.to_numeric(values, errors="coerce").dropna()
    if len(series) < 3:
        return MeanTest(
            mean=float(series.mean()) if not series.empty else np.nan,
            standard_error=np.nan,
            t_stat=np.nan,
            p_value=np.nan,
        )
    lags = (
        max_lags
        if max_lags is not None
        else max(1, int(np.floor(4 * (len(series) / 100) ** (2 / 9))))
    )
    result = sm.OLS(series.to_numpy(dtype=float), np.ones((len(series), 1))).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": lags},
    )
    return MeanTest(
        mean=float(result.params[0]),
        standard_error=float(result.bse[0]),
        t_stat=float(result.tvalues[0]),
        p_value=float(result.pvalues[0]),
    )


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(p_values, errors="coerce")
    valid = numeric.dropna().clip(0.0, 1.0)
    adjusted = pd.Series(np.nan, index=numeric.index, dtype=float)
    if valid.empty:
        return adjusted
    ordered = valid.sort_values()
    count = len(ordered)
    raw = ordered.to_numpy(dtype=float) * count / np.arange(1, count + 1)
    monotone = np.minimum.accumulate(raw[::-1])[::-1].clip(0.0, 1.0)
    adjusted.loc[ordered.index] = monotone
    return adjusted


def _metric_value(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    metric: str,
) -> float:
    portfolio = pd.to_numeric(portfolio_returns, errors="coerce").fillna(0.0)
    benchmark = pd.to_numeric(benchmark_returns, errors="coerce").fillna(0.0)
    if metric == "total_return":
        return float((1.0 + portfolio).prod() - 1.0)
    if metric == "cagr":
        return float(annualize_return(portfolio))
    if metric == "sharpe_ratio":
        volatility = float(portfolio.std(ddof=0))
        return (
            float(portfolio.mean() / volatility * np.sqrt(252))
            if volatility > 0
            else 0.0
        )
    if metric == "alpha_annualized":
        variance = float(benchmark.var(ddof=0))
        beta = (
            float(np.cov(portfolio, benchmark, ddof=0)[0, 1] / variance)
            if variance > 0
            else 0.0
        )
        return float((portfolio.mean() - beta * benchmark.mean()) * 252)
    if metric == "max_drawdown":
        drawdown = compute_drawdown(portfolio)
        return float(drawdown.min()) if not drawdown.empty else 0.0
    raise ValueError(f"Unsupported bootstrap metric: {metric}")


def _circular_block_indices(
    size: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    blocks = int(np.ceil(size / block_size))
    starts = rng.integers(0, size, size=blocks)
    indices = np.concatenate(
        [(start + np.arange(block_size)) % size for start in starts]
    )
    return indices[:size]


def block_bootstrap_summary(
    daily_returns: pd.DataFrame,
    *,
    primary_metric: str,
    block_size: int = 21,
    samples: int = 2000,
    seed: int = 42,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    required = {"portfolio_return", "benchmark_return"}
    if daily_returns.empty or not required.issubset(daily_returns.columns):
        return pd.DataFrame()
    frame = daily_returns.dropna(subset=list(required)).reset_index(drop=True)
    if frame.empty:
        return pd.DataFrame()
    metrics = list(
        dict.fromkeys(
            [
                primary_metric,
                "sharpe_ratio",
                "alpha_annualized",
                "max_drawdown",
            ]
        )
    )
    rng = np.random.default_rng(seed)
    distributions = {metric: [] for metric in metrics}
    portfolio = frame["portfolio_return"]
    benchmark = frame["benchmark_return"]
    for _ in range(samples):
        indices = _circular_block_indices(len(frame), block_size, rng)
        sampled_portfolio = portfolio.iloc[indices].reset_index(drop=True)
        sampled_benchmark = benchmark.iloc[indices].reset_index(drop=True)
        for metric in metrics:
            distributions[metric].append(
                _metric_value(sampled_portfolio, sampled_benchmark, metric)
            )
    alpha = 1.0 - confidence_level
    rows = []
    for metric in metrics:
        values = np.asarray(distributions[metric], dtype=float)
        rows.append(
            {
                "metric": metric,
                "estimate": _metric_value(portfolio, benchmark, metric),
                "lower_95": float(np.quantile(values, alpha / 2)),
                "upper_95": float(np.quantile(values, 1 - alpha / 2)),
                "one_sided_lower_95": float(np.quantile(values, alpha)),
                "bootstrap_samples": samples,
                "block_size": block_size,
                "seed": seed,
            }
        )
    return pd.DataFrame(rows)


def deflated_sharpe_probability(
    returns: pd.Series,
    trial_count: int,
) -> dict[str, float]:
    series = pd.to_numeric(returns, errors="coerce").dropna()
    if len(series) < 3:
        return {
            "observed_sharpe": np.nan,
            "expected_max_sharpe": np.nan,
            "probability": np.nan,
            "trial_count": float(max(trial_count, 1)),
        }
    standard_deviation = float(series.std(ddof=1))
    daily_sharpe = float(series.mean() / standard_deviation) if standard_deviation else 0.0
    annualized_sharpe = daily_sharpe * np.sqrt(252)
    trials = max(int(trial_count), 1)
    if trials == 1:
        expected_max_daily = 0.0
    else:
        euler_gamma = 0.5772156649015329
        first = stats.norm.ppf(1.0 - 1.0 / trials)
        second = stats.norm.ppf(1.0 - 1.0 / (trials * np.e))
        expected_max_daily = (
            (1.0 - euler_gamma) * first + euler_gamma * second
        ) / np.sqrt(max(len(series) - 1, 1))
    skewness = float(stats.skew(series, bias=False))
    kurtosis = float(stats.kurtosis(series, fisher=False, bias=False))
    denominator = np.sqrt(
        max(
            1.0
            - skewness * daily_sharpe
            + ((kurtosis - 1.0) / 4.0) * daily_sharpe**2,
            1e-12,
        )
    )
    statistic = (
        (daily_sharpe - expected_max_daily)
        * np.sqrt(max(len(series) - 1, 1))
        / denominator
    )
    return {
        "observed_sharpe": annualized_sharpe,
        "expected_max_sharpe": float(expected_max_daily * np.sqrt(252)),
        "probability": float(stats.norm.cdf(statistic)),
        "trial_count": float(trials),
    }


def fama_french_style_regression(
    daily_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    max_lags: int = 5,
) -> tuple[pd.DataFrame, dict[str, float]]:
    required_returns = {"date", "portfolio_return"}
    required_factors = {"date", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "RF"}
    if (
        daily_returns.empty
        or factor_returns.empty
        or not required_returns.issubset(daily_returns.columns)
        or not required_factors.issubset(factor_returns.columns)
    ):
        return pd.DataFrame(), {}
    returns = daily_returns.copy()
    factors = factor_returns.copy()
    returns["date"] = pd.to_datetime(returns["date"]).dt.normalize()
    factors["date"] = pd.to_datetime(factors["date"]).dt.normalize()
    merged = returns.merge(factors, on="date", how="inner")
    if len(merged) < 30:
        return pd.DataFrame(), {"observations": float(len(merged))}
    factor_columns = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
    dependent = merged["portfolio_return"] - merged["RF"]
    design = sm.add_constant(merged[factor_columns], has_constant="add")
    result = sm.OLS(dependent, design).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": max_lags},
    )
    rows = []
    for name in ["const", *factor_columns]:
        rows.append(
            {
                "factor": "alpha" if name == "const" else name,
                "coefficient": float(result.params[name]),
                "annualized_coefficient": (
                    float(result.params[name] * 252)
                    if name == "const"
                    else float(result.params[name])
                ),
                "t_stat": float(result.tvalues[name]),
                "p_value": float(result.pvalues[name]),
            }
        )
    summary = {
        "observations": float(result.nobs),
        "r_squared": float(result.rsquared),
        "alpha_annualized": float(result.params["const"] * 252),
        "alpha_t_stat": float(result.tvalues["const"]),
    }
    return pd.DataFrame(rows), summary
