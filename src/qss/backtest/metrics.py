from __future__ import annotations

import numpy as np
import pandas as pd

from qss.backtest.accounting import compute_drawdown
from qss.research.statistics import benjamini_hochberg, newey_west_mean_test
from qss.utils import annualize_return, annualize_volatility


def compounded_monthly_returns(daily_returns: pd.DataFrame) -> pd.DataFrame:
    if daily_returns.empty:
        return pd.DataFrame(columns=["month", "portfolio_return", "benchmark_return"])
    frame = daily_returns.copy()
    frame["month"] = pd.to_datetime(frame["date"]).dt.to_period("M").astype(str)
    return (
        frame.groupby("month")[["portfolio_return", "benchmark_return"]]
        .agg(lambda values: (1.0 + values).prod() - 1.0)
        .reset_index()
    )


def drawdown_episodes(returns: pd.Series, dates: pd.Series | None = None) -> pd.DataFrame:
    if returns.empty:
        return pd.DataFrame(
            columns=["start_date", "trough_date", "recovery_date", "max_drawdown", "duration_days"]
        )
    dates = pd.to_datetime(dates if dates is not None else returns.index)
    drawdown = compute_drawdown(returns).reset_index(drop=True)
    rows: list[dict] = []
    start: int | None = None
    for idx, value in enumerate(drawdown):
        if value < 0 and start is None:
            start = max(idx - 1, 0)
        recovered = value >= -1e-12 and start is not None
        last = idx == len(drawdown) - 1 and start is not None
        if recovered or last:
            end = idx
            section = drawdown.iloc[start : end + 1]
            trough = int(section.idxmin())
            recovery = dates[end] if recovered else pd.NaT
            rows.append(
                {
                    "start_date": dates[start],
                    "trough_date": dates[trough],
                    "recovery_date": recovery,
                    "max_drawdown": float(section.min()),
                    "duration_days": int((dates[end] - dates[start]).days),
                }
            )
            start = None
    return pd.DataFrame(rows)


def compute_backtest_metrics(
    daily_returns: pd.DataFrame,
    rebalance_history: pd.DataFrame,
) -> pd.DataFrame:
    if daily_returns.empty:
        return pd.DataFrame(columns=["metric", "value", "category"])
    p = daily_returns["portfolio_return"].astype(float)
    b = daily_returns["benchmark_return"].astype(float)
    active = p - b
    downside = np.sqrt(np.mean(np.square(np.minimum(p, 0)))) * np.sqrt(252)
    ann_return = annualize_return(p)
    ann_vol = annualize_volatility(p)
    total_return = float((1 + p).prod() - 1)
    gross_total_return = (
        float((1 + daily_returns["gross_return"].astype(float)).prod() - 1)
        if "gross_return" in daily_returns
        else total_return
    )
    benchmark_total_return = float((1 + b).prod() - 1)
    sharpe = float(p.mean() / p.std(ddof=0) * np.sqrt(252)) if p.std(ddof=0) else 0.0
    sortino = float(p.mean() * 252 / downside) if downside else 0.0
    drawdown = compute_drawdown(p)
    max_dd = float(drawdown.min()) if not drawdown.empty else 0.0
    calmar = float(ann_return / abs(max_dd)) if max_dd < 0 else 0.0
    beta = float(np.cov(p, b, ddof=0)[0, 1] / np.var(b, ddof=0)) if np.var(b) else 0.0
    alpha = float((p.mean() - beta * b.mean()) * 252)
    correlation = float(p.corr(b)) if p.std(ddof=0) and b.std(ddof=0) else 0.0
    r_squared = correlation**2
    tracking_error = float(active.std(ddof=0) * np.sqrt(252))
    information_ratio = (
        float(active.mean() / active.std(ddof=0) * np.sqrt(252))
        if active.std(ddof=0)
        else 0.0
    )
    threshold = 0.0
    gains = float((p[p > threshold] - threshold).sum())
    losses = float((threshold - p[p < threshold]).sum())
    omega = gains / losses if losses else np.inf
    var_95 = float(p.quantile(0.05))
    cvar_95 = float(p.loc[p <= var_95].mean()) if (p <= var_95).any() else var_95
    up = b > 0
    down = b < 0
    up_capture = float(p[up].mean() / b[up].mean()) if up.any() and b[up].mean() else 0.0
    down_capture = float(p[down].mean() / b[down].mean()) if down.any() and b[down].mean() else 0.0

    avg_turnover = (
        float(rebalance_history["turnover"].mean())
        if "turnover" in rebalance_history and not rebalance_history.empty
        else 0.0
    )
    avg_holdings = (
        float(rebalance_history["holding_count"].mean())
        if "holding_count" in rebalance_history and not rebalance_history.empty
        else 0.0
    )
    cost_drag = gross_total_return - total_return
    max_participation = (
        float(rebalance_history["max_adv_participation"].max())
        if "max_adv_participation" in rebalance_history and not rebalance_history.empty
        else 0.0
    )
    metrics = {
        "performance": {
            "total_return": total_return,
            "gross_total_return": gross_total_return,
            "net_total_return": total_return,
            "benchmark_total_return": benchmark_total_return,
            "cagr": ann_return,
            "annualized_volatility": ann_vol,
            "downside_volatility": float(downside),
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "omega_ratio": float(omega),
            "max_drawdown": max_dd,
            "var_95_daily": var_95,
            "cvar_95_daily": cvar_95,
        },
        "benchmark": {
            "alpha_annualized": alpha,
            "beta": beta,
            "correlation": correlation,
            "r_squared": r_squared,
            "tracking_error": tracking_error,
            "information_ratio": information_ratio,
            "up_capture": up_capture,
            "down_capture": down_capture,
        },
        "portfolio": {
            "average_turnover": avg_turnover,
            "average_number_of_holdings": avg_holdings,
            "cost_drag": cost_drag,
            "max_adv_participation": max_participation,
        },
    }
    rows = [
        {"category": category, "metric": metric, "value": value}
        for category, values in metrics.items()
        for metric, value in values.items()
    ]
    return pd.DataFrame(rows)


def factor_diagnostics(
    factor_values: pd.DataFrame,
    forward_returns: pd.DataFrame,
    quantiles: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"date", "symbol", "factor_name", "processed_value"}
    if factor_values.empty or not required.issubset(factor_values.columns):
        return pd.DataFrame(), pd.DataFrame()
    merged = factor_values.merge(forward_returns, on=["date", "symbol"], how="left")
    rows: list[dict] = []
    quantile_rows: list[dict] = []
    for factor_name, factor in merged.groupby("factor_name"):
        daily_ic: list[float] = []
        daily_pearson_ic: list[float] = []
        coverage: list[float] = []
        for date, cross in factor.groupby("date"):
            valid = cross.dropna(subset=["processed_value", "forward_return"])
            coverage.append(len(valid) / len(cross) if len(cross) else 0.0)
            if (
                len(valid) >= 3
                and valid["processed_value"].std(ddof=0) > 0
                and valid["forward_return"].std(ddof=0) > 0
            ):
                daily_ic.append(valid["processed_value"].corr(valid["forward_return"], method="spearman"))
                daily_pearson_ic.append(
                    valid["processed_value"].corr(valid["forward_return"], method="pearson")
                )
                ranks = pd.qcut(
                    valid["processed_value"].rank(method="first"),
                    min(quantiles, len(valid)),
                    labels=False,
                )
                for bucket, values in valid.assign(quantile=ranks).groupby("quantile"):
                    quantile_rows.append(
                        {
                            "date": date,
                            "factor_name": factor_name,
                            "quantile": int(bucket) + 1,
                            "forward_return": float(values["forward_return"].mean()),
                        }
                    )
        series = pd.Series(daily_ic, dtype=float).dropna()
        mean_ic = float(series.mean()) if not series.empty else np.nan
        std_ic = float(series.std(ddof=1)) if len(series) > 1 else np.nan
        mean_test = newey_west_mean_test(series)
        rows.append(
            {
                "factor_name": factor_name,
                "ic": float(pd.Series(daily_pearson_ic).mean()) if daily_pearson_ic else np.nan,
                "rank_ic": mean_ic,
                "ic_ir": mean_ic / std_ic if std_ic and not np.isnan(std_ic) else np.nan,
                "t_stat": mean_test.t_stat,
                "p_value": mean_test.p_value,
                "coverage": float(np.mean(coverage)) if coverage else 0.0,
                "missing_rate": float(merged.loc[merged["factor_name"] == factor_name, "processed_value"].isna().mean()),
            }
        )
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["fdr_q_value"] = benjamini_hochberg(summary["p_value"])
        summary["fdr_significant"] = summary["fdr_q_value"] <= 0.05
    return summary, pd.DataFrame(quantile_rows)


def forward_returns_for_factors(
    factor_values: pd.DataFrame,
    prices: pd.DataFrame,
    horizon_days: int,
) -> pd.DataFrame:
    if factor_values.empty or prices.empty:
        return pd.DataFrame(columns=["date", "symbol", "forward_return"])
    price_panel = prices.pivot_table(
        index="date", columns="symbol", values="adj_close", aggfunc="last"
    ).sort_index()
    rows: list[dict] = []
    for date, cross in factor_values[["date", "symbol"]].drop_duplicates().groupby("date"):
        available = price_panel.index[price_panel.index <= pd.Timestamp(date)]
        if len(available) == 0:
            continue
        start_position = price_panel.index.get_loc(available[-1])
        end_position = start_position + horizon_days
        if end_position >= len(price_panel.index):
            continue
        start_prices = price_panel.iloc[start_position]
        end_prices = price_panel.iloc[end_position]
        returns = end_prices / start_prices - 1.0
        for symbol in cross["symbol"]:
            value = returns.get(symbol, np.nan)
            rows.append(
                {"date": pd.Timestamp(date), "symbol": symbol, "forward_return": value}
            )
    return pd.DataFrame(rows, columns=["date", "symbol", "forward_return"])


def comprehensive_factor_diagnostics(
    factor_values: pd.DataFrame,
    prices: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    if factor_values.empty:
        return {
            "summary": pd.DataFrame(),
            "quantiles": pd.DataFrame(),
            "decay": pd.DataFrame(),
            "correlation": pd.DataFrame(),
        }
    forward_21 = forward_returns_for_factors(factor_values, prices, 21)
    summary, quantiles = factor_diagnostics(factor_values, forward_21)
    quantile_summary = (
        quantiles.groupby(["factor_name", "quantile"], as_index=False)["forward_return"].mean()
        if not quantiles.empty
        else quantiles
    )
    if not summary.empty and not quantile_summary.empty:
        monotonicity = (
            quantile_summary.groupby("factor_name")
            .apply(
                lambda group: group["quantile"].corr(
                    group["forward_return"], method="spearman"
                ),
                include_groups=False,
            )
            .rename("quantile_monotonicity")
            .reset_index()
        )
        summary = summary.merge(monotonicity, on="factor_name", how="left")

    turnover_rows: list[dict] = []
    for factor_name, factor in factor_values.groupby("factor_name"):
        prior: set[str] | None = None
        values: list[float] = []
        for _, cross in factor.sort_values("date").groupby("date"):
            valid = cross.dropna(subset=["processed_value"])
            if valid.empty:
                continue
            cutoff = valid["processed_value"].quantile(0.8)
            current = set(valid.loc[valid["processed_value"] >= cutoff, "symbol"])
            if prior is not None and prior:
                values.append(1 - len(current & prior) / len(prior))
            prior = current
        turnover_rows.append(
            {
                "factor_name": factor_name,
                "top_quantile_turnover": float(np.mean(values)) if values else np.nan,
            }
        )
    if not summary.empty:
        summary = summary.merge(pd.DataFrame(turnover_rows), on="factor_name", how="left")

    decay_rows: list[dict] = []
    for horizon in [1, 5, 21, 63]:
        forward = forward_returns_for_factors(factor_values, prices, horizon)
        diag, _ = factor_diagnostics(factor_values, forward)
        if not diag.empty:
            decay_rows.extend(
                {
                    "factor_name": row.factor_name,
                    "horizon_days": horizon,
                    "rank_ic": row.rank_ic,
                }
                for row in diag.itertuples(index=False)
            )
    pivot = factor_values.pivot_table(
        index=["date", "symbol"],
        columns="factor_name",
        values="processed_value",
        aggfunc="last",
    )
    correlation = pivot.corr(method="spearman").reset_index().rename(
        columns={"factor_name": "factor"}
    )
    return {
        "summary": summary,
        "quantiles": quantile_summary,
        "decay": pd.DataFrame(decay_rows),
        "correlation": correlation,
    }
