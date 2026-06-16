from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qss.config.schema import AppConfig
from qss.data.storage import resolve_path
from qss.ingestion.fama_french import load_fama_french_daily
from qss.research.statistics import newey_west_mean_test

FACTOR_COLUMNS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
VALUE_FACTORS = [
    "book_to_market",
    "earnings_yield",
    "free_cash_flow_yield",
    "sales_yield",
]
VOLATILITY_FACTORS = ["realized_vol_60d", "realized_vol_252d"]


@dataclass(frozen=True)
class AttributionBundle:
    root: Path
    markdown_report: Path
    manifest: Path


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


def _normalize_dates(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "date" in result:
        result["date"] = (
            pd.to_datetime(result["date"]).dt.normalize().astype("datetime64[ns]")
        )
    return result


def _price_returns(prices: pd.DataFrame) -> pd.DataFrame:
    frame = _normalize_dates(prices)
    required = {"date", "symbol", "return_1d"}
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame(columns=["date", "symbol", "return_1d"])
    return frame[["date", "symbol", "return_1d"]].dropna()


def holding_return_contributions(
    holdings: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    holdings_frame = _normalize_dates(holdings)
    if holdings_frame.empty:
        return pd.DataFrame()
    returns = _price_returns(prices)
    merged = holdings_frame.merge(returns, on=["date", "symbol"], how="left")
    merged["return_1d"] = pd.to_numeric(merged["return_1d"], errors="coerce")
    merged["weight"] = pd.to_numeric(merged["weight"], errors="coerce").fillna(0.0)
    merged["return_contribution"] = merged["weight"] * merged["return_1d"].fillna(0.0)
    merged["return_available"] = merged["return_1d"].notna()
    return merged


def sector_attribution(
    holdings: pd.DataFrame,
    prices: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    contributions = holding_return_contributions(holdings, prices)
    if contributions.empty:
        return pd.DataFrame(), pd.DataFrame()
    contributions["sector"] = contributions.get("sector", "Unknown")
    daily = (
        contributions.groupby(["date", "sector"], as_index=False)
        .agg(
            contribution=("return_contribution", "sum"),
            average_weight=("weight", "sum"),
            names=("symbol", "nunique"),
            return_coverage=("return_available", "mean"),
        )
        .sort_values(["date", "sector"])
    )
    summary = (
        daily.groupby("sector", as_index=False)
        .agg(
            total_contribution=("contribution", "sum"),
            average_weight=("average_weight", "mean"),
            active_days=("date", "nunique"),
            average_return_coverage=("return_coverage", "mean"),
        )
        .sort_values("total_contribution", ascending=False)
        .reset_index(drop=True)
    )
    return daily, summary


def _wide_features(features: pd.DataFrame, factor_names: list[str]) -> pd.DataFrame:
    frame = _normalize_dates(features)
    required = {"date", "symbol", "factor_name", "processed_value"}
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame()
    frame = frame.loc[frame["factor_name"].isin(factor_names)].copy()
    if frame.empty:
        return pd.DataFrame()
    wide = (
        frame.pivot_table(
            index=["date", "symbol"],
            columns="factor_name",
            values="processed_value",
            aggfunc="mean",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    return wide.sort_values(["date", "symbol"])


def _align_features_to_holdings(
    holdings: pd.DataFrame,
    features: pd.DataFrame,
    factor_names: list[str],
) -> pd.DataFrame:
    left = _normalize_dates(holdings)[["date", "symbol"]].drop_duplicates()
    wide = _wide_features(features, factor_names)
    if left.empty or wide.empty:
        return pd.DataFrame()
    left = left.sort_values(["date", "symbol"])
    return pd.merge_asof(
        left,
        wide,
        on="date",
        by="symbol",
        direction="backward",
    )


def _tercile_bucket(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    labels = pd.Series("missing", index=values.index, dtype=object)
    valid = numeric.dropna()
    if valid.empty:
        return labels
    ranked = valid.rank(method="first")
    try:
        labels.loc[valid.index] = pd.qcut(
            ranked,
            3,
            labels=["low", "middle", "high"],
        ).astype(str)
    except ValueError:
        labels.loc[valid.index] = "middle"
    return labels


def bucket_attribution(
    holdings: pd.DataFrame,
    prices: pd.DataFrame,
    features: pd.DataFrame,
    *,
    factor_names: list[str],
    exposure_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    contributions = holding_return_contributions(holdings, prices)
    aligned = _align_features_to_holdings(holdings, features, factor_names)
    if contributions.empty or aligned.empty:
        return pd.DataFrame(), pd.DataFrame()
    aligned[exposure_name] = aligned[factor_names].mean(axis=1, skipna=True)
    merged = contributions.merge(
        aligned[["date", "symbol", exposure_name]],
        on=["date", "symbol"],
        how="left",
    )
    merged["bucket"] = (
        merged.groupby("date", group_keys=False)[exposure_name]
        .apply(_tercile_bucket)
        .astype(str)
    )
    daily = (
        merged.groupby(["date", "bucket"], as_index=False)
        .agg(
            contribution=("return_contribution", "sum"),
            average_weight=("weight", "sum"),
            average_exposure=(exposure_name, "mean"),
            names=("symbol", "nunique"),
            return_coverage=("return_available", "mean"),
        )
        .sort_values(["date", "bucket"])
    )
    summary = (
        daily.groupby("bucket", as_index=False)
        .agg(
            total_contribution=("contribution", "sum"),
            average_weight=("average_weight", "mean"),
            average_exposure=("average_exposure", "mean"),
            active_days=("date", "nunique"),
            average_return_coverage=("return_coverage", "mean"),
        )
        .sort_values("total_contribution", ascending=False)
        .reset_index(drop=True)
    )
    return daily, summary


def fama_french_contribution_attribution(
    daily_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    exposures: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    daily = _normalize_dates(daily_returns)
    factors = _normalize_dates(factor_returns)
    if daily.empty or factors.empty or exposures.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float)
    coefficients = exposures.set_index("factor")["coefficient"].to_dict()
    merged = daily.merge(factors, on="date", how="inner")
    if merged.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float)
    rows = []
    predicted = pd.Series(0.0, index=merged.index, dtype=float)
    alpha = float(coefficients.get("alpha", 0.0))
    rows.extend(
        {
            "date": row.date,
            "component": "alpha",
            "contribution": alpha,
        }
        for row in merged.itertuples(index=False)
    )
    predicted = predicted + alpha
    for factor in FACTOR_COLUMNS:
        if factor not in merged:
            continue
        coefficient = float(coefficients.get(factor, 0.0))
        contribution = pd.to_numeric(merged[factor], errors="coerce").fillna(0.0) * coefficient
        predicted = predicted + contribution
        rows.extend(
            {
                "date": date,
                "component": factor,
                "contribution": float(value),
            }
            for date, value in zip(merged["date"], contribution, strict=False)
        )
    dependent = (
        pd.to_numeric(merged["portfolio_return"], errors="coerce").fillna(0.0)
        - pd.to_numeric(merged.get("RF", 0.0), errors="coerce").fillna(0.0)
    )
    residual = dependent - predicted
    rows.extend(
        {
            "date": date,
            "component": "residual",
            "contribution": float(value),
        }
        for date, value in zip(merged["date"], residual, strict=False)
    )
    attribution = pd.DataFrame(rows)
    summary = (
        attribution.groupby("component", as_index=False)
        .agg(
            total_contribution=("contribution", "sum"),
            average_daily_contribution=("contribution", "mean"),
            daily_contribution_volatility=("contribution", "std"),
            observations=("contribution", "count"),
        )
    )
    summary["annualized_average_contribution"] = (
        summary["average_daily_contribution"] * 252
    )
    denominator = float(dependent.sum())
    summary["share_of_excess_return"] = (
        summary["total_contribution"] / denominator
        if abs(denominator) > 1e-12
        else np.nan
    )
    summary = summary.sort_values(
        "total_contribution",
        ascending=False,
    ).reset_index(drop=True)
    return attribution, summary, pd.Series(residual.to_numpy(), index=merged["date"])


def residual_alpha_stability(residual: pd.Series) -> tuple[pd.DataFrame, dict[str, Any]]:
    series = pd.Series(residual).dropna()
    if series.empty:
        return pd.DataFrame(), {}
    test = newey_west_mean_test(series.reset_index(drop=True))
    rows: list[dict[str, Any]] = [
        {
            "window": "full",
            "observations": int(len(series)),
            "annualized_mean": float(series.mean() * 252),
            "daily_volatility": float(series.std(ddof=0)),
            "t_stat": test.t_stat,
            "p_value": test.p_value,
            "positive_window_share": float((series > 0).mean()),
        }
    ]
    for window in [63, 126]:
        rolling = series.rolling(window).mean().dropna() * 252
        rows.append(
            {
                "window": f"rolling_{window}",
                "observations": int(len(rolling)),
                "annualized_mean": float(rolling.mean()) if not rolling.empty else np.nan,
                "min_annualized_mean": float(rolling.min()) if not rolling.empty else np.nan,
                "max_annualized_mean": float(rolling.max()) if not rolling.empty else np.nan,
                "latest_annualized_mean": (
                    float(rolling.iloc[-1]) if not rolling.empty else np.nan
                ),
                "positive_window_share": (
                    float((rolling > 0).mean()) if not rolling.empty else np.nan
                ),
            }
        )
    frame = pd.DataFrame(rows)
    summary = {
        "full_sample_annualized_residual_alpha": rows[0]["annualized_mean"],
        "full_sample_t_stat": rows[0]["t_stat"],
        "full_sample_p_value": rows[0]["p_value"],
        "rolling_windows": [63, 126],
    }
    return frame, summary


def _top_summary_rows(frame: pd.DataFrame, label: str) -> list[str]:
    if frame.empty or "total_contribution" not in frame:
        return [f"- {label}: unavailable"]
    rows = []
    for item in frame.head(3).itertuples(index=False):
        name = getattr(item, frame.columns[0])
        rows.append(
            f"- {label} {name}: total contribution "
            f"`{getattr(item, 'total_contribution'):.4f}`"
        )
    return rows


def _write_markdown(
    root: Path,
    source_run: str,
    tables: dict[str, pd.DataFrame],
    residual_summary: dict[str, Any],
) -> Path:
    lines = [
        "# Attribution Analysis",
        "",
        f"- Source run: `{source_run}`",
        f"- Generated at: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Fama-French Contribution",
        *_top_summary_rows(tables.get("fama_french_summary", pd.DataFrame()), "Component"),
        "",
        "## Sector Attribution",
        *_top_summary_rows(tables.get("sector_summary", pd.DataFrame()), "Sector"),
        "",
        "## Beta / Volatility Bucket Attribution",
        *_top_summary_rows(tables.get("beta_bucket_summary", pd.DataFrame()), "Beta bucket"),
        *_top_summary_rows(
            tables.get("volatility_bucket_summary", pd.DataFrame()),
            "Volatility bucket",
        ),
        "",
        "## Value Exposure Attribution",
        *_top_summary_rows(tables.get("value_bucket_summary", pd.DataFrame()), "Value bucket"),
        "",
        "## Residual Alpha Stability",
        (
            "- Full-sample annualized residual alpha: "
            f"`{residual_summary.get('full_sample_annualized_residual_alpha', np.nan):.4f}`"
        ),
        (
            "- Full-sample HAC t-stat: "
            f"`{residual_summary.get('full_sample_t_stat', np.nan):.4f}`"
        ),
    ]
    target = root / "attribution_report.md"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def generate_attribution_report(
    run_path: str | Path,
    config: AppConfig,
    output_root: str | Path | None = None,
) -> AttributionBundle:
    run_root = Path(run_path).resolve()
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    decision_path = run_root / "research_decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8")) if decision_path.exists() else {}
    model = str(decision.get("selected_model") or "rule_score")
    evaluation_root = run_root / "holdout_evaluation" / model
    if not evaluation_root.exists():
        evaluation_root = run_root
    root = Path(output_root).resolve() if output_root else run_root / "attribution"
    root.mkdir(parents=True, exist_ok=True)

    daily = _read_csv(evaluation_root / "daily_returns.csv")
    holdings = _read_csv(evaluation_root / "holdings.csv")
    prices = pd.read_parquet(resolve_path(config.paths.silver_data) / "prices" / "prices_daily.parquet")
    features = pd.read_parquet(run_root / "feature_snapshot.parquet")
    exposures = _read_csv(run_root / "style_factor_exposures.csv")
    factor_returns = load_fama_french_daily(config.research_validation.style_factor_cache)

    ff_attr, ff_summary, residual = fama_french_contribution_attribution(
        daily,
        factor_returns,
        exposures,
    )
    sector_daily, sector_summary = sector_attribution(holdings, prices)
    beta_daily, beta_summary = bucket_attribution(
        holdings,
        prices,
        features,
        factor_names=["beta_to_spy"],
        exposure_name="beta_to_spy",
    )
    vol_daily, vol_summary = bucket_attribution(
        holdings,
        prices,
        features,
        factor_names=VOLATILITY_FACTORS,
        exposure_name="volatility_score",
    )
    value_daily, value_summary = bucket_attribution(
        holdings,
        prices,
        features,
        factor_names=VALUE_FACTORS,
        exposure_name="value_score",
    )
    residual_frame, residual_summary = residual_alpha_stability(residual)

    tables = {
        "fama_french_contribution": ff_attr,
        "fama_french_summary": ff_summary,
        "sector_attribution": sector_daily,
        "sector_summary": sector_summary,
        "beta_bucket_attribution": beta_daily,
        "beta_bucket_summary": beta_summary,
        "volatility_bucket_attribution": vol_daily,
        "volatility_bucket_summary": vol_summary,
        "value_exposure_attribution": value_daily,
        "value_bucket_summary": value_summary,
        "residual_alpha_stability": residual_frame,
    }
    for name, frame in tables.items():
        frame.to_csv(root / f"{name}.csv", index=False)
    (root / "residual_alpha_stability.json").write_text(
        json.dumps(residual_summary, indent=2),
        encoding="utf-8",
    )
    markdown = _write_markdown(
        root,
        str(manifest.get("run_id", run_root.name)),
        tables,
        residual_summary,
    )
    manifest_path = root / "attribution_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "source_run_id": manifest.get("run_id", run_root.name),
                "source_run_path": str(run_root),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "model": model,
                "outputs": sorted(path.name for path in root.iterdir() if path.is_file()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return AttributionBundle(root=root, markdown_report=markdown, manifest=manifest_path)
