from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from qss.config.schema import AppConfig


class FactorMetadata(BaseModel):
    name: str
    category: str
    description: str
    inputs: list[str]
    lookback_days: int | None = None
    skip_days: int = 0
    expected_horizon: str
    cost_sensitivity: str
    point_in_time_requirements: list[str] = Field(default_factory=list)
    leakage_checks: list[str] = Field(default_factory=list)
    version: str = "v1"


def _metadata(
    name: str,
    category: str,
    description: str,
    inputs: list[str],
    lookback_days: int | None,
    expected_horizon: str,
    cost_sensitivity: str,
    *,
    skip_days: int = 0,
    point_in_time_requirements: list[str] | None = None,
) -> FactorMetadata:
    return FactorMetadata(
        name=name,
        category=category,
        description=description,
        inputs=inputs,
        lookback_days=lookback_days,
        skip_days=skip_days,
        expected_horizon=expected_horizon,
        cost_sensitivity=cost_sensitivity,
        point_in_time_requirements=point_in_time_requirements or ["as_of_date"],
        leakage_checks=["look_ahead", "timestamp_alignment", "survivorship"],
    )


FACTOR_METADATA = {
    "earnings_yield": _metadata("earnings_yield", "value", "Net income divided by market capitalization.", ["net_income", "market_cap"], None, "60d+", "low", point_in_time_requirements=["filing_date", "available_date"]),
    "free_cash_flow_yield": _metadata("free_cash_flow_yield", "value", "Free cash flow divided by market capitalization.", ["operating_cash_flow", "capital_expenditure", "market_cap"], None, "60d+", "low", point_in_time_requirements=["filing_date", "available_date"]),
    "book_to_market": _metadata("book_to_market", "value", "Shareholders equity divided by market capitalization.", ["shareholders_equity", "market_cap"], None, "60d+", "low", point_in_time_requirements=["filing_date", "available_date"]),
    "sales_yield": _metadata("sales_yield", "value", "Revenue divided by market capitalization.", ["revenue", "market_cap"], None, "60d+", "low", point_in_time_requirements=["filing_date", "available_date"]),
    "roe": _metadata("roe", "quality", "Net income divided by shareholders equity.", ["net_income", "shareholders_equity"], None, "60d+", "low", point_in_time_requirements=["filing_date", "available_date"]),
    "gross_margin": _metadata("gross_margin", "quality", "Gross profit divided by revenue.", ["gross_profit", "revenue"], None, "60d+", "low", point_in_time_requirements=["filing_date", "available_date"]),
    "operating_margin": _metadata("operating_margin", "quality", "Operating income divided by revenue.", ["operating_income", "revenue"], None, "60d+", "low", point_in_time_requirements=["filing_date", "available_date"]),
    "debt_to_equity": _metadata("debt_to_equity", "quality", "Total liabilities divided by shareholders equity.", ["total_liabilities", "shareholders_equity"], None, "60d+", "low", point_in_time_requirements=["filing_date", "available_date"]),
    "accruals": _metadata("accruals", "quality", "Accounting earnings less operating cash flow, scaled by assets.", ["net_income", "operating_cash_flow", "total_assets"], None, "60d+", "low", point_in_time_requirements=["filing_date", "available_date"]),
    "momentum_12_1": _metadata("momentum_12_1", "momentum", "Twelve-month price momentum excluding the latest month.", ["adj_close"], 252, "20-60d", "medium", skip_days=21),
    "momentum_6m": _metadata("momentum_6m", "momentum", "Six-month trailing price return.", ["adj_close"], 126, "20-60d", "medium"),
    "momentum_3m": _metadata("momentum_3m", "momentum", "Three-month trailing price return.", ["adj_close"], 63, "20-60d", "high"),
    "realized_vol_60d": _metadata("realized_vol_60d", "risk", "Annualized realized volatility over 60 trading days.", ["return_1d"], 60, "20-60d", "medium"),
    "realized_vol_252d": _metadata("realized_vol_252d", "risk", "Annualized realized volatility over 252 trading days.", ["return_1d"], 252, "60d+", "low"),
    "beta_to_spy": _metadata("beta_to_spy", "risk", "Trailing market beta to the configured benchmark.", ["return_1d", "benchmark_return"], 252, "60d+", "low"),
    "max_drawdown_252d": _metadata("max_drawdown_252d", "risk", "Worst trailing drawdown over 252 trading days.", ["adj_close"], 252, "60d+", "low"),
    "risk_disclosure_score": _metadata("risk_disclosure_score", "text_event", "Deterministic SEC filing risk-term intensity and event severity score.", ["filing_type", "filing_timestamp", "cached_text"], 365, "5-60d", "medium", point_in_time_requirements=["filing_timestamp", "text_cache_key"]),
}


def configured_factor_metadata(config: AppConfig) -> list[FactorMetadata]:
    names = {
        name for group in config.factor_groups.values() for name in group.factors
    }
    missing = sorted(names - set(FACTOR_METADATA))
    if missing:
        raise ValueError(f"Missing factor metadata definitions: {missing}")
    return [FACTOR_METADATA[name] for name in sorted(names)]


def write_factor_metadata_snapshot(config: AppConfig, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            [item.model_dump(mode="json") for item in configured_factor_metadata(config)],
            indent=2,
        ),
        encoding="utf-8",
    )
    return target
