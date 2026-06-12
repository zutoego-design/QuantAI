from __future__ import annotations

import gzip
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from qss.config.schema import AppConfig
from qss.data.fundamentals import observations_to_wide
from qss.data.identifiers import normalize_symbol
from qss.data.quality import check_data_quality, write_quality_report
from qss.data.storage import append_or_replace_parquet, append_with_source_precedence
from qss.logging_utils import logger

FIELD_MAP: dict[str, tuple[str, ...]] = {
    "revenue": ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"),
    "gross_profit": ("GrossProfit",),
    "operating_income": ("OperatingIncomeLoss",),
    "net_income": ("NetIncomeLoss", "ProfitLoss"),
    "total_assets": ("Assets",),
    "total_liabilities": ("Liabilities",),
    "shareholders_equity": (
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ),
    "operating_cash_flow": ("NetCashProvidedByUsedInOperatingActivities",),
    "capital_expenditure": ("PaymentsToAcquirePropertyPlantAndEquipment", "CapitalExpendituresIncurredButNotYetPaid"),
    "shares_outstanding": ("EntityCommonStockSharesOutstanding", "CommonStockSharesOutstanding"),
}


@dataclass
class FundamentalsIngestionResult:
    fundamentals: pd.DataFrame
    quality_report: pd.DataFrame


def _sic_to_sector(sic: int | str | None) -> str:
    try:
        code = int(sic or 0)
    except (TypeError, ValueError):
        return "Unknown"
    if 100 <= code <= 999:
        return "Consumer Staples"
    if 1000 <= code <= 1499:
        return "Materials"
    if 1500 <= code <= 1799:
        return "Industrials"
    if 2000 <= code <= 2199:
        return "Consumer Staples"
    if 2200 <= code <= 2399:
        return "Consumer Discretionary"
    if 2400 <= code <= 2799:
        return "Materials"
    if 2830 <= code <= 2839 or 3840 <= code <= 3859 or 8000 <= code <= 8099:
        return "Health Care"
    if 2800 <= code <= 2899:
        return "Materials"
    if 2900 <= code <= 2999:
        return "Energy"
    if 3500 <= code <= 3699 or 7370 <= code <= 7379:
        return "Information Technology"
    if 4000 <= code <= 4799:
        return "Industrials"
    if 4800 <= code <= 4899:
        return "Communication Services"
    if 4900 <= code <= 4999:
        return "Utilities"
    if 5000 <= code <= 5199:
        return "Industrials"
    if 5200 <= code <= 5999:
        return "Consumer Discretionary"
    if 6000 <= code <= 6799:
        return "Financials"
    if 7000 <= code <= 7299 or 7800 <= code <= 7999:
        return "Consumer Discretionary"
    if 7300 <= code <= 7799:
        return "Industrials"
    return "Unknown"


def _headers(user_agent: str) -> dict[str, str]:
    return {
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
    }


def _load_ticker_cik_map(user_agent: str) -> dict[str, str]:
    url = "https://www.sec.gov/files/company_tickers.json"
    response = requests.get(url, headers=_headers(user_agent), timeout=60)
    response.raise_for_status()
    payload = response.json()
    mapping = {}
    for record in payload.values():
        mapping[normalize_symbol(record["ticker"])] = f'{int(record["cik_str"]):010d}'
    return mapping


def _pick_unit(units: dict[str, list[dict]], prefer: str) -> list[dict]:
    if prefer in units:
        return units[prefer]
    for key, items in units.items():
        if key.startswith(prefer):
            return items
    return []


def _fact_namespaces(payload: dict, output_field: str) -> list[dict]:
    facts = payload.get("facts", {})
    if output_field == "shares_outstanding":
        return [facts.get("dei", {}), facts.get("us-gaap", {})]
    return [facts.get("us-gaap", {})]


def _extract_company_facts(payload: dict, symbol: str, filing_types: Iterable[str]) -> pd.DataFrame:
    filing_types = set(filing_types)
    rows: list[dict] = []

    for output_field, tags in FIELD_MAP.items():
        found_for_field = False
        for namespace in _fact_namespaces(payload, output_field):
            for tag in tags:
                fact = namespace.get(tag)
                if not fact:
                    continue
                unit_name = "shares" if output_field == "shares_outstanding" else "USD"
                entries = _pick_unit(fact.get("units", {}), unit_name)
                for item in entries:
                    if item.get("form") not in filing_types:
                        continue
                    if "end" not in item or "filed" not in item:
                        continue
                    rows.append(
                        {
                            "symbol": symbol,
                            "metric": output_field,
                            "value": float(item.get("val")) if item.get("val") is not None else np.nan,
                            "unit": unit_name,
                            "period_end_date": pd.Timestamp(item["end"]).normalize(),
                            "filing_date": pd.Timestamp(item["filed"]).normalize(),
                            "available_date": pd.Timestamp(item["filed"]).normalize(),
                            "fiscal_year": int(item.get("fy") or pd.Timestamp(item["end"]).year),
                            "fiscal_period": item.get("fp") or "FY",
                            "form": item.get("form"),
                            "accession": item.get("accn", ""),
                            "source": "sec_edgar",
                            "quality_status": "live",
                            "ingestion_time": pd.Timestamp.utcnow().tz_localize(None),
                        }
                    )
                if entries:
                    found_for_field = True
                    break
            if found_for_field:
                break

    observations = pd.DataFrame(rows)
    if observations.empty:
        return observations
    observations.loc[
        observations["metric"] == "capital_expenditure", "value"
    ] = observations.loc[observations["metric"] == "capital_expenditure", "value"].abs()
    frame = observations_to_wide(observations)
    if frame.empty:
        return frame
    numeric_cols = [col for col in FIELD_MAP if col != "shares_outstanding"]
    for col in numeric_cols + ["shares_outstanding"]:
        if col not in frame:
            frame[col] = np.nan
    frame["capital_expenditure"] = frame["capital_expenditure"].abs()
    frame["free_cash_flow"] = frame["operating_cash_flow"] - frame["capital_expenditure"].fillna(0.0)
    frame.attrs["observations"] = observations
    return frame.sort_values(["symbol", "period_end_date", "filing_date"]).drop_duplicates(
        ["symbol", "period_end_date", "filing_date"], keep="last"
    )


def _synthetic_fundamentals(symbols: list[str], start_date: str = "2014-01-01", end_date: str | None = None) -> pd.DataFrame:
    end = pd.Timestamp(end_date or pd.Timestamp.today()).normalize()
    quarter_ends = pd.date_range(pd.Timestamp(start_date), end, freq="QE")
    frames: list[pd.DataFrame] = []
    for symbol in symbols:
        seed = int(hashlib.md5(symbol.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        base_revenue = rng.uniform(5e9, 50e9)
        shares = rng.uniform(5e8, 8e9)
        margin = rng.uniform(0.08, 0.28)
        for idx, period_end in enumerate(quarter_ends):
            growth = 1 + idx * rng.uniform(0.002, 0.02)
            revenue = base_revenue * growth * rng.uniform(0.96, 1.04)
            gross_profit = revenue * rng.uniform(0.35, 0.70)
            operating_income = revenue * margin * rng.uniform(0.9, 1.1)
            net_income = operating_income * rng.uniform(0.72, 0.88)
            assets = revenue * rng.uniform(1.4, 3.0)
            liabilities = assets * rng.uniform(0.35, 0.70)
            equity = max(assets - liabilities, revenue * 0.3)
            ocf = net_income * rng.uniform(0.9, 1.2)
            capex = revenue * rng.uniform(0.02, 0.08)
            frames.append(
                pd.DataFrame(
                    {
                        "symbol": [symbol],
                        "period_end_date": [period_end.normalize()],
                        "filing_date": [(period_end + pd.Timedelta(days=45)).normalize()],
                        "available_date": [(period_end + pd.Timedelta(days=45)).normalize()],
                        "fiscal_year": [period_end.year],
                        "fiscal_period": [f"Q{((period_end.month - 1) // 3) + 1}"],
                        "revenue": [revenue],
                        "gross_profit": [gross_profit],
                        "operating_income": [operating_income],
                        "net_income": [net_income],
                        "total_assets": [assets],
                        "total_liabilities": [liabilities],
                        "shareholders_equity": [equity],
                        "operating_cash_flow": [ocf],
                        "capital_expenditure": [capex],
                        "free_cash_flow": [ocf - capex],
                        "shares_outstanding": [shares],
                        "source": ["synthetic_fallback"],
                        "quality_status": ["invalid_for_research"],
                        "ingestion_time": [pd.Timestamp.utcnow().tz_localize(None)],
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def ingest_fundamentals(config: AppConfig, tickers: list[str]) -> FundamentalsIngestionResult:
    tickers = [normalize_symbol(ticker) for ticker in tickers]
    frames: list[pd.DataFrame] = []
    observation_frames: list[pd.DataFrame] = []
    metadata_rows: list[dict] = []
    user_agent = os.getenv("SEC_USER_AGENT") or config.data_sources.fundamentals.user_agent

    try:
        cik_map = _load_ticker_cik_map(user_agent)
    except Exception as exc:
        logger.warning("Failed to load SEC ticker mapping: {}", exc)
        cik_map = {}

    raw_root = (
        Path(config.paths.raw_data)
        / "fundamentals"
        / pd.Timestamp.now(tz="UTC").strftime("%Y%m%d")
    )
    raw_root.mkdir(parents=True, exist_ok=True)
    rate_lock = threading.Lock()
    last_request = [0.0]

    def fetch_ticker(ticker: str) -> tuple[str, dict | None, Exception | None]:
        cik = cik_map.get(ticker)
        if not cik:
            return ticker, None, ValueError("No SEC CIK mapping")
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        try:
            with rate_lock:
                wait = max(0.0, 0.12 - (time.monotonic() - last_request[0]))
                if wait:
                    time.sleep(wait)
                last_request[0] = time.monotonic()
            response = requests.get(url, headers=_headers(user_agent), timeout=60)
            response.raise_for_status()
            return ticker, response.json(), None
        except Exception as exc:
            return ticker, None, exc

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(fetch_ticker, ticker) for ticker in tickers]
        for future in as_completed(futures):
            ticker, payload, error = future.result()
            if error is not None or payload is None:
                logger.warning("SEC fundamentals fetch failed for {}: {}", ticker, error)
                continue
            with gzip.open(raw_root / f"{ticker}.json.gz", "wt", encoding="utf-8") as handle:
                json.dump(payload, handle)
            metadata_rows.append(
                {
                    "symbol": ticker,
                    "sic": payload.get("sic"),
                    "sic_description": payload.get("sicDescription"),
                    "sector": _sic_to_sector(payload.get("sic")),
                    "metadata_source": "sec_sic",
                    "metadata_ingestion_time": pd.Timestamp.now(tz="UTC").tz_localize(None),
                }
            )
            frame = _extract_company_facts(payload, ticker, config.data_sources.fundamentals.filing_types)
            if not frame.empty:
                observations = frame.attrs.get("observations")
                if isinstance(observations, pd.DataFrame) and not observations.empty:
                    observation_frames.append(observations)
                frames.append(frame)

    live = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    missing = sorted(set(tickers) - set(live["symbol"].unique())) if not live.empty else tickers
    if missing and config.data_sources.fundamentals.fallback_to_synthetic:
        if config.runtime.research_mode and not config.runtime.allow_synthetic:
            raise RuntimeError(
                f"Research mode forbids synthetic fundamentals; missing SEC coverage for {len(missing)} symbols."
            )
        logger.warning("Synthetic fundamentals fallback activated for {} symbols.", len(missing))
        live = pd.concat([live, _synthetic_fundamentals(missing)], ignore_index=True)

    live = live.sort_values(["symbol", "available_date", "period_end_date"])
    if not live.empty and "shares_outstanding" in live.columns:
        live["shares_outstanding"] = live.groupby("symbol")["shares_outstanding"].transform(lambda s: s.ffill().bfill())
    quality = check_data_quality("fundamentals_quarterly", live, ["symbol", "period_end_date", "filing_date"])
    observations = (
        pd.concat(observation_frames, ignore_index=True)
        if observation_frames
        else pd.DataFrame()
    )
    if not observations.empty:
        append_or_replace_parquet(
            observations,
            Path(config.paths.silver_data) / "fundamentals" / "fundamental_observations.parquet",
            ["symbol", "metric", "period_end_date", "filing_date", "accession"],
        )
    if metadata_rows:
        metadata = pd.DataFrame(metadata_rows).drop_duplicates("symbol", keep="last")
        universe_root = Path(config.paths.silver_data) / "universe"
        append_or_replace_parquet(
            metadata,
            universe_root / "security_metadata.parquet",
            ["symbol"],
        )
        master_path = universe_root / "security_master.parquet"
        master = (
            pd.read_parquet(master_path)
            if master_path.exists()
            else metadata[["symbol"]].copy()
        )
        master = master.merge(
            metadata[["symbol", "sic", "sic_description", "sector"]],
            on="symbol",
            how="left",
            suffixes=("", "_sec"),
        )
        for column in ["sic", "sic_description", "sector"]:
            sec_column = f"{column}_sec"
            if sec_column in master:
                if column in master:
                    master[column] = master[sec_column].combine_first(master[column])
                else:
                    master[column] = master[sec_column]
                master = master.drop(columns=sec_column)
        master.to_parquet(master_path, index=False)
    append_with_source_precedence(
        live,
        Path(config.paths.silver_data) / "fundamentals" / "fundamentals_quarterly.parquet",
        ["symbol", "period_end_date", "filing_date"],
        {"sec_edgar": 100, "synthetic_fallback": 0},
    )
    write_quality_report(quality, Path(config.paths.reports) / "data_quality" / f"data_quality_{pd.Timestamp.today():%Y%m%d}.csv")
    return FundamentalsIngestionResult(fundamentals=live, quality_report=quality)
