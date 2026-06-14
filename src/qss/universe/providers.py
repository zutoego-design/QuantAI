from __future__ import annotations

import hashlib
import io
import os
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests

from qss.config.schema import AppConfig
from qss.data.identifiers import normalize_symbol
from qss.data.providers import UniverseProvider
from qss.data.storage import query_parquet, read_parquet, resolve_path

OPERATING_TYPES = {"Common Stock", "ADR", "REIT"}


def permanent_security_id(exchange: str, name: str, first_symbol: str) -> str:
    normalized_name = "" if pd.isna(name) else str(name).strip().upper()
    normalized_symbol = (
        "" if pd.isna(first_symbol) else normalize_symbol(str(first_symbol))
    )
    identity = normalized_name or normalized_symbol
    canonical = f"{exchange.upper()}|{identity}"
    return "sec_" + hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:20]


def classify_security(name: str, asset_type: str | None = None, flags: dict | None = None) -> str:
    text = f"{name} {asset_type or ''}".upper()
    flags = flags or {}
    if flags.get("test_issue") in {"Y", True}:
        return "Test"
    if flags.get("etf") in {"Y", True} or " ETF" in f" {text}":
        return "ETF"
    if "ETN" in text:
        return "ETN"
    if any(token in text for token in ("WARRANT", " WT", "RIGHT")):
        return "Warrant"
    if any(token in text for token in (" UNIT", "UNITS")):
        return "Unit"
    if any(token in text for token in ("PREFERRED", "PREF ", " DEPOSITARY SH")):
        return "Preferred"
    if "REIT" in text or "REAL ESTATE INVESTMENT TRUST" in text:
        return "REIT"
    if "ADR" in text or "ADS" in text or "DEPOSITARY SHARE" in text:
        return "ADR"
    if asset_type and asset_type.lower() not in {"stock", "common stock", "equity"}:
        return asset_type
    return "Common Stock"


@dataclass
class ParquetUniverseProvider(UniverseProvider):
    membership_path: Path
    security_master_path: Path | None = None

    def snapshot(self, as_of_date: str | pd.Timestamp) -> pd.DataFrame:
        as_of = pd.Timestamp(as_of_date).normalize()
        membership_target = resolve_path(self.membership_path)
        if membership_target.is_dir():
            parquet_source = membership_target / "**" / "*.parquet"
            membership = query_parquet(
                f"""
                SELECT *
                FROM membership
                WHERE date = (
                    SELECT MAX(date) FROM membership WHERE date <= DATE '{as_of.date()}'
                )
                  AND included = TRUE
                """,
                membership=parquet_source,
            )
        else:
            membership = read_parquet(membership_target)
        if membership.empty:
            return membership
        membership["date"] = pd.to_datetime(membership["date"]).dt.normalize()
        available = membership.loc[membership["date"] <= as_of]
        if available.empty:
            return available
        latest_date = available["date"].max()
        result = available.loc[
            (available["date"] == latest_date) & available["included"]
        ].copy()
        if self.security_master_path is not None and "security_id" in result:
            master = read_parquet(self.security_master_path)
            if not master.empty:
                extra = [c for c in master.columns if c not in result.columns or c == "security_id"]
                result = result.merge(master[extra].drop_duplicates("security_id"), on="security_id", how="left")
        return result


class AlphaVantageListingProvider:
    endpoint = "https://www.alphavantage.co/query"

    def __init__(
        self,
        api_key: str | None = None,
        timeout: int = 60,
        retry_delays: tuple[float, ...] = (5.0, 15.0),
    ):
        self.api_key = api_key or os.getenv("ALPHAVANTAGE_API_KEY")
        self.timeout = timeout
        self.retry_delays = retry_delays

    def fetch(self, as_of_date: str | pd.Timestamp) -> pd.DataFrame:
        if not self.api_key:
            raise RuntimeError("ALPHAVANTAGE_API_KEY is required for historical listing status.")
        params = {
            "function": "LISTING_STATUS",
            "date": str(pd.Timestamp(as_of_date).date()),
            "state": "active",
            "apikey": self.api_key,
        }
        attempts = len(self.retry_delays) + 1
        for attempt in range(attempts):
            try:
                response = requests.get(
                    self.endpoint,
                    params=params,
                    timeout=self.timeout,
                )
                response.raise_for_status()
            except requests.RequestException as exc:
                detail = (
                    "Alpha Vantage network request failed "
                    f"({type(exc).__name__}); credentials were redacted."
                )
                if attempt >= len(self.retry_delays):
                    raise RuntimeError(detail) from None
                time.sleep(self.retry_delays[attempt])
                continue
            text = response.text.strip()
            if text.startswith("{"):
                try:
                    payload = response.json()
                except requests.JSONDecodeError:
                    payload = {"response": text[:300]}
                if payload:
                    detail = next(
                        (
                            str(payload[key])
                            for key in ("Error Message", "Information", "Note")
                            if payload.get(key)
                        ),
                        str(payload)[:300],
                    )
                    raise RuntimeError(f"Alpha Vantage API unavailable: {detail}")
                detail = (
                    "Alpha Vantage returned an empty JSON response, usually caused "
                    "by temporary throttling."
                )
            elif not text:
                detail = "Alpha Vantage returned an empty response."
            else:
                frame = pd.read_csv(io.StringIO(text))
                required = {"symbol", "name", "exchange", "assetType"}
                if required.issubset(frame.columns) and not frame.empty:
                    return normalize_alpha_vantage(frame, pd.Timestamp(as_of_date))
                detail = (
                    "Alpha Vantage returned malformed listing data; "
                    f"columns={sorted(frame.columns)}."
                )
            if attempt >= len(self.retry_delays):
                raise RuntimeError(detail)
            time.sleep(self.retry_delays[attempt])
        raise RuntimeError("Alpha Vantage retry loop ended unexpectedly.")


def normalize_alpha_vantage(frame: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.DataFrame:
    rename = {
        "symbol": "symbol",
        "name": "name",
        "exchange": "exchange",
        "assetType": "asset_type",
        "ipoDate": "listing_date",
        "delistingDate": "delisting_date",
        "status": "status",
    }
    result = frame.rename(columns=rename).copy()
    for required in rename.values():
        if required not in result:
            result[required] = pd.NA
    result["symbol"] = result["symbol"].astype(str).map(normalize_symbol)
    result["exchange"] = result["exchange"].astype(str).str.upper()
    result = result.loc[result["exchange"].isin(["NASDAQ", "XNAS"])]
    result["security_type"] = [
        classify_security(str(name), str(asset))
        for name, asset in zip(result["name"], result["asset_type"], strict=False)
    ]
    result["security_id"] = [
        permanent_security_id("XNAS", str(name), symbol)
        for name, symbol in zip(result["name"], result["symbol"], strict=False)
    ]
    result["date"] = pd.Timestamp(as_of_date).normalize()
    result["source"] = "alpha_vantage_listing_status"
    return result[
        [
            "date",
            "security_id",
            "symbol",
            "name",
            "exchange",
            "security_type",
            "listing_date",
            "delisting_date",
            "status",
            "source",
        ]
    ].reset_index(drop=True)


class NasdaqTraderProvider:
    endpoint = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"

    def fetch(self) -> pd.DataFrame:
        response = requests.get(self.endpoint, timeout=60)
        response.raise_for_status()
        frame = pd.read_csv(io.StringIO(response.text), sep="|")
        symbols = frame["Symbol"].astype("string").str.strip()
        frame = frame.loc[
            symbols.notna()
            & symbols.ne("")
            & ~symbols.str.startswith("File Creation Time", na=False)
        ].copy()
        frame["Security Name"] = frame["Security Name"].fillna(frame["Symbol"])
        result = pd.DataFrame(
            {
                "symbol": frame["Symbol"].astype(str).map(normalize_symbol),
                "name": frame["Security Name"].astype(str),
                "exchange": "XNAS",
                "test_issue": frame.get("Test Issue", "N"),
                "etf": frame.get("ETF", "N"),
                "financial_status": frame.get("Financial Status", pd.NA),
                "source": "nasdaq_trader",
            }
        )
        result["security_type"] = [
            classify_security(name, flags={"test_issue": test, "etf": etf})
            for name, test, etf in zip(
                result["name"], result["test_issue"], result["etf"], strict=False
            )
        ]
        result["security_id"] = [
            permanent_security_id("XNAS", name, symbol)
            for name, symbol in zip(result["name"], result["symbol"], strict=False)
        ]
        result["date"] = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
        return result


class MassiveTickerProvider:
    endpoint = "https://api.massive.com/v3/reference/tickers"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("MASSIVE_API_KEY") or os.getenv("POLYGON_API_KEY")

    def fetch(self, as_of_date: str | pd.Timestamp) -> pd.DataFrame:
        if not self.api_key:
            raise RuntimeError("MASSIVE_API_KEY or POLYGON_API_KEY is required for validation.")
        params = {
            "market": "stocks",
            "exchange": "XNAS",
            "active": "true",
            "date": str(pd.Timestamp(as_of_date).date()),
            "limit": 1000,
            "apiKey": self.api_key,
        }
        rows: list[dict] = []
        url = f"{self.endpoint}?{urlencode(params)}"
        while url:
            try:
                response = requests.get(url, timeout=60)
                response.raise_for_status()
            except requests.RequestException as exc:
                raise RuntimeError(
                    "Massive reference request failed "
                    f"({type(exc).__name__}); credentials were redacted."
                ) from None
            payload = response.json()
            rows.extend(payload.get("results", []))
            next_url = payload.get("next_url")
            url = f"{next_url}&apiKey={self.api_key}" if next_url else ""
        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        result = pd.DataFrame(
            {
                "date": pd.Timestamp(as_of_date).normalize(),
                "symbol": frame["ticker"].astype(str).map(normalize_symbol),
                "name": frame["name"].astype(str),
                "exchange": "XNAS",
                "asset_type": frame.get("type", "stock"),
                "active": frame.get("active", True),
                "source": "massive_reference_tickers",
            }
        )
        result["security_type"] = [
            classify_security(name, asset)
            for name, asset in zip(result["name"], result["asset_type"], strict=False)
        ]
        result["security_id"] = [
            permanent_security_id("XNAS", name, symbol)
            for name, symbol in zip(result["name"], result["symbol"], strict=False)
        ]
        return result

    def fetch_details(
        self,
        symbol: str,
        as_of_date: str | pd.Timestamp,
    ) -> dict[str, object]:
        if not self.api_key:
            raise RuntimeError(
                "MASSIVE_API_KEY or POLYGON_API_KEY is required for ticker details."
            )
        normalized = normalize_symbol(symbol)
        provider_symbol = normalized.replace(".", "-")
        response = requests.get(
            f"{self.endpoint}/{provider_symbol}",
            params={
                "date": str(pd.Timestamp(as_of_date).date()),
                "apiKey": self.api_key,
            },
            timeout=60,
        )
        if response.status_code == 404:
            return {}
        if response.status_code == 429:
            raise RuntimeError("Massive ticker-details quota was exceeded.")
        try:
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                "Massive ticker-details request failed "
                f"({type(exc).__name__}); credentials were redacted."
            ) from None
        details = response.json().get("results") or {}
        if not details:
            return {}
        return {
            "symbol": normalized,
            "as_of_date": pd.Timestamp(as_of_date).normalize(),
            "cik": details.get("cik"),
            "sic": details.get("sic_code"),
            "sic_description": details.get("sic_description"),
            "name": details.get("name"),
            "active": details.get("active"),
            "metadata_source": "massive_ticker_details",
            "metadata_ingestion_time": pd.Timestamp.now(tz="UTC").tz_localize(None),
        }


def default_universe_provider(config: AppConfig) -> ParquetUniverseProvider:
    root = resolve_path(config.paths.silver_data) / "universe"
    return ParquetUniverseProvider(
        membership_path=root / "universe_membership.parquet",
        security_master_path=root / "security_master.parquet",
    )
