from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf

from qss.config.schema import AppConfig
from qss.data.identifiers import normalize_symbol
from qss.data.quality import check_data_quality, write_quality_report
from qss.data.storage import append_with_source_precedence, read_parquet, write_parquet
from qss.ingestion.base import DataProvider
from qss.ingestion.prices_stooq import StooqPriceProvider
from qss.logging_utils import logger
from qss.utils import project_root

PRICE_COLUMNS = [
    "symbol",
    "date",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "return_1d",
    "source",
    "quality_status",
    "ingestion_time",
]


@dataclass
class PriceIngestionResult:
    prices: pd.DataFrame
    security_master: pd.DataFrame
    quality_report: pd.DataFrame


class YFinancePriceProvider(DataProvider):
    def fetch(self, tickers: list[str], start_date: str, end_date: str | None = None) -> pd.DataFrame:
        yahoo_tickers = [ticker.replace(".", "-") for ticker in tickers]
        raw = yf.download(
            tickers=yahoo_tickers,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=True,
        )
        if isinstance(raw.columns, pd.MultiIndex):
            return raw
        if raw.empty:
            return raw
        return pd.concat({yahoo_tickers[0]: raw}, axis=1)

    def normalize(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        if raw_data.empty:
            return pd.DataFrame(columns=PRICE_COLUMNS)
        frames: list[pd.DataFrame] = []
        ingestion_time = pd.Timestamp.now(tz="UTC").tz_localize(None)
        for symbol in raw_data.columns.get_level_values(0).unique():
            sub = raw_data[symbol].copy()
            if sub.empty:
                continue
            sub = sub.reset_index().rename(
                columns={
                    "Date": "date",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Adj Close": "adj_close",
                    "Volume": "volume",
                }
            )
            if "adj_close" not in sub:
                sub["adj_close"] = sub["close"]
            sub["symbol"] = normalize_symbol(symbol)
            sub["source"] = "yfinance"
            sub["quality_status"] = "live"
            sub["ingestion_time"] = ingestion_time
            frames.append(
                sub[
                    [
                        "symbol",
                        "date",
                        "open",
                        "high",
                        "low",
                        "close",
                        "adj_close",
                        "volume",
                        "source",
                        "quality_status",
                        "ingestion_time",
                    ]
                ]
            )
        if not frames:
            return pd.DataFrame(columns=PRICE_COLUMNS)
        prices = pd.concat(frames, ignore_index=True)
        prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None)
        prices = prices.sort_values(["symbol", "date"])
        prices["return_1d"] = prices.groupby("symbol")["adj_close"].pct_change()
        return prices[PRICE_COLUMNS]


def _stooq_supports(symbol: str) -> bool:
    return not symbol.startswith("^")


def _retry_yfinance_symbols(
    provider: YFinancePriceProvider,
    symbols: list[str],
    start_date: str,
    end_date: str | None,
    attempts: int = 2,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for symbol in symbols:
        for attempt in range(1, attempts + 1):
            try:
                normalized = provider.normalize(
                    provider.fetch([symbol], start_date, end_date)
                )
                normalized = normalized.loc[normalized["symbol"] == symbol]
                if not normalized.empty and normalized["adj_close"].notna().any():
                    frames.append(normalized)
                    break
            except Exception as exc:
                logger.warning(
                    "Yahoo retry {}/{} failed for {}: {}",
                    attempt,
                    attempts,
                    symbol,
                    exc,
                )
            if attempt < attempts:
                time.sleep(0.5 * attempt)
    return (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=PRICE_COLUMNS)
    )


def _flatten_etf_tickers(config: AppConfig) -> list[str]:
    tickers: list[str] = []
    for group in config.data_sources.etf.tickers.values():
        tickers.extend(group.values())
    tickers.append(config.strategy.benchmark)
    tickers.append(config.backtest.primary_benchmark)
    tickers.append(config.backtest.secondary_benchmark)
    return sorted(set(tickers))


def _load_seed_security_master(config: AppConfig) -> pd.DataFrame:
    existing_path = (
        project_root()
        / config.paths.silver_data
        / "universe"
        / "security_master.parquet"
    )
    if existing_path.exists():
        existing = pd.read_parquet(existing_path)
        if not existing.empty and "symbol" in existing:
            existing["symbol"] = existing["symbol"].map(normalize_symbol)
            return existing
    seed_path = project_root() / config.universe.seed_metadata_path
    frame = pd.read_csv(seed_path)
    frame["symbol"] = frame["symbol"].map(normalize_symbol)
    frame["is_active"] = frame["is_active"].astype(bool)
    frame["source"] = "seed_config"
    frame["ingestion_time"] = pd.Timestamp.utcnow().tz_localize(None)
    return frame


def _fallback_prices(symbols: Iterable[str], start_date: str, end_date: str | None = None) -> pd.DataFrame:
    end = pd.Timestamp(end_date) if end_date else pd.Timestamp.today().normalize()
    dates = pd.bdate_range(pd.Timestamp(start_date), end)
    frames: list[pd.DataFrame] = []
    for offset, symbol in enumerate(symbols, start=1):
        base = 50 + 3 * offset
        noise = pd.Series(range(len(dates)), dtype="float64")
        close = base * (1 + 0.0005 * noise + 0.02 * pd.Series(np.sin(noise / 17 + offset)))
        frame = pd.DataFrame(
            {
                "symbol": symbol,
                "date": dates,
                "open": close * 0.995,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "adj_close": close,
                "volume": 1_000_000 + offset * 10_000,
                "source": "synthetic_fallback",
                "quality_status": "invalid_for_research",
                "ingestion_time": pd.Timestamp.utcnow().tz_localize(None),
            }
        )
        frame["return_1d"] = frame["adj_close"].pct_change()
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)[PRICE_COLUMNS]


def _valid_price_symbols(
    prices: pd.DataFrame,
    start_date: str,
    end_date: str | None,
) -> set[str]:
    if prices.empty or not {"symbol", "adj_close"}.issubset(prices.columns):
        return set()
    frame = prices.loc[prices["adj_close"].notna()].copy()
    if "date" in frame:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.tz_localize(None)
        frame = frame.loc[frame["date"] >= pd.Timestamp(start_date)]
        if end_date:
            frame = frame.loc[frame["date"] <= pd.Timestamp(end_date)]
    return set(frame["symbol"].dropna().astype(str).map(normalize_symbol))


def _persist_prices(
    config: AppConfig,
    prices: pd.DataFrame,
    seed_master: pd.DataFrame,
) -> pd.DataFrame:
    prices = prices.sort_values(["symbol", "date"]).drop_duplicates(
        ["symbol", "date"],
        keep="last",
    )
    quality = check_data_quality(
        "prices_daily",
        prices,
        primary_keys=["symbol", "date"],
        as_of_date=pd.Timestamp.today(),
    )
    timestamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
    write_parquet(
        prices,
        Path(config.paths.raw_data) / "prices" / f"prices_raw_{timestamp}.parquet",
    )
    append_with_source_precedence(
        prices,
        Path(config.paths.silver_data) / "prices" / "prices_daily.parquet",
        ["symbol", "date"],
        {"yfinance": 100, "stooq": 80, "synthetic_fallback": 0},
    )
    persisted_path = Path(config.paths.silver_data) / "prices" / "prices_daily.parquet"
    persisted = read_parquet(persisted_path)
    if not persisted.empty and {"symbol", "date", "adj_close"}.issubset(persisted):
        persisted = persisted.copy()
        persisted["date"] = pd.to_datetime(persisted["date"]).dt.tz_localize(None)
        persisted["adj_close"] = pd.to_numeric(persisted["adj_close"], errors="coerce")
        persisted = persisted.sort_values(["symbol", "date"])
        persisted["return_1d"] = persisted.groupby("symbol")["adj_close"].pct_change()
        write_parquet(persisted, persisted_path)
    write_parquet(
        seed_master,
        Path(config.paths.silver_data) / "universe" / "security_master.parquet",
    )
    write_quality_report(
        quality,
        Path(config.paths.reports)
        / "data_quality"
        / f"data_quality_{pd.Timestamp.today():%Y%m%d}.csv",
    )
    return quality


def ingest_prices(
    config: AppConfig,
    start_date: str,
    end_date: str | None = None,
    tickers: list[str] | None = None,
) -> PriceIngestionResult:
    seed_master = _load_seed_security_master(config)
    requested = tickers or seed_master["symbol"].tolist()
    all_tickers = sorted(set([*requested, *_flatten_etf_tickers(config)]))
    stored_prices = read_parquet(
        Path(config.paths.silver_data) / "prices" / "prices_daily.parquet"
    )
    provider = YFinancePriceProvider()
    fallback = StooqPriceProvider()

    primary_frames: list[pd.DataFrame] = []
    batch_size = max(int(config.data_sources.prices.batch_size), 1)
    for offset in range(0, len(all_tickers), batch_size):
        batch = all_tickers[offset : offset + batch_size]
        try:
            raw = provider.fetch(batch, start_date, end_date)
            normalized = provider.normalize(raw)
            if not normalized.empty:
                primary_frames.append(normalized)
        except Exception as exc:
            logger.warning("Primary price provider failed for batch {}: {}", batch, exc)
    prices = (
        pd.concat(primary_frames, ignore_index=True)
        if primary_frames
        else pd.DataFrame(columns=PRICE_COLUMNS)
    )
    prices = prices.dropna(subset=["adj_close"], how="all").copy()

    valid_symbols = _valid_price_symbols(prices, start_date, end_date)
    missing = sorted(set(all_tickers) - valid_symbols)
    if missing:
        logger.warning(
            "Batch download missed {} symbols; retrying Yahoo individually. Sample: {}",
            len(missing),
            ", ".join(missing[:20]),
        )
        retried = _retry_yfinance_symbols(
            provider,
            missing,
            start_date,
            end_date,
        )
        if not retried.empty:
            prices = pd.concat([prices, retried], ignore_index=True)

    valid_symbols = _valid_price_symbols(prices, start_date, end_date)
    missing = sorted(set(all_tickers) - valid_symbols)
    if missing:
        logger.warning(
            "Yahoo still missing {} symbols; using Stooq where supported. Sample: {}",
            len(missing),
            ", ".join(missing[:20]),
        )
        fallback_frames: list[pd.DataFrame] = []
        for symbol in missing:
            if not _stooq_supports(symbol):
                logger.warning(
                    "Skipping Stooq fallback for index symbol {}; Yahoo is required.",
                    symbol,
                )
                continue
            try:
                fallback_frames.append(fallback.normalize(fallback.fetch(symbol, start_date, end_date)))
            except Exception as exc:
                logger.warning("Fallback provider failed for {}: {}", symbol, exc)
        if fallback_frames:
            prices = pd.concat([prices, *fallback_frames], ignore_index=True)

    valid_symbols = _valid_price_symbols(prices, start_date, end_date)
    if config.runtime.research_mode:
        valid_symbols |= _valid_price_symbols(stored_prices, start_date, end_date)
    still_missing = sorted(set(all_tickers) - valid_symbols)
    if still_missing:
        if config.runtime.research_mode and not (
            config.runtime.allow_synthetic
            or config.data_sources.prices.allow_synthetic_fallback
        ):
            critical_symbols = {
                config.backtest.primary_benchmark,
                config.backtest.secondary_benchmark,
                config.strategy.benchmark,
            }
            missing_critical = sorted(critical_symbols & set(still_missing))
            requested_symbols = set(requested)
            research_coverage = (
                len(requested_symbols & valid_symbols) / len(requested_symbols)
                if requested_symbols
                else 0.0
            )
            if (
                missing_critical
                or research_coverage < config.universe.min_long_price_coverage
            ):
                _persist_prices(config, prices, seed_master)
                raise RuntimeError(
                    "Research price coverage is below the strict threshold: "
                    f"{research_coverage:.1%}; missing critical={missing_critical}; "
                    f"unavailable={len(still_missing)}. "
                    "Successful live rows were saved; rerun to resume from the cache."
                )
            logger.warning(
                "Live price coverage {:.1%} passes the strict threshold; "
                "{} unavailable symbols are retained as explicit coverage gaps.",
                research_coverage,
                len(still_missing),
            )
        else:
            logger.warning(
                "Synthetic price fallback activated for {} symbols.",
                len(still_missing),
            )
            prices = pd.concat(
                [prices, _fallback_prices(still_missing, start_date, end_date)],
                ignore_index=True,
            )

    quality = _persist_prices(config, prices, seed_master)
    return PriceIngestionResult(prices=prices, security_master=seed_master, quality_report=quality)
