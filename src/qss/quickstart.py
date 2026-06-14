from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd
import requests

from qss.backtest.engine import BacktestResult, run_backtest
from qss.config.schema import AppConfig
from qss.data.identifiers import normalize_symbol
from qss.data.storage import read_parquet, resolve_path, write_parquet
from qss.ingestion.fred import generate_synthetic_macro
from qss.ingestion.prices_yfinance import ingest_prices
from qss.ingestion.sec_edgar import generate_synthetic_fundamentals
from qss.logging_utils import logger
from qss.universe.providers import NasdaqTraderProvider


@dataclass
class QuickstartResult:
    backtest: BacktestResult
    symbol_count: int
    price_rows: int
    fundamental_rows: int
    membership_rows: int
    macro_rows: int


SP500_CONSTITUENTS_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


class _HtmlTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = []

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._cell is not None and self._row is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None


def _read_sp500_table_without_lxml() -> pd.DataFrame:
    response = requests.get(
        SP500_CONSTITUENTS_URL,
        headers={"User-Agent": "QuantAI Quickstart universe loader"},
        timeout=60,
    )
    response.raise_for_status()
    html = response.text
    marker = 'id="constituents"'
    marker_index = html.find(marker)
    if marker_index < 0:
        raise RuntimeError("S&P 500 constituents table marker was not found.")
    start = html.rfind("<table", 0, marker_index)
    end = html.find("</table>", marker_index)
    if start < 0 or end < 0:
        raise RuntimeError("S&P 500 constituents table HTML was incomplete.")
    parser = _HtmlTableParser()
    parser.feed(html[start : end + len("</table>")])
    if len(parser.rows) < 2:
        raise RuntimeError("S&P 500 constituents table had no data rows.")
    header = parser.rows[0]
    rows = [row for row in parser.rows[1:] if len(row) == len(header)]
    return pd.DataFrame(rows, columns=header)


def _seed_master(config: AppConfig) -> pd.DataFrame:
    seed_path = resolve_path(config.universe.seed_metadata_path)
    master = pd.read_csv(seed_path)
    master["symbol"] = master["symbol"].astype(str).map(normalize_symbol)
    if "name" not in master and "company_name" in master:
        master["name"] = master["company_name"]
    if "company_name" not in master and "name" in master:
        master["company_name"] = master["name"]
    master["source"] = "quickstart_seed"
    return master


def _sp500_master() -> pd.DataFrame:
    try:
        tables = pd.read_html(SP500_CONSTITUENTS_URL)
        source = next(
            (
                table
                for table in tables
                if {"Symbol", "Security", "GICS Sector"}.issubset(table.columns)
            ),
            pd.DataFrame(),
        )
    except ImportError:
        source = _read_sp500_table_without_lxml()
    if source.empty:
        raise RuntimeError("S&P 500 constituents table was not found.")

    master = pd.DataFrame(
        {
            "symbol": source["Symbol"].astype(str).map(normalize_symbol),
            "company_name": source["Security"].astype(str),
            "name": source["Security"].astype(str),
            "exchange": "US",
            "sector": source["GICS Sector"].astype(str),
            "industry": source.get("GICS Sub-Industry", "Unknown"),
            "security_type": "Common Stock",
            "currency": "USD",
            "is_active": True,
            "source": "quickstart_sp500_wikipedia",
        }
    )
    # Yahoo's class-share symbols are inconsistent with the internal dot convention.
    # Skipping the two common cases still leaves at least 500 S&P 500 constituents.
    master = master.loc[~master["symbol"].str.contains(".", regex=False)]
    return master.reset_index(drop=True)


def _nasdaq_current_master() -> pd.DataFrame:
    local = resolve_path("data/silver/universe/security_master.parquet")
    if local.exists():
        source = read_parquet(local)
    else:
        source = NasdaqTraderProvider().fetch()
    if source.empty:
        return source

    master = pd.DataFrame(
        {
            "symbol": source["symbol"].astype(str).map(normalize_symbol),
            "company_name": source.get("name", source["symbol"]).astype(str),
            "name": source.get("name", source["symbol"]).astype(str),
            "exchange": source.get("exchange", "XNAS"),
            "sector": source.get("sector", "Unknown"),
            "industry": source.get("sic_description", "Unknown"),
            "security_type": source.get("security_type", "Common Stock"),
            "currency": "USD",
            "is_active": True,
            "source": "quickstart_nasdaq_current",
        }
    )
    master = master.loc[
        master["security_type"].isin({"Common Stock", "ADR", "REIT"})
    ].copy()
    return master.reset_index(drop=True)


def _load_quickstart_master(
    config: AppConfig,
    target_symbols: int | None = None,
) -> pd.DataFrame:
    target = int(target_symbols or config.quickstart.target_symbols)
    if target < 1:
        raise ValueError("Quickstart target_symbols must be positive.")
    target = min(target, int(config.quickstart.max_symbols))

    frames: list[pd.DataFrame] = []
    seed = _seed_master(config)
    if config.quickstart.prefer_seed_symbols:
        frames.append(seed)

    source = config.quickstart.universe_source
    if source == "seed":
        frames.append(seed)
    elif source == "nasdaq_current":
        frames.append(_nasdaq_current_master())
    elif source == "sp500":
        try:
            frames.append(_sp500_master())
        except Exception as exc:
            logger.warning(
                "Could not load S&P 500 constituents for Quickstart: {}. "
                "Falling back to local Nasdaq current universe.",
                exc,
            )
        if sum(len(frame) for frame in frames) < target:
            frames.append(_nasdaq_current_master())
    else:
        raise ValueError(f"Unsupported Quickstart universe source: {source}")

    if not frames:
        frames.append(seed)
    master = pd.concat(frames, ignore_index=True)
    master["symbol"] = master["symbol"].astype(str).map(normalize_symbol)
    master = master.loc[master["symbol"].ne("")]
    master = master.drop_duplicates("symbol", keep="first").head(target).copy()
    for column, default in [
        ("company_name", ""),
        ("name", ""),
        ("exchange", "US"),
        ("sector", "Unknown"),
        ("industry", "Unknown"),
        ("security_type", "Common Stock"),
        ("currency", "USD"),
        ("is_active", True),
        ("source", "quickstart"),
    ]:
        if column not in master:
            master[column] = default
        master[column] = master[column].fillna(default)
    master["ingestion_time"] = pd.Timestamp.now(tz="UTC").tz_localize(None)
    return master.reset_index(drop=True)


def _seed_membership(
    config: AppConfig,
    start_date: str,
    end_date: str,
    master: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if master is None:
        master = _load_quickstart_master(config)
    master = master.copy()
    master["symbol"] = master["symbol"].astype(str).str.upper()
    if "source" not in master:
        master["source"] = "quickstart"
    if "ingestion_time" not in master:
        master["ingestion_time"] = pd.Timestamp.now(tz="UTC").tz_localize(None)
    write_parquet(
        master,
        Path(config.paths.silver_data) / "universe" / "security_master.parquet",
    )

    dates = pd.date_range(
        pd.Timestamp(start_date).to_period("M").to_timestamp("M"),
        pd.Timestamp(end_date).to_period("M").to_timestamp("M"),
        freq="ME",
    )
    membership = pd.MultiIndex.from_product(
        [dates, master["symbol"].tolist()],
        names=["date", "symbol"],
    ).to_frame(index=False)
    membership = membership.merge(
        master[["symbol", "security_type"]],
        on="symbol",
        how="left",
    )
    membership["included"] = True
    membership["exclusion_reason"] = ""
    membership["source"] = "quickstart_current_membership"
    write_parquet(
        membership,
        Path(config.paths.silver_data) / "universe" / "universe_membership.parquet",
    )
    return membership


def _seed_fundamentals(
    config: AppConfig,
    symbols: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    fundamentals = generate_synthetic_fundamentals(
        symbols,
        start_date=start_date,
        end_date=end_date,
    )
    root = Path(config.paths.silver_data) / "fundamentals"
    write_parquet(fundamentals, root / "fundamentals_quarterly.parquet")
    # Quickstart always uses the complete wide fallback instead of a stale partial SEC cache.
    write_parquet(pd.DataFrame(), root / "fundamental_observations.parquet")
    return fundamentals


def run_quickstart(
    config: AppConfig,
    start_date: str,
    end_date: str,
    target_symbols: int | None = None,
) -> QuickstartResult:
    if config.runtime.research_mode:
        raise ValueError("Quickstart requires a non-research configuration.")

    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if end <= start:
        raise ValueError("Quickstart end date must be after start date.")

    price_start = (start - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
    master = _load_quickstart_master(config, target_symbols=target_symbols)
    membership = _seed_membership(config, price_start, end_date, master)
    symbols = sorted(master["symbol"].dropna().astype(str).unique().tolist())
    prices = ingest_prices(
        config,
        start_date=price_start,
        end_date=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        tickers=symbols,
    ).prices
    write_parquet(
        prices,
        Path(config.paths.silver_data) / "prices" / "prices_daily.parquet",
    )
    fundamentals = _seed_fundamentals(
        config,
        symbols,
        start_date=(start - pd.DateOffset(years=5)).strftime("%Y-%m-%d"),
        end_date=end_date,
    )
    macro = generate_synthetic_macro(config)
    write_parquet(
        macro,
        Path(config.paths.silver_data) / "macro" / "macro_observations.parquet",
    )
    backtest = run_backtest(
        start_date,
        end_date,
        config,
        enforce_data_gate=False,
    )
    return QuickstartResult(
        backtest=backtest,
        symbol_count=len(symbols),
        price_rows=len(prices),
        fundamental_rows=len(fundamentals),
        membership_rows=len(membership),
        macro_rows=len(macro),
    )
