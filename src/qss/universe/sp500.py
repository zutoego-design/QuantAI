from __future__ import annotations

import hashlib
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

import exchange_calendars as xcals
import pandas as pd
import requests

from qss.data.identifiers import normalize_symbol
from qss.data.storage import write_parquet

SP500_CONSTITUENTS_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SP500_SOURCE = "sp500_wikipedia_point_in_time"
XNYS_CALENDAR = xcals.get_calendar("XNYS")


@dataclass
class Sp500History:
    security_master: pd.DataFrame
    symbol_history: pd.DataFrame
    listing_intervals: pd.DataFrame
    membership: pd.DataFrame
    changes: pd.DataFrame
    current_constituents: pd.DataFrame


class _HtmlTableByIdParser(HTMLParser):
    def __init__(self, table_id: str) -> None:
        super().__init__()
        self.table_id = table_id
        self.rows: list[list[str]] = []
        self._in_target = False
        self._table_depth = 0
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = dict(attrs)
        if tag == "table" and attr_map.get("id") == self.table_id:
            self._in_target = True
            self._table_depth = 1
            return
        if not self._in_target:
            return
        if tag == "table":
            self._table_depth += 1
        elif tag == "tr":
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = []

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if not self._in_target:
            return
        if tag in {"td", "th"} and self._cell is not None and self._row is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None
        elif tag == "table":
            self._table_depth -= 1
            if self._table_depth <= 0:
                self._in_target = False


def _fetch_wikipedia_html() -> str:
    response = requests.get(
        SP500_CONSTITUENTS_URL,
        headers={"User-Agent": "QuantAI S&P 500 research universe loader"},
        timeout=60,
    )
    response.raise_for_status()
    return response.text


def _parse_table(html: str, table_id: str) -> list[list[str]]:
    parser = _HtmlTableByIdParser(table_id)
    parser.feed(html)
    if not parser.rows:
        raise RuntimeError(f"S&P 500 Wikipedia table {table_id!r} was not found.")
    return parser.rows


def _security_id(symbol: str, cik: str | None = None) -> str:
    cik_text = "" if cik is None or pd.isna(cik) else str(cik).strip()
    if cik_text and cik_text.lower() != "nan":
        digits = "".join(ch for ch in cik_text if ch.isdigit())
        if digits:
            return f"sec_cik_{int(digits):010d}"
    digest = hashlib.sha1(symbol.encode("utf-8")).hexdigest()[:20]
    return f"sp500_{digest}"


def load_current_constituents(html: str | None = None) -> pd.DataFrame:
    html = html or _fetch_wikipedia_html()
    rows = _parse_table(html, "constituents")
    header = rows[0]
    body = [row for row in rows[1:] if len(row) == len(header)]
    frame = pd.DataFrame(body, columns=header)
    required = {"Symbol", "Security", "GICS Sector"}
    if not required.issubset(frame.columns):
        raise RuntimeError("S&P 500 constituents table has an unexpected shape.")

    result = pd.DataFrame(
        {
            "symbol": frame["Symbol"].astype(str).map(normalize_symbol),
            "name": frame["Security"].astype(str),
            "company_name": frame["Security"].astype(str),
            "exchange": "US",
            "security_type": "Common Stock",
            "sector": frame["GICS Sector"].astype(str),
            "industry": frame.get("GICS Sub-Industry", "Unknown"),
            "date_added": pd.to_datetime(frame.get("Date added"), errors="coerce"),
            "cik": frame.get("CIK", pd.Series(pd.NA, index=frame.index)).astype(str),
            "source": "sp500_wikipedia_constituents",
        }
    )
    result = result.loc[result["symbol"].ne("")]
    result["security_id"] = [
        _security_id(symbol, cik)
        for symbol, cik in zip(result["symbol"], result["cik"], strict=False)
    ]
    result["currency"] = "USD"
    result["is_active"] = True
    result["ingestion_time"] = pd.Timestamp.now(tz="UTC").tz_localize(None)
    return result.drop_duplicates("symbol", keep="last").reset_index(drop=True)


def load_constituent_changes(html: str | None = None) -> pd.DataFrame:
    html = html or _fetch_wikipedia_html()
    rows = _parse_table(html, "changes")
    changes: list[dict[str, object]] = []
    for row in rows:
        if len(row) < 6 or row[0] in {"Effective Date", "Ticker"}:
            continue
        effective_date = pd.to_datetime(row[0], errors="coerce")
        if pd.isna(effective_date):
            continue
        added_symbol = normalize_symbol(row[1])
        removed_symbol = normalize_symbol(row[3])
        changes.append(
            {
                "effective_date": pd.Timestamp(effective_date).normalize(),
                "added_symbol": added_symbol or pd.NA,
                "added_name": row[2],
                "removed_symbol": removed_symbol or pd.NA,
                "removed_name": row[4],
                "reason": row[5],
                "source": "sp500_wikipedia_changes",
            }
        )
    frame = pd.DataFrame(changes)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "effective_date",
                "added_symbol",
                "added_name",
                "removed_symbol",
                "removed_name",
                "reason",
                "source",
            ]
        )
    return frame.sort_values("effective_date").reset_index(drop=True)


def _metadata_from_changes(changes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in changes.itertuples(index=False):
        if isinstance(row.added_symbol, str) and row.added_symbol:
            rows.append(
                {
                    "symbol": row.added_symbol,
                    "name": row.added_name,
                    "company_name": row.added_name,
                    "exchange": "US",
                    "security_type": "Common Stock",
                    "sector": "Unknown",
                    "industry": "Unknown",
                    "cik": pd.NA,
                    "source": "sp500_wikipedia_changes",
                }
            )
        if isinstance(row.removed_symbol, str) and row.removed_symbol:
            rows.append(
                {
                    "symbol": row.removed_symbol,
                    "name": row.removed_name,
                    "company_name": row.removed_name,
                    "exchange": "US",
                    "security_type": "Common Stock",
                    "sector": "Unknown",
                    "industry": "Unknown",
                    "cik": pd.NA,
                    "source": "sp500_wikipedia_changes",
                }
            )
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["symbol"] = frame["symbol"].astype(str).map(normalize_symbol)
    frame["security_id"] = [_security_id(symbol) for symbol in frame["symbol"]]
    frame["currency"] = "USD"
    frame["is_active"] = False
    frame["ingestion_time"] = pd.Timestamp.now(tz="UTC").tz_localize(None)
    return frame.drop_duplicates("symbol", keep="last")


def _snapshot_symbols(
    current_symbols: set[str],
    changes: pd.DataFrame,
    as_of: pd.Timestamp,
) -> set[str]:
    symbols = set(current_symbols)
    future_changes = changes.loc[changes["effective_date"] > as_of].sort_values(
        "effective_date",
        ascending=False,
    )
    for row in future_changes.itertuples(index=False):
        if isinstance(row.added_symbol, str) and row.added_symbol:
            symbols.discard(row.added_symbol)
        if isinstance(row.removed_symbol, str) and row.removed_symbol:
            symbols.add(row.removed_symbol)
    return symbols


def _monthly_trading_ends(
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[pd.Timestamp]:
    calendar_month_ends = pd.date_range(start, end, freq="ME")
    return [
        pd.Timestamp(
            XNYS_CALENDAR.date_to_session(month_end, direction="previous")
        ).normalize()
        for month_end in calendar_month_ends
    ]


def build_sp500_history(
    start_date: str,
    end_date: str,
    raw_root: Path | None = None,
) -> Sp500History:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if end < start:
        raise ValueError("S&P 500 universe end date must be on or after start date.")

    html = _fetch_wikipedia_html()
    current = load_current_constituents(html)
    changes = load_constituent_changes(html)
    if raw_root is not None:
        write_parquet(current, raw_root / "current_constituents.parquet")
        write_parquet(changes, raw_root / "constituent_changes.parquet")

    month_ends = _monthly_trading_ends(start, end)
    if not month_ends:
        month_ends = [end]
    current_symbols = set(current["symbol"])
    snapshots: list[pd.DataFrame] = []
    for date in month_ends:
        symbols = sorted(_snapshot_symbols(current_symbols, changes, date))
        snapshot = pd.DataFrame({"date": date, "symbol": symbols})
        snapshot["source"] = SP500_SOURCE
        snapshots.append(snapshot)
    membership = pd.concat(snapshots, ignore_index=True)

    change_meta = _metadata_from_changes(changes)
    master = pd.concat([change_meta, current], ignore_index=True)
    master["symbol"] = master["symbol"].astype(str).map(normalize_symbol)
    master = master.loc[master["symbol"].isin(set(membership["symbol"]))]
    master = master.sort_values("source").drop_duplicates("symbol", keep="last")
    if "security_id" not in master:
        master["security_id"] = [_security_id(symbol) for symbol in master["symbol"]]
    else:
        master["security_id"] = [
            security_id if isinstance(security_id, str) and security_id else _security_id(symbol)
            for symbol, security_id in zip(master["symbol"], master["security_id"], strict=False)
        ]
    for column, default in [
        ("name", ""),
        ("company_name", ""),
        ("exchange", "US"),
        ("security_type", "Common Stock"),
        ("sector", "Unknown"),
        ("industry", "Unknown"),
        ("currency", "USD"),
        ("source", "sp500_wikipedia"),
    ]:
        if column not in master:
            master[column] = default
        master[column] = master[column].fillna(default)
    master = master.reset_index(drop=True)

    membership = membership.merge(
        master[["symbol", "security_id", "security_type"]],
        on="symbol",
        how="left",
    )
    membership["included"] = True
    membership["exclusion_reason"] = ""
    membership = membership[
        [
            "date",
            "security_id",
            "symbol",
            "security_type",
            "source",
            "included",
            "exclusion_reason",
        ]
    ]

    history = (
        membership.groupby(["security_id", "symbol"], as_index=False)
        .agg(valid_from=("date", "min"), valid_to=("date", "max"), source=("source", "last"))
    )
    final_symbols = set(membership.loc[membership["date"] == max(month_ends), "symbol"])
    history.loc[history["symbol"].isin(final_symbols), "valid_to"] = pd.NaT
    intervals = history[["security_id", "symbol", "valid_from", "valid_to"]].copy()
    intervals["delisting_date"] = intervals["valid_to"]
    intervals["exchange"] = "US"
    intervals["interval_quality"] = "sp500_change_log"
    return Sp500History(
        security_master=master,
        symbol_history=history,
        listing_intervals=intervals,
        membership=membership,
        changes=changes,
        current_constituents=current,
    )
