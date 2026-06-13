"""
data_provider.py — Multi-source price data with SQLite incremental cache.

Fetch order:
  1. SQLite cache (instant, no network)
  2. yfinance (primary network source)
  3. Tiingo REST API (fallback — requires TIINGO_API_KEY env var)
  4. Alpaca Markets API (fallback — requires ALPACA_API_KEY and ALPACA_SECRET_KEY env vars)
  5. Alpha Vantage API (fallback — requires ALPHAVANTAGE_API_KEY env var)

The cache stores adjusted OHLCV per ticker.  On each call it checks the last
cached date and downloads only the missing tail (incremental patching), so
full history is never re-downloaded after the first run.
"""
from __future__ import annotations

import contextlib
import os
import sqlite3
import time
from pathlib import Path
from typing import Generator
from urllib.request import Request, urlopen
import json
import logging

import pandas as pd

logger = logging.getLogger(__name__)

LONGTERM_START = "1986-01-01"
_DB_PATH = Path(__file__).resolve().parent / ".price_cache.db"

# ---------------------------------------------------------------------------
# SQLite cache
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _db_conn(db_path: Path = _DB_PATH) -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(str(db_path), timeout=10)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _init_cache_db(db_path: Path = _DB_PATH) -> None:
    with _db_conn(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv (
                ticker TEXT NOT NULL,
                date   TEXT NOT NULL,
                open   REAL,
                high   REAL,
                low    REAL,
                close  REAL,
                volume REAL,
                PRIMARY KEY (ticker, date)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ticker_date ON ohlcv (ticker, date)")


def _cache_get_last_date(ticker: str, db_path: Path = _DB_PATH) -> str | None:
    with _db_conn(db_path) as conn:
        row = conn.execute(
            "SELECT MAX(date) FROM ohlcv WHERE ticker = ?", (ticker,)
        ).fetchone()
    return row[0] if row and row[0] else None


def _cache_load(ticker: str, start: str, db_path: Path = _DB_PATH) -> pd.DataFrame:
    with _db_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT date, open, high, low, close, volume FROM ohlcv "
            "WHERE ticker = ? AND date >= ? ORDER BY date",
            (ticker, start),
        ).fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    return df.apply(pd.to_numeric, errors="coerce")


def _cache_save(ticker: str, df: pd.DataFrame, db_path: Path = _DB_PATH) -> None:
    if df.empty:
        return
    rows = []
    for idx, row in df.iterrows():
        date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)[:10]
        rows.append((
            ticker, date_str,
            _safe_float(row.get("Open")),
            _safe_float(row.get("High")),
            _safe_float(row.get("Low")),
            _safe_float(row.get("Close")),
            _safe_float(row.get("Volume")),
        ))
    with _db_conn(db_path) as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO ohlcv (ticker, date, open, high, low, close, volume) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )


def _safe_float(v: object) -> float | None:
    try:
        f = float(v)  # type: ignore[arg-type]
        import math
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# yfinance fetch
# ---------------------------------------------------------------------------

def _fetch_yfinance(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    import yfinance as yf

    delays = [0, 5, 15, 30, 60]
    last_exc: Exception | None = None

    for attempt, delay in enumerate(delays):
        if delay:
            time.sleep(delay)
        try:
            kwargs: dict = {"start": start, "interval": "1d", "auto_adjust": True}
            if end:
                kwargs["end"] = end
            data = yf.Ticker(ticker).history(**kwargs)
            if not data.empty:
                return _normalize(data)

            # fallback to yf.download on 3rd+ attempt
            if attempt >= 2:
                dl_kw = {"start": start, "interval": "1d", "auto_adjust": True, "progress": False}
                if end:
                    dl_kw["end"] = end
                data2 = yf.download(ticker, **dl_kw)
                if not data2.empty:
                    return _normalize(data2)
        except Exception as exc:
            last_exc = exc
            logger.debug("yfinance attempt %d failed for %s: %s", attempt, ticker, exc)

    if last_exc:
        raise RuntimeError(f"yfinance failed for {ticker}: {last_exc}") from last_exc
    raise RuntimeError(f"yfinance returned empty data for {ticker}")


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns from yf.download and keep OHLCV."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    rename = {c: c.title() for c in df.columns}
    df = df.rename(columns=rename)
    needed = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    return df[needed].copy()


# ---------------------------------------------------------------------------
# Tiingo fallback
# ---------------------------------------------------------------------------

def _fetch_tiingo(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    """
    Tiingo REST fallback.  Requires TIINGO_API_KEY environment variable.
    Returns adjusted OHLCV DataFrame; empty if key missing or request fails.
    """
    api_key = os.getenv("TIINGO_API_KEY", "")
    if not api_key:
        return pd.DataFrame()

    end_str = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    url = (
        f"https://api.tiingo.com/tiingo/daily/{ticker.lower()}/prices"
        f"?startDate={start}&endDate={end_str}&resampleFreq=daily&token={api_key}"
    )
    try:
        req = Request(url, headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df = df.set_index("date").sort_index()
        col_map = {
            "adjOpen": "Open", "adjHigh": "High", "adjLow": "Low",
            "adjClose": "Close", "adjVolume": "Volume",
        }
        # Fall back to unadjusted if adjusted columns not present
        fallback_map = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
        for src, dst in {**fallback_map, **col_map}.items():
            if src in df.columns and dst not in df.columns:
                df[dst] = df[src]
        available = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        return df[available].apply(pd.to_numeric, errors="coerce")
    except Exception as exc:
        logger.debug("Tiingo fetch failed for %s: %s", ticker, exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Alpaca fallback
# ---------------------------------------------------------------------------

def _fetch_alpaca(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    """
    Alpaca Markets v2 historical bars fallback.
    Requires ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables.
    Returns adjusted OHLCV DataFrame; empty if keys missing or request fails.
    """
    api_key = os.getenv("ALPACA_API_KEY", "")
    secret_key = os.getenv("ALPACA_SECRET_KEY", "")
    if not api_key or not secret_key:
        return pd.DataFrame()

    end_str = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    bars: list = []
    next_page_token: str | None = None

    try:
        while True:
            url = (
                f"https://data.alpaca.markets/v2/stocks/{ticker.upper()}/bars"
                f"?start={start}T00:00:00Z&end={end_str}T00:00:00Z"
                f"&timeframe=1Day&feed=sip&adjustment=all&limit=10000"
            )
            if next_page_token:
                from urllib.parse import quote
                url += f"&page_token={quote(next_page_token)}"

            req = Request(
                url,
                headers={
                    "APCA-API-KEY-ID": api_key,
                    "APCA-API-SECRET-KEY": secret_key,
                },
            )
            with urlopen(req, timeout=30) as resp:
                payload = json.loads(resp.read())

            page_bars = payload.get("bars")
            if not page_bars:
                break
            bars.extend(page_bars)

            next_page_token = payload.get("next_page_token")
            if not next_page_token:
                break

        if not bars:
            return pd.DataFrame()

        df = pd.DataFrame(bars)
        df = df.rename(columns={"t": "Date", "o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
        df["Date"] = pd.to_datetime(df["Date"])
        # Convert tz-aware timestamps to tz-naive
        if df["Date"].dt.tz is not None:
            df["Date"] = df["Date"].dt.tz_convert("UTC").dt.tz_localize(None)
        df = df.set_index("Date").sort_index()
        available = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        return df[available].apply(pd.to_numeric, errors="coerce")
    except Exception as exc:
        logger.debug("Alpaca fetch failed for %s: %s", ticker, exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Alpha Vantage fallback
# ---------------------------------------------------------------------------

def _fetch_alphavantage(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    """
    Alpha Vantage TIME_SERIES_DAILY_ADJUSTED fallback.
    Requires ALPHAVANTAGE_API_KEY environment variable.
    Returns adjusted OHLCV DataFrame; empty if key missing, rate-limited, or request fails.
    """
    api_key = os.getenv("ALPHAVANTAGE_API_KEY", "")
    if not api_key:
        return pd.DataFrame()

    end_str = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    url = (
        f"https://www.alphavantage.co/query"
        f"?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker.upper()}"
        f"&outputsize=full&apikey={api_key}"
    )
    try:
        req = Request(url, headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read())

        # Detect rate-limit / informational responses
        if "Note" in payload or "Information" in payload:
            logger.debug("Alpha Vantage rate-limited for %s", ticker)
            return pd.DataFrame()

        ts = payload.get("Time Series (Daily)")
        if not ts:
            return pd.DataFrame()

        rows = []
        for date_str, entry in ts.items():
            if date_str < start or date_str > end_str:
                continue
            rows.append({
                "Date": date_str,
                "Open": entry.get("1. open"),
                "High": entry.get("2. high"),
                "Low": entry.get("3. low"),
                "Close": entry.get("5. adjusted close"),
                "Volume": entry.get("6. volume"),
            })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        return df[["Open", "High", "Low", "Close", "Volume"]].apply(pd.to_numeric, errors="coerce")
    except Exception as exc:
        logger.debug("Alpha Vantage fetch failed for %s: %s", ticker, exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def fetch_price_data(
    ticker: str,
    period: str = "max",
    start: str = LONGTERM_START,
    db_path: Path = _DB_PATH,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch adjusted OHLCV for *ticker*.

    Strategy:
    1. Consult SQLite cache; if fresh enough, return immediately.
    2. Download only the missing tail from the first available source
       (incremental patch), trying in order:
         yfinance → Tiingo → Alpaca → Alpha Vantage
    3. Save new rows to cache.
    """
    _init_cache_db(db_path)

    today = pd.Timestamp.today().normalize()
    effective_start = start if period == "max" else _period_to_start(period)

    # --- Check cache ---
    last_cached = _cache_get_last_date(ticker, db_path)

    if not force_refresh and last_cached:
        last_dt = pd.Timestamp(last_cached)
        # Cache is fresh if last date is today or the most recent trading day
        if (today - last_dt).days <= 3:
            cached = _cache_load(ticker, effective_start, db_path)
            if not cached.empty:
                return cached

    # --- Incremental download ---
    dl_start = effective_start
    if last_cached and not force_refresh:
        # Only download from day after last cached
        dl_start = (pd.Timestamp(last_cached) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    new_data = pd.DataFrame()
    _sources = [
        ("yfinance",       lambda: _fetch_yfinance(ticker, start=dl_start)),
        ("Tiingo",         lambda: _fetch_tiingo(ticker, start=dl_start)),
        ("Alpaca",         lambda: _fetch_alpaca(ticker, start=dl_start)),
        ("Alpha Vantage",  lambda: _fetch_alphavantage(ticker, start=dl_start)),
    ]
    for source_name, fetch_fn in _sources:
        logger.info("Trying %s for %s ...", source_name, ticker)
        try:
            result = fetch_fn()
        except Exception as exc:
            logger.warning("%s failed for %s: %s", source_name, ticker, exc)
            continue
        if not result.empty:
            logger.info("%s succeeded for %s", source_name, ticker)
            new_data = result
            break
        logger.debug("%s returned empty data for %s", source_name, ticker)

    if not new_data.empty:
        _cache_save(ticker, new_data, db_path)

    # --- Return from cache ---
    cached = _cache_load(ticker, effective_start, db_path)
    if not cached.empty:
        return cached

    # --- No cache, return whatever we fetched ---
    if not new_data.empty:
        return new_data

    raise RuntimeError(
        f"{ticker} 데이터를 가져오지 못했습니다. "
        "야후파이낸스 요청이 제한되었을 수 있습니다. 1~2분 후 다시 시도해주세요."
    )


def _period_to_start(period: str) -> str:
    today = pd.Timestamp.today()
    mapping = {"5y": today - pd.DateOffset(years=5), "10y": today - pd.DateOffset(years=10)}
    dt = mapping.get(period, pd.Timestamp(LONGTERM_START))
    return dt.strftime("%Y-%m-%d")


def validate_adjusted_close(df: pd.DataFrame) -> list[str]:
    """
    Check whether the price series looks like adjusted (total-return) data.
    Returns list of warning strings (empty = OK).
    """
    warnings: list[str] = []
    if df.empty or "Close" not in df.columns:
        warnings.append("Close 컬럼이 없거나 데이터가 비어있습니다.")
        return warnings

    # Heuristic: adjusted series should not have large single-day jumps > 30%
    # that are exactly on ex-dividend dates (unadjusted data shows these as drops)
    daily_ret = df["Close"].pct_change().dropna()
    extreme = (daily_ret.abs() > 0.30).sum()
    if extreme > 0:
        warnings.append(
            f"종가에서 ±30% 초과 일간 변동이 {extreme}건 발견되었습니다. "
            "분할/배당 미조정 데이터일 수 있습니다. yfinance auto_adjust=True를 확인하세요."
        )

    # Check for suspiciously flat periods (possible stale/missing data)
    zero_change = (daily_ret == 0).sum()
    total = len(daily_ret)
    if total > 0 and zero_change / total > 0.05:
        warnings.append(
            f"일간 변동이 0인 날이 {zero_change}일({zero_change/total:.0%}) — "
            "데이터 누락 또는 거래 정지 구간이 포함되어 있을 수 있습니다."
        )

    return warnings
