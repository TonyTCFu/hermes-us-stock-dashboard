"""資料層：SQLite 本地快取，減少重複抓取"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class DataStore:
    """SQLite 快取層，儲存日線股價與基本面快照。"""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS price (
                    ticker TEXT NOT NULL,
                    date   TEXT NOT NULL,
                    open   REAL,
                    high   REAL,
                    low    REAL,
                    close  REAL,
                    volume INTEGER,
                    PRIMARY KEY (ticker, date)
                );
                CREATE TABLE IF NOT EXISTS fundamentals (
                    ticker       TEXT PRIMARY KEY,
                    fetched_at   TEXT NOT NULL DEFAULT (datetime('now')),
                    market_cap   INTEGER,
                    pe_ratio     REAL,
                    pb_ratio     REAL,
                    ps_ratio     REAL,
                    dividend_yield REAL,
                    roe           REAL,
                    debt_to_equity REAL,
                    gross_margin  REAL,
                    revenue_growth REAL,
                    beta          REAL
                );
                CREATE TABLE IF NOT EXISTS macro_data (
                    name   TEXT NOT NULL,
                    date   TEXT NOT NULL,
                    value  REAL,
                    PRIMARY KEY (name, date)
                );
                CREATE TABLE IF NOT EXISTS sector_mapping (
                    ticker TEXT PRIMARY KEY,
                    sector TEXT,
                    industry TEXT,
                    foreign_revenue_pct REAL
                );
                CREATE INDEX IF NOT EXISTS idx_price_date ON price(date);
                CREATE INDEX IF NOT EXISTS idx_macro_name ON macro_data(name);
            """)

    def _conn(self):
        return sqlite3.connect(str(self.db_path))

    # ── 股價 ──

    def save_price(self, ticker: str, df: pd.DataFrame):
        """寫入日線資料（UPSERT）。"""
        if df.empty:
            return
        df = df.reset_index()
        # 統一欄位名
        cols = {c.lower(): c for c in df.columns}
        df = df.rename(columns={
            cols.get("date", "date"): "date",
            cols.get("open", "open"): "open",
            cols.get("high", "high"): "high",
            cols.get("low", "low"): "low",
            cols.get("close", "close"): "close",
            cols.get("volume", "volume"): "volume",
        })
        df["ticker"] = ticker
        # Timestamp → 字串，否則 sqlite3 無法綁定
        df["date"] = df["date"].astype(str)
        with self._conn() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO price (ticker, date, open, high, low, close, volume)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                df[["ticker", "date", "open", "high", "low", "close", "volume"]].itertuples(index=False, name=None),
            )

    def load_price(self, ticker: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        """讀取日線資料，回傳 DataFrame(date 為 index)。"""
        parts = ["SELECT date, open, high, low, close, volume FROM price WHERE ticker = ?"]
        params: list = [ticker]
        if start:
            parts.append("AND date >= ?")
            params.append(start)
        if end:
            parts.append("AND date <= ?")
            params.append(end)
        with self._conn() as conn:
            df = pd.read_sql(" ".join(parts), conn, params=params, parse_dates=["date"])
        if df.empty:
            return df
        return df.set_index("date").sort_index()

    def has_price(self, ticker: str, date: str) -> bool:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM price WHERE ticker = ? AND date = ? LIMIT 1", (ticker, date)
            ).fetchone()
            return row is not None

    def latest_price_date(self) -> str | None:
        """回傳資料庫中最新的交易日。"""
        with self._conn() as conn:
            row = conn.execute("SELECT MAX(date) FROM price").fetchone()
            return row[0] if row else None

    def latest_date_for_ticker(self, ticker: str) -> str | None:
        """回傳某檔股票最新的交易日。"""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT MAX(date) FROM price WHERE ticker = ?", (ticker,)
            ).fetchone()
            return row[0] if row else None

    # ── 宏觀資料（DXY, VIX） ──

    def save_macro(self, name: str, df: pd.DataFrame):
        """寫入宏觀指數資料。df 須含 date + value 欄位。"""
        if df.empty:
            return
        d = df.copy()
        d["name"] = name
        # 兼容不同日期欄位名稱
        date_col = next((c for c in d.columns if c.lower() in ("date", "index", "datetime")), None)
        if date_col and date_col != "date":
            d["date"] = d[date_col]
            d = d.drop(columns=[date_col])
        elif date_col is None:
            # 使用 index
            d["date"] = d.index
        d["date"] = d["date"].astype(str)
        with self._conn() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO macro_data (name, date, value) VALUES (?, ?, ?)",
                d[["name", "date", "value"]].itertuples(index=False, name=None),
            )

    def load_macro(self, name: str, start: str | None = None) -> pd.Series:
        """讀取宏觀指數時間序列。"""
        parts = ["SELECT date, value FROM macro_data WHERE name = ?"]
        params: list = [name]
        if start:
            parts.append("AND date >= ?")
            params.append(start)
        parts.append("ORDER BY date")
        with self._conn() as conn:
            df = pd.read_sql(" ".join(parts), conn, params=params, parse_dates=["date"])
        if df.empty:
            return pd.Series(dtype=float)
        return df.set_index("date")["value"].sort_index()

    # ── 行業對應 ──

    def save_sector_mapping(self, df: pd.DataFrame):
        """寫入行業對應 + 海外營收佔比。index=ticker。"""
        if df.empty:
            return
        d = df.reset_index()
        with self._conn() as conn:
            for row in d.itertuples(index=False):
                conn.execute(
                    """INSERT OR REPLACE INTO sector_mapping
                       (ticker, sector, industry, foreign_revenue_pct)
                       VALUES (?, ?, ?, ?)""",
                    (row.ticker,
                     getattr(row, "sector", None),
                     getattr(row, "industry", None),
                     getattr(row, "foreign_revenue_pct", None)),
                )

    def load_sector_mapping(self) -> pd.DataFrame:
        """讀取所有行業對應。"""
        with self._conn() as conn:
            return pd.read_sql("SELECT * FROM sector_mapping", conn, index_col="ticker")

    # ── 全域 ──

    def load_all_prices(self, tickers: list[str] | None = None,
                        start: str | None = None,
                        end: str | None = None) -> dict[str, pd.DataFrame]:
        """一次讀取多檔股票的全部日線資料。

        Returns
        -------
        dict[str, pd.DataFrame]
            ticker → DataFrame(date 為 index)，與 fetch_price_data 格式一致
        """
        parts = ["SELECT ticker, date, open, high, low, close, volume FROM price"]
        params: list = []
        clauses = []
        if tickers:
            placeholders = ",".join("?" for _ in tickers)
            clauses.append(f"ticker IN ({placeholders})")
            params.extend(tickers)
        if start:
            clauses.append("date >= ?")
            params.append(start)
        if end:
            clauses.append("date <= ?")
            params.append(end)
        if clauses:
            parts.append("WHERE " + " AND ".join(clauses))
        parts.append("ORDER BY ticker, date")

        with self._conn() as conn:
            df = pd.read_sql(" ".join(parts), conn, params=params,
                             parse_dates=["date"])

        result: dict[str, pd.DataFrame] = {}
        for ticker, grp in df.groupby("ticker"):
            grp = grp.set_index("date").sort_index()
            grp = grp[["open", "high", "low", "close", "volume"]]
            grp.columns = [c.capitalize() for c in grp.columns]
            result[ticker] = grp
        return result

    # ── 基本面 ──

    def save_fundamentals(self, df: pd.DataFrame):
        """寫入基本面快照（UPSERT）。"""
        if df.empty:
            return
        df = df.reset_index()  # ticker 從 index 變欄位
        with self._conn() as conn:
            for row in df.itertuples(index=False):
                conn.execute(
                    """INSERT OR REPLACE INTO fundamentals
                       (ticker, market_cap, pe_ratio, pb_ratio, ps_ratio,
                        dividend_yield, roe, debt_to_equity, gross_margin,
                        revenue_growth, beta)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (row.ticker,
                     getattr(row, "market_cap", None),
                     getattr(row, "pe_ratio", None),
                     getattr(row, "pb_ratio", None),
                     getattr(row, "ps_ratio", None),
                     getattr(row, "dividend_yield", None),
                     getattr(row, "roe", None),
                     getattr(row, "debt_to_equity", None),
                     getattr(row, "gross_margin", None),
                     getattr(row, "revenue_growth", None),
                     getattr(row, "beta", None)),
                )

    def load_fundamentals(self) -> pd.DataFrame:
        """讀取所有基本面資料。"""
        with self._conn() as conn:
            return pd.read_sql("SELECT * FROM fundamentals", conn, index_col="ticker")

    # ── 全域 ──

    def available_tickers(self) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute("SELECT DISTINCT ticker FROM price ORDER BY ticker").fetchall()
            return [r[0] for r in rows]
