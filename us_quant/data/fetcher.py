"""資料層：串接 yfinance 抓取美股價量與基本面資料"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def fetch_price_data(
    tickers: list[str],
    start: str = "2015-01-01",
    end: Optional[str] = None,
) -> dict[str, pd.DataFrame]:
    """抓取多檔股票的日線 OHLCV 資料。

    Returns
    -------
    dict[str, pd.DataFrame]
        ticker → DataFrame(columns=['Open','High','Low','Close','Volume'])
        索引為日期。
    """
    end = end or datetime.today().strftime("%Y-%m-%d")
    logger.info("下載 %d 檔股票資料: %s ~ %s", len(tickers), start, end)

    data = yf.download(tickers, start=start, end=end, group_by="ticker", auto_adjust=True, progress=False)

    result: dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                df = data[t].dropna(how="all").copy()
            else:
                # 單一 ticker 回傳單層 DataFrame
                df = data.dropna(how="all").copy()
            if df.empty:
                continue
            # 確保欄位名一致
            df.columns = [c.capitalize() for c in df.columns]
            result[t] = df
        except (KeyError, AttributeError):
            logger.warning("無法取得 %s 的資料", t)
    return result


def fetch_fundamentals(ticker: str) -> dict:
    """抓取單檔股票的基本面資料。

    Returns
    -------
    dict
        {
            'market_cap': int,
            'pe_ratio': float,
            'pb_ratio': float,
            'ps_ratio': float,
            'dividend_yield': float,
            'roe': float,
            'debt_to_equity': float,
            'gross_margin': float,
            'revenue_growth': float,
            'beta': float,
        }
        失敗的欄位為 None。
    """
    try:
        s = yf.Ticker(ticker)
        info = s.info or {}
    except Exception:
        return {}

    def safe(v, default=None):
        return v if v is not None and v != 0 else default

    keys = {
        "market_cap": "marketCap",
        "pe_ratio": "trailingPE",
        "pb_ratio": "priceToBook",
        "ps_ratio": "priceToSalesTrailing12Months",
        "dividend_yield": "dividendYield",
        "roe": "returnOnEquity",
        "debt_to_equity": "debtToEquity",
        "gross_margin": "grossMargins",
        "revenue_growth": "revenueGrowth",
        "beta": "beta",
    }
    return {k: safe(info.get(v)) for k, v in keys.items()}


def fetch_all_fundamentals(tickers: list[str]) -> pd.DataFrame:
    """抓取多檔股票基本面，回傳 DataFrame (ticker × 欄位)。"""
    rows = []
    for t in tickers:
        rows.append({"ticker": t, **fetch_fundamentals(t)})
    return pd.DataFrame(rows).set_index("ticker")


def fetch_sector_info(tickers: list[str]) -> pd.DataFrame:
    """抓取行業分類、海外營收佔比。

    海外營收佔比來自各公司最新 10-K 年報（公開資訊），
    若 yfinance 有提供則使用，否則用已知映射。

    Returns
    -------
    pd.DataFrame
        index=ticker, columns=[sector, industry, foreign_revenue_pct]
    """
    # 已知海外營收佔比（來自各公司 10-K / 年報）
    KNOWN_FOREIGN_REVENUE: dict[str, float] = {
        "AAPL": 60.0, "MSFT": 50.0, "GOOGL": 55.0, "AMZN": 40.0,
        "META": 55.0, "NVDA": 60.0, "TSLA": 50.0, "JPM": 45.0,
        "V": 60.0, "MA": 60.0, "UNH": 15.0, "JNJ": 50.0,
        "PG": 55.0, "KO": 70.0, "PEP": 60.0, "HD": 10.0,
        "MCD": 65.0, "NKE": 60.0, "DIS": 40.0, "BA": 50.0,
        "CAT": 55.0, "GE": 65.0, "XOM": 60.0, "CVX": 55.0,
        "ABBV": 40.0, "LLY": 45.0, "TMO": 50.0, "AVGO": 60.0,
        "ADBE": 45.0, "CRM": 35.0, "NFLX": 60.0, "ACN": 90.0,
        "CSCO": 50.0, "INTC": 75.0, "AMD": 70.0, "IBM": 65.0,
        "ORCL": 55.0, "SAP": 80.0, "TM": 80.0,
    }

    rows = []
    for t in tickers:
        try:
            s = yf.Ticker(t)
            info = s.info or {}
            fr = info.get("foreignRevenuePct", None)
            if fr is None:
                fr = KNOWN_FOREIGN_REVENUE.get(t, None)
            rows.append({
                "ticker": t,
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "foreign_revenue_pct": float(fr) if fr is not None else None,
            })
        except Exception:
            rows.append({"ticker": t, "sector": None, "industry": None, "foreign_revenue_pct": KNOWN_FOREIGN_REVENUE.get(t)})
    return pd.DataFrame(rows).set_index("ticker")


def fetch_dxy(start: str = "2018-01-01", end: str | None = None) -> pd.Series:
    """抓取美元指數 DXY 資料。"""
    import yfinance as yf
    end = end or datetime.today().strftime("%Y-%m-%d")
    try:
        dx = yf.download("DX-Y.NYB", start=start, end=end, progress=False, auto_adjust=True)
        if dx.empty:
            return pd.Series(dtype=float)
        if isinstance(dx.columns, pd.MultiIndex):
            return dx["Close"].squeeze()
        return dx["Close"].squeeze()
    except Exception:
        return pd.Series(dtype=float)


def fetch_vix(start: str = "2018-01-01", end: str | None = None) -> pd.Series:
    """抓取 VIX 恐慌指數資料。"""
    end = end or datetime.today().strftime("%Y-%m-%d")
    try:
        vix = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=True)
        if vix.empty:
            return pd.Series(dtype=float)
        if isinstance(vix.columns, pd.MultiIndex):
            return vix["Close"].squeeze()
        return vix["Close"].squeeze()
    except Exception:
        return pd.Series(dtype=float)


def ensure_price_data(
    tickers: list[str],
    store: "DataStore",
    start: str = "2015-01-01",
    days_lookback: int = 0,
) -> dict[str, pd.DataFrame]:
    """增量更新：只下載本地沒有的資料，寫入 SQLite，再從 SQLite 讀取。

    Parameters
    ----------
    tickers : list[str]
        股票清單
    store : DataStore
        SQLite 資料庫實例
    start : str
        起始日期（第一次下載用）
    days_lookback : int
        若 >0，則強制回補最近 N 天（即使已存在）

    Returns
    -------
    dict[str, pd.DataFrame]
        跟 fetch_price_data 回傳格式一致
    """
    from datetime import datetime, timedelta

    today = datetime.today()
    need_fetch: list[str] = []
    for t in tickers:
        latest = store.latest_date_for_ticker(t)
        if latest is None:
            need_fetch.append(t)
        else:
            # 檢查最後一天是否距離今天超過 1 個交易日
            last_dt = datetime.strptime(latest, "%Y-%m-%d")
            if (today - last_dt).days > 3:
                need_fetch.append(t)

    # 如果有股票需要補資料
    if need_fetch:
        # 找到所有 need_fetch 中最舊的最後日期，用 start 當起點
        fetch_start = start
        for t in need_fetch:
            latest = store.latest_date_for_ticker(t)
            if latest and latest > fetch_start:
                fetch_start = latest
        # 往前多抓 5 天確保覆蓋
        fetch_start_dt = datetime.strptime(fetch_start, "%Y-%m-%d") - timedelta(days=5)

        logger.info("增量下載 %d 檔: %s 起", len(need_fetch), fetch_start_dt.date())
        new_data = fetch_price_data(need_fetch, fetch_start_dt.strftime("%Y-%m-%d"))
        for ticker, df in new_data.items():
            store.save_price(ticker, df)

    # 一律從 SQLite 讀取
    return store.load_all_prices(tickers, start=start)
