#!/usr/bin/env python3
"""更新資料庫：抓取最新股價與基本面資料寫入 SQLite。

Usage:
    python scripts/update_data.py [--tickers AAPL,MSFT,...] [--start 2018-01-01]
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from us_quant.config import STOCK_UNIVERSE, DATA_DIR
from us_quant.data import fetch_price_data, fetch_all_fundamentals, DataStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="更新資料庫")
    parser.add_argument("--tickers", default=",".join(STOCK_UNIVERSE), help="股票列表")
    parser.add_argument("--start", default="2015-01-01", help="開始日期")
    args = parser.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",")]
    logger.info("開始更新 %d 檔股票資料...", len(tickers))
    store = DataStore(DATA_DIR / "market.db")

    # 股價
    price_data = fetch_price_data(tickers, args.start)
    for ticker, df in price_data.items():
        store.save_price(ticker, df)
        logger.info("  ✅ %s: %d 筆", ticker, len(df))

    # 基本面
    fundamentals = fetch_all_fundamentals(list(price_data.keys()))
    store.save_fundamentals(fundamentals)
    logger.info("  ✅ 基本面: %d 檔", len(fundamentals))

    logger.info("更新完成。最新交易日: %s", store.latest_price_date())


if __name__ == "__main__":
    main()
