#!/usr/bin/env python3
"""Alpaca Paper Trading — 依實際現金調整後下限價單"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import time
import logging
from datetime import datetime
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

client = TradingClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"), paper=True)

# ── 原始買入清單（因子加權比例） ──
# (ticker, 權重%, 限價)
TARGET_WEIGHTS = [
    ("AMD",  18.6, 490.33),
    ("AVGO", 12.8, 396.60),
    ("CAT",  12.7, 915.64),
    ("INTC", 12.2, 110.27),
    ("LLY",   9.5, 1149.15),
    ("ABBV",  7.3, 223.07),
    ("CSCO",  6.4, 124.15),
    ("GE",    5.1, 322.04),
    ("ORCL",  3.9, 211.82),
    ("CVX",   3.7, 189.24),
    ("XOM",   2.3, 151.75),
    ("IBM",   1.9, 280.82),
    ("ADBE",  1.8, 244.99),
    ("NFLX",  1.6, 82.64),
    ("TMO",   0.5, 469.63),
]


def calc_orders(available_cash: float):
    """依可用現金計算實際可買股數"""
    total_weight = sum(w for _, w, _ in TARGET_WEIGHTS)
    orders = []
    total_used = 0
    for ticker, wgt, price in TARGET_WEIGHTS:
        budget = available_cash * (wgt / total_weight)
        qty = max(int(budget / price), 1)  # 至少 1 股
        cost = round(qty * price, 2)
        # 確保不把錢全部用完
        if total_used + cost > available_cash * 0.98:
            qty = int((available_cash * 0.98 - total_used) / price)
            if qty <= 0:
                continue
            cost = round(qty * price, 2)
        orders.append((ticker, qty, round(price, 2), cost))
        total_used += cost
    return orders


def place_limit_order(ticker: str, qty: int, limit_price: float):
    """下單限價單"""
    try:
        order = LimitOrderRequest(
            symbol=ticker,
            qty=qty,
            side=OrderSide.BUY,
            type="limit",
            limit_price=limit_price,
            time_in_force=TimeInForce.DAY,
        )
        resp = client.submit_order(order)
        logger.info(f"  ✅ {ticker}: 買 {qty} 股 @ ${limit_price:.2f} = ${qty*limit_price:.0f} | ID: {resp.id}")
        return True
    except Exception as e:
        logger.error(f"  ❌ {ticker}: 下單失敗 — {e}")
        return False


def main():
    print()
    print("=" * 55)
    print("  Alpaca Paper Trading — 調整後下單")
    print(f"  時間: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 55)

    # 1. 查帳戶
    account = client.get_account()
    cash = float(account.cash)
    equity = float(account.equity)
    logger.info(f"帳戶現金: ${cash:,.2f}")
    logger.info(f"總權益:   ${equity:,.2f}")

    # 2. 依現金調整買入數量
    use_cash = min(cash, equity * 0.95)  # 最多用 95% 的權益
    orders = calc_orders(use_cash)
    total_cost = sum(c for _, _, _, c in orders)

    print()
    print(f"依可用資金 ${use_cash:,.0f} 調整後：")
    print(f"{'股票':>6} {'股數':>6} {'限價':>8} {'預算':>10} {'佔比':>8}")
    print("-" * 42)
    for ticker, qty, price, cost in orders:
        print(f"{ticker:>6} {qty:>6} ${price:>5.2f} {cost:>10,.0f} {cost/use_cash*100:>7.1f}%")
    print("-" * 42)
    print(f"{'總計':>20} {total_cost:>10,.0f} {total_cost/use_cash*100:>7.1f}%")
    print(f"{'餘額':>20} {use_cash-total_cost:>10,.0f} {(use_cash-total_cost)/use_cash*100:>7.1f}%")

    # 3. 下單
    print()
    confirm = input("是否下單？(y/n): ").strip().lower()
    if confirm != "y":
        logger.info("取消下單")
        return

    logger.info("開始下單...")
    success = 0
    failed = 0
    for ticker, qty, price, _ in orders:
        if place_limit_order(ticker, qty, price):
            success += 1
        else:
            failed += 1
        time.sleep(0.5)

    print()
    logger.info(f"下單完成：✅ {success} 成功 | ❌ {failed} 失敗")
    logger.info("限價單當日有效，未成交盤後自動取消")


if __name__ == "__main__":
    main()
