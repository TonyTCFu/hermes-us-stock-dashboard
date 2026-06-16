#!/usr/bin/env python3
"""兩週調倉腳本：10因子分析 → 對比持倉 → 保留高分股 → 賣低分 → 買新標的"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import logging
from datetime import datetime

import pandas as pd
import numpy as np

from us_quant.data import (
    ensure_price_data, fetch_all_fundamentals, fetch_sector_info,
    fetch_dxy, fetch_vix, DataStore,
)
from us_quant.factors import get_factor
from us_quant.signals import SignalCombiner, regime_adjust_weights
from us_quant.config import DB_PATH, STOCK_UNIVERSE, FACTOR_WEIGHTS, MAX_HOLDING

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

import os
try:
    from dotenv import load_dotenv
    _env = Path(__file__).resolve().parent.parent / ".env"
    if _env.exists():
        load_dotenv(_env)
except ImportError:
    pass
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

client = TradingClient(
    os.getenv("ALPACA_API_KEY"),
    os.getenv("ALPACA_SECRET_KEY"),
    paper=True,
)

MAX_HOLD = MAX_HOLDING  # F-003 修復：讀自 config（.env 可改）
CASH_BUFFER = 0.98

FACTOR_PARAMS = {
    'momentum': {},
    'value': {'fundamentals': None},
    'quality': {'fundamentals': None},
    'low_vol': {},
    'size': {'fundamentals': None},
    'div_yield': {'fundamentals': None},
    'revenue_growth': {'fundamentals': None},
    'industry_momentum': {'sector_map': None},
    'flow': {},
    'fx_exposure': {'dxy': None, 'sector_map': None},
}


def run_factor_analysis(store):
    """完整因子分析，回傳所有股票的最新加權分數。"""
    price_data = ensure_price_data(STOCK_UNIVERSE, store, start='2018-01-01')
    price_data = {k: v for k, v in price_data.items() if not v.empty}

    fundamentals = fetch_all_fundamentals(list(price_data.keys()))
    store.save_fundamentals(fundamentals)

    sector_map = store.load_sector_mapping()
    if sector_map.empty:
        sector_map = fetch_sector_info(list(price_data.keys()))
        store.save_sector_mapping(sector_map)

    dxy = store.load_macro('DXY')
    if dxy.empty:
        dxy = fetch_dxy(start='2018-01-01')
        store.save_macro('DXY', dxy.to_frame('value').reset_index())

    vix = store.load_macro('VIX')
    if vix.empty:
        vix = fetch_vix(start='2018-01-01')
        store.save_macro('VIX', vix.to_frame('value').reset_index())

    params = FACTOR_PARAMS.copy()
    params['value']['fundamentals'] = fundamentals
    params['quality']['fundamentals'] = fundamentals
    params['size']['fundamentals'] = fundamentals
    params['div_yield']['fundamentals'] = fundamentals
    params['revenue_growth']['fundamentals'] = fundamentals
    params['industry_momentum']['sector_map'] = sector_map
    params['fx_exposure']['dxy'] = dxy
    params['fx_exposure']['sector_map'] = sector_map

    factor_results = []
    for name in FACTOR_WEIGHTS:
        if name not in params:
            continue
        r = get_factor(name).compute(price_data, **params[name])
        if not r.scores.empty:
            factor_results.append(r)

    signal = SignalCombiner(FACTOR_WEIGHTS).combine(factor_results)
    vix_current = vix.iloc[-1] if not vix.empty else 0
    signal_adj = regime_adjust_weights(signal, vix)

    # 取每檔股票最新的分數（而非精準對齊最新日期）
    latest_signal = signal_adj.groupby(level=0).last().sort_values(ascending=False)

    # 最新收盤價
    price_matrix = pd.concat({t: df['Close'] for t, df in price_data.items()}, axis=1)
    price_matrix = price_matrix.ffill()
    latest_prices = price_matrix.iloc[-1]

    # 只保留有價格的
    all_scores = latest_signal[latest_signal.index.isin(latest_prices.index)]

    # 強勢因子分析
    def top_factors(ticker):
        tf = []
        for fr in factor_results:
            w = FACTOR_WEIGHTS.get(fr.name, 0)
            if w == 0 or fr.scores.empty:
                continue
            try:
                val = fr.scores.xs(ticker, level='ticker').iloc[-1]
                tf.append((fr.name, round(val * w, 3)))
            except (KeyError, IndexError):
                continue
        return sorted(tf, key=lambda x: -x[1])[:3]

    info = []
    for t in all_scores.index:
        tf = top_factors(t)
        info.append({
            'ticker': t,
            'signal': round(all_scores[t], 4),
            'rank': len(info) + 1,
            'price': round(latest_prices[t], 2),
            'top_factors': ', '.join([f'{n}({v:.2f})' for n, v in tf[:2]]),
        })

    df = pd.DataFrame(info).set_index('ticker')
    return df, vix_current, all_scores


def get_current_positions():
    """取得 Alpaca 當前持倉。"""
    account = client.get_account()
    cash = float(account.cash)
    equity = float(account.equity)

    positions = client.get_all_positions()
    holdings = []
    for p in positions:
        holdings.append({
            'ticker': p.symbol,
            'qty': float(p.qty),
            'avg_price': float(p.avg_entry_price),
            'current_price': float(p.current_price),
            'market_value': float(p.market_value),
            'unrealized_pl': float(p.unrealized_pl),
        })

    return cash, equity, pd.DataFrame(holdings) if holdings else pd.DataFrame()


def generate_plan(all_scores_df, positions, cash, latest_prices):
    """產生調倉計畫：保留高分股、賣低分股、買新標的。

    規則：
    - 持倉中排名 Top 20 → 保留，調整至目標權重
    - 持倉中排名 > 20 → 全賣
    - 剩餘預算 → 買入 Top 15 中尚未持有的標的
    """
    target_top = all_scores_df.head(MAX_HOLD)
    target_tickers = set(target_top.index)

    current_tickers = set(positions['ticker']) if not positions.empty else set()

    # ── 保留股：排名 ≤ 15（Top 15）且已持有的才保留 ──
    keep_tickers = set()
    if not positions.empty:
        for _, p in positions.iterrows():
            t = p['ticker']
            if t in all_scores_df.index and t in target_tickers:
                keep_tickers.add(t)

    # 賣出：不在 Top 15 的持倉全部賣出
    sell_orders = []
    for _, p in positions.iterrows():
        ticker = p['ticker']
        if ticker not in keep_tickers:
            rank = all_scores_df.loc[ticker, 'rank'] if ticker in all_scores_df.index else 999
            sell_orders.append({
                'ticker': ticker,
                'qty': int(p['qty']),
                'price': round(p['current_price'], 2),
                'value': float(p['market_value']),
                'reason': f'排名 {rank}，不在 Top 15',
            })

    # ── 計算可用資金 ──
    sell_proceeds = sum(o['value'] for o in sell_orders)
    available_cash = cash + sell_proceeds
    buy_budget = available_cash * CASH_BUFFER

    # ── 買入：Top 15 中尚未持有或權重不足的 ──
    buy_orders = []
    remaining = buy_budget

    for _, row in target_top.iterrows():
        t = row.name
        if remaining <= 0:
            break

        if t in keep_tickers:
            # 已持有 → 補到目標
            pos = positions[positions['ticker'] == t].iloc[0]
            target_value = buy_budget * (1.0 / MAX_HOLD)
            current_value = float(pos['market_value'])
            top_up = target_value - current_value
            if top_up > 0:
                qty = max(int(top_up / float(pos['current_price'])), 0)
                if qty > 0:
                    cost = round(qty * float(pos['current_price']), 2)
                    if cost <= remaining:
                        buy_orders.append({
                            'ticker': t, 'qty': qty,
                            'price': round(float(pos['current_price']), 2),
                            'value': cost,
                            'reason': f'補倉（排名 {row["rank"]}）',
                        })
                        remaining -= cost
        else:
            # 未持有 → 新買入
            price = row['price']
            qty = max(int(remaining * (1.0 / MAX_HOLD) / price), 1)
            cost = round(qty * price, 2)
            if cost > remaining or cost <= 0:
                cost = remaining
                qty = max(int(remaining / price), 0)
                if qty == 0:
                    continue
                cost = round(qty * price, 2)
            buy_orders.append({
                'ticker': t, 'qty': qty,
                'price': price,
                'value': cost,
                'reason': f'新買入（排名 {row["rank"]}）',
            })
            remaining -= cost

    return sell_orders, buy_orders, sell_proceeds, keep_tickers


def execute_orders(sell_orders, buy_orders):
    """執行買賣單。先賣後買，使用市價單。"""
    success = 0
    failed = 0

    if sell_orders:
        print()
        print('📉 執行賣出（市價單）...')
        for o in sell_orders:
            try:
                order = MarketOrderRequest(
                    symbol=o['ticker'],
                    qty=o['qty'],
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                )
                resp = client.submit_order(order)
                logger.info(f"  ✅ {o['ticker']}: 賣 {o['qty']} 股（市價）")
                success += 1
            except Exception as e:
                logger.error(f"  ❌ {o['ticker']}: 賣出失敗 — {e}")
                failed += 1
            time.sleep(0.5)

    if buy_orders:
        print()
        print('📈 執行買入（市價單）...')
        for o in buy_orders:
            try:
                order = MarketOrderRequest(
                    symbol=o['ticker'],
                    qty=o['qty'],
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                )
                resp = client.submit_order(order)
                logger.info(f"  ✅ {o['ticker']}: 買 {o['qty']} 股（市價）")
                success += 1
            except Exception as e:
                logger.error(f"  ❌ {o['ticker']}: 買入失敗 — {e}")
                failed += 1
            time.sleep(0.5)

    return success, failed


def main():
    store = DataStore(DB_PATH)

    print()
    print('=' * 60)
    print(f'  🔄 兩週調倉 — {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    print('=' * 60)
    print()

    # ── 1. Alpaca 狀態 ──
    print('📊 Alpaca 帳戶...')
    cash, equity, positions = get_current_positions()
    print(f'   現金: ${cash:>10,.2f}  權益: ${equity:>10,.2f}')

    if not positions.empty:
        print(f'   持倉 ({len(positions)} 檔):')
        print(f'   {"股票":>6} {"股數":>6} {"均價":>8} {"現價":>8} {"市值":>10} {"損益":>10}')
        print('   ' + '-' * 48)
        for _, p in positions.iterrows():
            print(f'   {p["ticker"]:>6} {p["qty"]:>6.0f} ${p["avg_price"]:>7.2f} ${p["current_price"]:>7.2f} ${p["market_value"]:>8,.0f} ${p["unrealized_pl"]:>+8,.0f}')
    else:
        print('   持倉: 無')
    print()

    # ── 2. 因子分析 ──
    print('🔬 10 因子分析...')
    all_scores, vix, _ = run_factor_analysis(store)
    print(f'   VIX: {vix:.2f} ({"⚠️ 高於30" if vix > 30 else "🟡 警戒" if vix > 20 else "🟢 正常"})')
    print()

    top_n = all_scores.head(MAX_HOLDING)
    print(f'   📋 Top {MAX_HOLDING} 買入清單:')
    print(f'   {"排名":>4} {"股票":>6} {"信號":>8} {"目前持有":>10} {"現價":>8} {"強勢因子":>30}')
    print('   ' + '-' * 65)
    current_set = set(positions['ticker']) if not positions.empty else set()
    for _, row in top_n.iterrows():
        held = '✅ 持有中' if row.name in current_set else '❌ 新買入'
        print(f'   {row["rank"]:>4} {row.name:>6} {row["signal"]:>8.3f} {held:>10} ${row["price"]:>6.2f} {row["top_factors"]:>30}')
    print()

    # 顯示當前持股的排名
    if not positions.empty:
        target_set = set(top_n.index)
        print(f'   📋 當前持股排名:')
        print(f'   {"股票":>6} {"排名":>6} {"信號":>8} {"決策":>10}')
        print('   ' + '-' * 32)
        for _, p in positions.iterrows():
            t = p['ticker']
            if t in all_scores.index:
                rank = all_scores.loc[t, 'rank']
                sig = all_scores.loc[t, 'signal']
                decision = '✅ 保留' if t in target_set else '❌ 賣出'
                print(f'   {t:>6} {rank:>6} {sig:>8.3f} {decision:>10}')
            else:
                print(f'   {t:>6} {"N/A":>6} {"N/A":>8} {"❌ 賣出":>10}')
        print()

    # ── 3. 調倉計畫 ──
    sell_orders, buy_orders, sell_proceeds, keep_tickers = generate_plan(
        all_scores, positions, cash, None
    )
    total_sell = sum(o['value'] for o in sell_orders)
    total_buy = sum(o['value'] for o in buy_orders)
    cash_after = cash + total_sell - total_buy

    print('=' * 60)
    print('  📋 調倉計畫')
    print('=' * 60)
    print(f'   保留持股: {len(keep_tickers)} 檔')
    print(f'   可用資金: ${cash + total_sell:>10,.2f} (現金 ${cash:,.0f} + 賣出 ${total_sell:,.0f})')
    print(f'   買入總額: ${total_buy:>10,.2f}')
    print(f'   調倉後現金: ${cash_after:>10,.2f}')
    if total_sell > 0 or total_buy > 0:
        print(f'   預估交易成本: ${(total_sell + total_buy) * 0.001:,.2f} (0.1%)')
    print()

    if sell_orders:
        print(f'   📉 賣出 ({len(sell_orders)} 筆):')
        for o in sell_orders:
            print(f'     {o["ticker"]:>6}: {o["qty"]:>4}股 = ${o["value"]:>7,.0f}  ({o["reason"]})')
        print()

    if buy_orders:
        print(f'   📈 買入 ({len(buy_orders)} 筆):')
        for o in buy_orders:
            print(f'     {o["ticker"]:>6}: {o["qty"]:>4}股 × ${o["price"]:>6.2f} = ${o["value"]:>7,.0f}  ({o["reason"]})')
        print()
    else:
        print('   無需調整')
        print()

    # ── 4. 執行 ──
    if not sell_orders and not buy_orders:
        print('✅ 持倉已最優，無需調倉')
        return

    print('=' * 60)
    confirm = input('是否執行調倉？(y/n): ').strip().lower()
    if confirm != 'y':
        print('❌ 取消調倉')
        return

    success, failed = execute_orders(sell_orders, buy_orders)

    print()
    print('=' * 60)
    print(f'  🔄 調倉完成 ✅ {success} 成功 | ❌ {failed} 失敗')
    print(f'  市價單當日有效，未成交盤後自動取消')
    print(f'  下次調倉: 約 2 週後')
    print('=' * 60)


if __name__ == '__main__':
    main()
