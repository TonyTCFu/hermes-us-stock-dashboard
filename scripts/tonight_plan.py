#!/usr/bin/env python3
"""回測 + 今晚買入清單（{MAX_HOLD}檔、10因子加權、VIX 市場狀態調整、市價單）"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from us_quant.data import (
    ensure_price_data, fetch_all_fundamentals, fetch_sector_info,
    fetch_dxy, fetch_vix, DataStore,
)
from us_quant.factors import get_factor
from us_quant.signals import SignalCombiner, regime_adjust_weights
from us_quant.backtest import BacktestEngine
from us_quant.reporting import PerformanceAnalyzer
from us_quant.config import DB_PATH, STOCK_UNIVERSE, FACTOR_WEIGHTS, REBALANCE_FREQUENCY, MAX_HOLDING
import pandas as pd
import numpy as np

store = DataStore(DB_PATH)
INITIAL = 100_000
MAX_HOLD = MAX_HOLDING  # F-003 修復：讀自 config（.env 可改）

# ── 0. 加載因子權重 ──
print('🔧 因子權重設定')
for name, w in sorted(FACTOR_WEIGHTS.items(), key=lambda x: -x[1]):
    print(f'   {name:>20}: {w*100:>5.1f}%')
print()

# ── 1. 股價資料 ──
price_data = ensure_price_data(STOCK_UNIVERSE, store, start='2018-01-01')
price_data = {k: v for k, v in price_data.items() if not v.empty}
print(f'📦 股價: {len(price_data)} 檔')

# ── 2. 基本面 ──
fundamentals = fetch_all_fundamentals(list(price_data.keys()))
store.save_fundamentals(fundamentals)
print(f'📦 基本面: {len(fundamentals)} 檔')

# ── 3. 行業分類 + 海外營收佔比 ──
sector_map = store.load_sector_mapping()
if sector_map.empty:
    print('📦 抓取行業分類中...')
    sector_map = fetch_sector_info(list(price_data.keys()))
    store.save_sector_mapping(sector_map)
print(f'📦 行業分類: {len(sector_map)} 檔')

# ── 4. 宏觀資料（DXY, VIX） ──
dxy = store.load_macro('DXY')
if dxy.empty:
    print('📦 抓取 DXY 美元指數...')
    dxy = fetch_dxy(start='2018-01-01')
    store.save_macro('DXY', dxy.to_frame('value').reset_index())
print(f'📦 DXY: {len(dxy)} 筆' if not dxy.empty else '⚠️ DXY 無資料')

vix = store.load_macro('VIX')
if vix.empty:
    print('📦 抓取 VIX 恐慌指數...')
    vix = fetch_vix(start='2018-01-01')
    store.save_macro('VIX', vix.to_frame('value').reset_index())
print(f'📦 VIX:  {len(vix)} 筆' if not vix.empty else '⚠️ VIX 無資料')
print()

# ── 5. 計算所有因子 ──
factor_results = []
FACTOR_PARAMS = {
    'momentum': {},
    'value': {'fundamentals': fundamentals},
    'quality': {'fundamentals': fundamentals},
    'low_vol': {},
    'size': {'fundamentals': fundamentals},
    'div_yield': {'fundamentals': fundamentals},
    'revenue_growth': {'fundamentals': fundamentals},
    'industry_momentum': {'sector_map': sector_map},
    'flow': {},
    'fx_exposure': {'dxy': dxy, 'sector_map': sector_map},
}

for name in FACTOR_WEIGHTS:
    if name not in FACTOR_PARAMS:
        print(f'⚠️ 跳過未知因子: {name}')
        continue
    params = FACTOR_PARAMS[name]
    r = get_factor(name).compute(price_data, **params)
    if not r.scores.empty:
        factor_results.append(r)
        print(f'✅ {name:>20}: {len(r.scores):>6} 筆分數')
    else:
        print(f'⚠️ {name:>20}: 無效結果')
print()

# ── 6. 信號合成 + VIX 調整 ──
signal = SignalCombiner(FACTOR_WEIGHTS).combine(factor_results)
print(f'📊 綜合信號: {len(signal)} 筆')

# VIX 市場狀態調整
vix_current = vix.iloc[-1] if not vix.empty else 0
print(f'📊 當前 VIX: {vix_current:.2f}')
signal_adj = regime_adjust_weights(signal, vix)
print(f'📊 VIX 調整後信號: {len(signal_adj)} 筆')
print()

# ── 7. 回測（原始信號） ──
engine = BacktestEngine(initial_capital=INITIAL, max_holding=MAX_HOLD,
                        rebalance_freq=REBALANCE_FREQUENCY, transaction_cost=0.001)
bt = engine.run(price_data, signal)

# 也用調整後信號跑一次回測
bt_adj = engine.run(price_data, signal_adj)

perf = PerformanceAnalyzer().analyze(bt.returns)
perf_adj = PerformanceAnalyzer().analyze(bt_adj.returns)

print('=' * 60)
print(f'  📊 回測績效對比（$100K | {MAX_HOLD}檔 | 2018~2026）')
print('=' * 60)
print(f'{"指標":>15} {"原始信號":>12} {"+VIX調整":>12}')
print('-' * 60)
print(f'{"累積報酬":>15} {perf["total_return"]*100:>10.2f}% {perf_adj["total_return"]*100:>10.2f}%')
print(f'{"年化報酬":>15} {perf["annualized_return"]*100:>10.2f}% {perf_adj["annualized_return"]*100:>10.2f}%')
print(f'{"年化波動":>15} {perf["annualized_volatility"]*100:>10.2f}% {perf_adj["annualized_volatility"]*100:>10.2f}%')
print(f'{"夏普比率":>15} {perf["sharpe_ratio"]:>10.3f} {perf_adj["sharpe_ratio"]:>10.3f}')
print(f'{"最大回撤":>15} {perf["max_drawdown"]*100:>10.2f}% {perf_adj["max_drawdown"]*100:>10.2f}%')
print(f'{"勝率":>15} {perf["win_rate"]*100:>10.2f}% {perf_adj["win_rate"]*100:>10.2f}%')
print(f'{"終值":>15} ${bt.portfolio_value.iloc[-1]:>9,.0f} ${bt_adj.portfolio_value.iloc[-1]:>9,.0f}')
print()

# ── 8. 最新因子信號 → 今晚買入清單 ──
latest_date = signal_adj.index.get_level_values('date').max()
latest_signal = signal_adj.xs(latest_date, level='date').sort_values(ascending=False)
top = latest_signal.head(MAX_HOLD)

# 最新收盤價
price_matrix = pd.concat({t: df['Close'] for t, df in price_data.items()}, axis=1)
price_matrix = price_matrix.ffill()
latest_prices = price_matrix.iloc[-1]

# 只保留有價格的 ticker
top = top[top.index.isin(latest_prices.index)]
n = len(top)

# 因子加權配置
scores = top.values
min_score = scores.min()
if min_score < 0:
    scores = scores - min_score + 0.01
weights_pct = scores / scores.sum()

# 計算每檔配置金額與股數
allocations = []
total_cost = 0
for ticker, wgt in zip(top.index, weights_pct):
    amount = INITIAL * wgt
    price = latest_prices[ticker]
    shares = max(int(amount / price), 1)
    cost = shares * price
    if shares == 1 and cost > INITIAL * 0.05:
        continue
    total_cost += cost
    allocations.append((ticker, price, shares, cost, wgt * 100))

# 找出每檔的強勢因子（貢獻最大的 top-3 因子）
def top_factors_for_stock(ticker, factor_results, weights):
    scores = {}
    for fr in factor_results:
        w = weights.get(fr.name, 0)
        if w == 0:
            continue
        if fr.scores.empty:
            continue
        try:
            val = fr.scores.xs(ticker, level='ticker').iloc[-1]
            scores[fr.name] = val * w
        except (KeyError, IndexError):
            continue
    return sorted(scores.items(), key=lambda x: -x[1])[:3]

print('=' * 60)
vix_label = '⚠️ 恐慌' if vix_current > 30 else ('🟢 安全' if vix_current < 15 else '🟡 警戒')
print(f'  📋 今晚買入清單（10因子 + {vix_label} VIX={vix_current:.1f}）')
print(f'     初始資金: ${INITIAL:,}')
print('=' * 60)
print(f'{"排名":>4} {"股票":>6} {"綜合信號":>9} {"權重":>7} {"現價":>8} {"股數":>6} {"預算":>10} {"強勢因子":>25}')
print('-' * 60)
for rank, (ticker, price, shares, cost, pct) in enumerate(allocations, 1):
    tf = top_factors_for_stock(ticker, factor_results, FACTOR_WEIGHTS)
    tf_str = ', '.join([f'{n}({v:.2f})' for n, v in tf[:2]])
    print(f'{rank:>4} {ticker:>6} {top[ticker]:>8.3f} {pct:>5.1f}% {price:>8.2f} {shares:>6} {cost:>10,.0f} {tf_str:>25}')

remaining = INITIAL - total_cost
print('-' * 60)
print(f'{"合計":>32} {total_cost:>10,.0f}')
print(f'{"餘額":>32} {remaining:>10,.0f}')
print()

print('=' * 60)
print('  💡 執行建議')
print('=' * 60)
print(f'  1. 市價單（Market Order，當日有效，未成交盤後自動取消）')
print(f'  2. 如果開盤跳空超過 ±2%，等價格回到合理區間再買')
print(f'  3. 每兩週再平衡（biweekly）')
print(f'  4. 現金緩衝 ${remaining:,.0f} 可留著或補到權重最高的標的')
if vix_current > 25:
    print(f'  5. ⚠️ VIX={vix_current:.1f} 偏高，建議分 2-3 天建倉，降低一次性風險')
print()

print('=' * 60)
print('  📌 買入價格（實際下單用市價單）')
print('=' * 60)
for rank, (ticker, price, shares, cost, pct) in enumerate(allocations, 1):
    market_price = round(price, 2)
    print(f'  {ticker:>6}: 買 {shares} 股 × 市價 ${market_price} = ${cost:,.0f}')
