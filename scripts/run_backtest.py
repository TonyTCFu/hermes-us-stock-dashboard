#!/usr/bin/env python3
"""命令列回測工具。

Usage:
    python scripts/run_backtest.py --tickers AAPL,MSFT --start 2018-01-01
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from us_quant.config import FACTOR_WEIGHTS, INITIAL_CAPITAL, BENCHMARK_TICKER
from us_quant.data import fetch_price_data, fetch_all_fundamentals
from us_quant.factors import get_factor
from us_quant.signals import SignalCombiner
from us_quant.backtest import BacktestEngine
from us_quant.reporting import PerformanceAnalyzer
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="命令列回測")
    parser.add_argument("--tickers", required=True, help="股票列表（逗號分隔）")
    parser.add_argument("--start", default="2018-01-01", help="開始日期")
    parser.add_argument("--end", default=None, help="結束日期")
    parser.add_argument("--capital", type=float, default=INITIAL_CAPITAL, help="初始資金")
    parser.add_argument("--max-holdings", type=int, default=20, help="最大持倉數量")
    parser.add_argument("--rebalance", default="monthly", choices=["monthly", "quarterly", "yearly"])
    args = parser.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",")]

    print("📥 載入資料...")
    price_data = fetch_price_data(tickers, args.start, args.end)
    price_data = {k: v for k, v in price_data.items() if not v.empty}
    fundamentals = fetch_all_fundamentals(list(price_data.keys()))

    print("🧮 計算因子...")
    factor_results = []
    for factor_name, weight in FACTOR_WEIGHTS.items():
        if weight == 0:
            continue
        factor = get_factor(factor_name)
        params = {"fundamentals": fundamentals} if factor_name in ("value", "quality", "size", "div_yield", "revenue_growth") else {}
        result = factor.compute(price_data, **params)
        if not result.scores.empty:
            factor_results.append(result)
            print(f"  {factor_name}: {len(result.scores)} 筆")

    combiner = SignalCombiner(FACTOR_WEIGHTS)
    signal = combiner.combine(factor_results)

    print("🏃 執行回測...")
    engine = BacktestEngine(
        initial_capital=args.capital,
        max_holding=args.max_holdings,
        rebalance_freq=args.rebalance,
    )
    bt = engine.run(price_data, signal)

    print("📊 績效分析...")
    analyzer = PerformanceAnalyzer()
    perf = analyzer.analyze(bt.returns)

    print("\n" + "=" * 50)
    print("回測結果")
    print("=" * 50)
    print(f"期間:           {perf.get('date_range', 'N/A')}")
    print(f"累積報酬:       {perf.get('total_return', 'N/A')}")
    print(f"年化報酬:       {perf.get('annualized_return', 'N/A')}")
    print(f"年化波動:       {perf.get('annualized_volatility', 'N/A')}")
    print(f"夏普比率:       {perf.get('sharpe_ratio', 'N/A')}")
    print(f"最大回撤:       {perf.get('max_drawdown', 'N/A')}")
    print(f"勝率:           {perf.get('win_rate', 'N/A')}")
    print(f"獲利因子:       {perf.get('profit_factor', 'N/A')}")
    print("=" * 50)


if __name__ == "__main__":
    main()
