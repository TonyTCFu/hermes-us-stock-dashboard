#!/usr/bin/env python3
"""信号驱动调仓：8因子 + AI主题 + 止损止盈 + 宏观覆盖

调仓逻辑（激进短期）：
1. 每日检查持仓止损(-5%)/止盈(+10%)，触发即平仓
2. 计算 8 因子信号排名
3. 调仓触发条件（满足任一即调）：
   - SL/TP 触发
   - Top-N 名单变化 ≥ 2 檔
   - 距上次调仓 ≥ 3 个交易日
4. 宏观事件日自动降仓
5. 信号强度加权分配仓位（越强越多）
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import json
import time
import logging
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np

from us_quant.data import (
    ensure_price_data, fetch_all_fundamentals, fetch_sector_info,
    fetch_dxy, fetch_vix, DataStore, fetch_price_data,
)
from us_quant.factors import get_factor
from us_quant.signals import SignalCombiner, regime_adjust_weights
from us_quant.config import DB_PATH, STOCK_UNIVERSE, FACTOR_WEIGHTS, MAX_HOLDING
from us_quant.risk import RiskManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Alpaca ──
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

# ── 策略参数 ──
MAX_HOLD = MAX_HOLDING          # 最大持仓数（默认 8）
CASH_BUFFER = 0.95              # 保留 5% 现金
SL_PCT = 0.05                   # 个券止损 5%
TP_PCT = 0.10                   # 个券止盈 10%
MAX_IDLE_DAYS = 3               # 超过此天数强制调仓
CHANGE_THRESHOLD = 2            # Top-N 名单变化超过此数触发调仓

STATE_FILE = Path(__file__).resolve().parent.parent / "data" / "rebalance_state.json"
ACTIVITY_FILE = Path(__file__).resolve().parent.parent / "data" / "activity.json"

# 仅需价格数据的因子（无额外依赖）
PRICE_ONLY_FACTORS = {"momentum", "low_vol", "flow", "ai_industry"}

# 因子参数模板
FACTOR_PARAMS: dict[str, dict] = {
    "momentum": {},
    "value": {"fundamentals": None},
    "quality": {"fundamentals": None},
    "low_vol": {},
    "revenue_growth": {"fundamentals": None},
    "industry_momentum": {"sector_map": None},
    "flow": {},
    "ai_industry": {"benchmark": None},
}


# ══════════════════════════════════════════════════════════════════════
# 状态管理
# ══════════════════════════════════════════════════════════════════════

def load_state() -> dict:
    """加载调仓状态。"""
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {"last_rebalance": None, "total_trades": 0, "holdings_snapshot": []}


def save_state(holdings: list[str]) -> None:
    """保存调仓状态。"""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "last_rebalance": date.today().isoformat(),
        "total_trades": load_state()["total_trades"] + 1,
        "holdings_snapshot": sorted(holdings),
    }
    STATE_FILE.write_text(json.dumps(state, indent=2))


def save_activity(
    action: str,
    details: dict,
    positions_before: pd.DataFrame,
    positions_after: pd.DataFrame | None = None,
) -> None:
    """追加调仓活动记录。"""
    ACTIVITY_FILE.parent.mkdir(parents=True, exist_ok=True)
    # 读取已有记录
    if ACTIVITY_FILE.exists():
        try:
            history = json.loads(ACTIVITY_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            history = []
    else:
        history = []

    entry = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M ET"),
        "action": action,
        "equity": details.get("equity", 0),
        "cash": details.get("cash", 0),
        "vix": details.get("vix", 0),
        "macro_mult": details.get("macro_mult", 1.0),
        "details": details,
        "positions_before": [
            {"ticker": r["ticker"], "qty": int(r["qty"]), "pnl_pct": round(
                (float(r["current_price"]) - float(r["avg_price"])) / float(r["avg_price"]) * 100, 1
            ) if float(r["avg_price"]) > 0 else 0}
            for _, r in positions_before.iterrows()
        ] if not positions_before.empty else [],
    }
    if positions_after is not None and not positions_after.empty:
        entry["positions_after"] = [
            {"ticker": r["ticker"], "qty": int(r["qty"]), "pnl_pct": round(
                (float(r["current_price"]) - float(r["avg_price"])) / float(r["avg_price"]) * 100, 1
            ) if float(r["avg_price"]) > 0 else 0}
            for _, r in positions_after.iterrows()
        ]

    history.append(entry)
    # 保留最近 60 条
    if len(history) > 60:
        history = history[-60:]

    ACTIVITY_FILE.write_text(json.dumps(history, indent=2, ensure_ascii=False))


def days_since_last() -> int:
    """距上次调仓的日历天数。"""
    last = load_state()["last_rebalance"]
    if last is None:
        return 999
    return (date.today() - date.fromisoformat(last)).days


# ══════════════════════════════════════════════════════════════════════
# Alpaca
# ══════════════════════════════════════════════════════════════════════

def get_alpaca_client() -> TradingClient:
    return TradingClient(
        os.getenv("ALPACA_API_KEY"),
        os.getenv("ALPACA_SECRET_KEY"),
        paper=True,
    )


def get_current_positions(client: TradingClient) -> tuple[float, float, pd.DataFrame]:
    """获取当前账户状态。"""
    account = client.get_account()
    cash = float(account.cash)
    equity = float(account.equity)

    positions = client.get_all_positions()
    holdings = []
    for p in positions:
        holdings.append({
            "ticker": p.symbol,
            "qty": float(p.qty),
            "avg_price": float(p.avg_entry_price),
            "current_price": float(p.current_price),
            "market_value": float(p.market_value),
            "unrealized_pl": float(p.unrealized_pl),
        })

    return cash, equity, pd.DataFrame(holdings) if holdings else pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════
# 因子分析
# ══════════════════════════════════════════════════════════════════════

def run_factor_analysis(store: DataStore):
    """运行 8 因子分析，返回排名 DataFrame 和元数据。"""
    # 价格数据
    price_data = ensure_price_data(STOCK_UNIVERSE, store, start="2018-01-01")
    price_data = {k: v for k, v in price_data.items() if not v.empty}

    if not price_data:
        raise RuntimeError("无有效价格数据")

    tickers = list(price_data.keys())

    # 基本面
    fundamentals = fetch_all_fundamentals(tickers)
    store.save_fundamentals(fundamentals)

    # 行业映射
    sector_map = store.load_sector_mapping()
    if sector_map.empty:
        sector_map = fetch_sector_info(tickers)
        store.save_sector_mapping(sector_map)

    # DXY / VIX
    dxy = store.load_macro("DXY")
    if dxy.empty:
        dxy = fetch_dxy(start="2018-01-01")
        store.save_macro("DXY", dxy.to_frame("value").reset_index())

    vix = store.load_macro("VIX")
    if vix.empty:
        vix = fetch_vix(start="2018-01-01")
        store.save_macro("VIX", vix.to_frame("value").reset_index())

    # SPY 基准（AI 因子需要）
    spy_raw = fetch_price_data(["SPY"], start="2018-01-01")
    spy_df = spy_raw.get("SPY", pd.DataFrame())

    # 组装参数
    params = {k: dict(v) for k, v in FACTOR_PARAMS.items()}
    params["value"]["fundamentals"] = fundamentals
    params["quality"]["fundamentals"] = fundamentals
    params["revenue_growth"]["fundamentals"] = fundamentals
    params["industry_momentum"]["sector_map"] = sector_map
    params["ai_industry"]["benchmark"] = spy_df

    factor_results = []
    for name in FACTOR_WEIGHTS:
        if name not in params:
            logger.warning("因子 %s 不在参数表中，跳过", name)
            continue
        try:
            r = get_factor(name).compute(price_data, **params[name])
            if not r.scores.empty:
                factor_results.append(r)
                logger.info("  ✓ %s: %d 条记录", name, len(r.scores))
            else:
                logger.warning("  ✗ %s: 无数据", name)
        except Exception as e:
            logger.warning("  ✗ %s: %s", name, e)

    if not factor_results:
        raise RuntimeError("所有因子计算失败")

    # 信号合成
    signal = SignalCombiner(FACTOR_WEIGHTS).combine(factor_results)

    # VIX 调节
    vix_current = float(vix.iloc[-1]) if not vix.empty else 0
    signal_adj = regime_adjust_weights(signal, vix)

    # 每档股票最新分数
    latest_signal = signal_adj.groupby(level=0).last().sort_values(ascending=False)

    # 最新收盤价
    price_matrix = pd.concat({t: df["Close"] for t, df in price_data.items()}, axis=1).ffill()
    latest_prices = price_matrix.iloc[-1]

    # 只保留有价格的
    all_scores = latest_signal[latest_signal.index.isin(latest_prices.index)]

    # 构建排名表
    info = []
    for rank_idx, t in enumerate(all_scores.index, start=1):
        # 强势因子
        top_f = []
        for fr in factor_results:
            w = FACTOR_WEIGHTS.get(fr.name, 0)
            if w == 0 or fr.scores.empty:
                continue
            try:
                val = fr.scores.xs(t, level="ticker").iloc[-1]
                top_f.append((fr.name, round(val * w, 3)))
            except (KeyError, IndexError):
                continue
        top_f.sort(key=lambda x: -x[1])

        info.append({
            "ticker": t,
            "signal": round(float(all_scores[t]), 4),
            "rank": rank_idx,
            "price": round(float(latest_prices[t]), 2),
            "top_factors": ", ".join([f"{n}({v:.2f})" for n, v in top_f[:2]]),
        })

    df = pd.DataFrame(info).set_index("ticker")
    return df, vix_current, all_scores, price_data


# ══════════════════════════════════════════════════════════════════════
# 调仓决策
# ══════════════════════════════════════════════════════════════════════

def close_positions_simple(client: TradingClient, tickers: list[str], positions: pd.DataFrame) -> int:
    """平仓指定 ticker（用于调仓卖出，非 SL/TP）。"""
    success = 0
    for t in tickers:
        row = positions[positions["ticker"] == t]
        if row.empty:
            continue
        try:
            qty = int(row.iloc[0]["qty"])
            order = MarketOrderRequest(
                symbol=t, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY,
            )
            client.submit_order(order)
            logger.info("  ✅ %s: 卖出 %d 股", t, qty)
            success += 1
        except Exception as e:
            logger.error("  ❌ %s: 卖出失败 — %s", t, e)
        time.sleep(0.3)
    return success


def handle_sl_tp(
    client: TradingClient,
    risk: RiskManager,
    positions: pd.DataFrame,
    top_tickers: set[str],
    dry_run: bool = False,
) -> tuple[list[str], list[str]]:
    """处理止损止盈。

    - SL 触发 → 全部卖出
    - TP 触发 + 仍在 Top-N → 卖一半（锁利，让利润跑）
    - TP 触发 + 不在 Top-N → 全部卖出
    """
    sl_hits, tp_hits = risk.check_sl_tp(positions)
    closed_full: list[str] = []
    closed_half: list[str] = []

    for t in sl_hits:
        # 止损：全部卖出
        row = positions[positions["ticker"] == t]
        if row.empty:
            continue
        qty = int(row.iloc[0]["qty"])
        if not dry_run:
            try:
                order = MarketOrderRequest(
                    symbol=t, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY,
                )
                client.submit_order(order)
                logger.info("  🛑 %s: 止损全卖 %d 股", t, qty)
            except Exception as e:
                logger.error("  ❌ %s: 止损卖失败 — %s", t, e)
        else:
            logger.info("  🔍 %s: 止损全卖 %d 股（仅分析）", t, qty)
        closed_full.append(t)
        time.sleep(0.3)

    for t in tp_hits:
        row = positions[positions["ticker"] == t]
        if row.empty:
            continue
        qty = int(row.iloc[0]["qty"])

        if t in top_tickers:
            # 止盈 + 信号仍强 → 卖一半
            half = max(qty // 2, 1)
            if not dry_run:
                try:
                    order = MarketOrderRequest(
                        symbol=t, qty=half, side=OrderSide.SELL, time_in_force=TimeInForce.DAY,
                    )
                    client.submit_order(order)
                    logger.info("  🎯 %s: 止盈半卖 %d/%d 股（信号仍强）", t, half, qty)
                except Exception as e:
                    logger.error("  ❌ %s: 止盈半卖失败 — %s", t, e)
            else:
                logger.info("  🔍 %s: 止盈半卖 %d/%d 股（仅分析）", t, half, qty)
            closed_half.append(t)
        else:
            # 止盈 + 信号不再强 → 全部卖出
            if not dry_run:
                try:
                    order = MarketOrderRequest(
                        symbol=t, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY,
                    )
                    client.submit_order(order)
                    logger.info("  🎯 %s: 止盈全卖 %d 股（信号弱化）", t, qty)
                except Exception as e:
                    logger.error("  ❌ %s: 止盈全卖失败 — %s", t, e)
            else:
                logger.info("  🔍 %s: 止盈全卖 %d 股（仅分析）", t, qty)
            closed_full.append(t)
        time.sleep(0.3)

    return closed_full, closed_half


def execute_rebalance(
    client: TradingClient,
    risk: RiskManager,
    positions: pd.DataFrame,
    cash: float,
    all_scores_df: pd.DataFrame,
    latest_prices: pd.Series,
    macro_mult: float,
) -> dict:
    """执行完整调仓。

    Returns
    -------
    dict with sell_count, buy_count, new_holdings
    """
    current_tickers = set(positions["ticker"]) if not positions.empty else set()
    target_top = all_scores_df.head(MAX_HOLD)
    target_tickers = set(target_top.index)

    # ── 卖：不在 Top-N 的持仓 ──
    sell_tickers = [t for t in current_tickers if t not in target_tickers]
    sell_count = close_positions_simple(client, sell_tickers, positions) if sell_tickers else 0

    # 更新现金和持仓
    cash2, _, positions2 = get_current_positions(client)
    current_tickers2 = set(positions2["ticker"]) if not positions2.empty else set()

    # ── 信号强度加权仓位分配 ──
    positions_new = risk.size_positions(
        target_top["signal"],
        cash2 + (positions2["market_value"].sum() if not positions2.empty else 0),
        latest_prices,
        macro_mult=macro_mult,
        cash_buffer=CASH_BUFFER,
    )

    # ── 买：需要新增或补仓的 ──
    buy_count = 0
    current_values = {}
    if not positions2.empty:
        for _, p in positions2.iterrows():
            current_values[p["ticker"]] = float(p["market_value"])

    for target in positions_new:
        t = target["ticker"]
        current_val = current_values.get(t, 0)
        diff = target["target_value"] - current_val

        if diff <= 0:
            # 已持足够，跳过
            continue

        if t in current_tickers2:
            # 补仓
            qty = max(int(diff / target["price"]), 1)
            logger.info("  📈 %s: 补仓 %d 股 (目标 $%.0f, 当前 $%.0f)", t, qty, target["target_value"], current_val)
        else:
            # 新买入
            qty = target["shares"]
            logger.info("  🆕 %s: 新买入 %d 股 (目标 $%.0f)", t, qty, target["target_value"])

        try:
            order = MarketOrderRequest(
                symbol=t, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
            )
            client.submit_order(order)
            buy_count += 1
        except Exception as e:
            logger.error("  ❌ %s: 买入失败 — %s", t, e)
        time.sleep(0.3)

    return {
        "sell_count": sell_count,
        "buy_count": buy_count,
        "new_holdings": [p["ticker"] for p in positions_new],
    }


# ══════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="信号驱动调仓 v2")
    parser.add_argument("--auto", "-y", action="store_true", help="非交互模式（调度任务用）")
    parser.add_argument("--dry-run", "-n", action="store_true", help="仅分析不执行")
    args = parser.parse_args()

    store = DataStore(DB_PATH)
    client = get_alpaca_client()
    risk = RiskManager(stop_loss_pct=SL_PCT, take_profit_pct=TP_PCT, max_holding=MAX_HOLD)

    print()
    print("=" * 64)
    print(f"  🔄 信号驱动调仓 v2 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  策略: 8因子 + AI主题 | 激进短期 | SL5%/TP10%")
    print("=" * 64)
    print()

    # ── 1. 当前状态 ──
    print("📊 Alpaca 账户状态...")
    cash, equity, positions = get_current_positions(client)
    print(f"   权益: ${equity:,.2f}  现金: ${cash:,.2f}  持仓: {len(positions)} 档")
    print()

    if not positions.empty:
        print(f"   {'股票':>6} {'股数':>6} {'成本':>9} {'现价':>9} {'市值':>10} {'盈亏%':>7}")
        print("   " + "-" * 50)
        for _, p in positions.iterrows():
            pnl = (float(p["current_price"]) - float(p["avg_price"])) / float(p["avg_price"]) * 100
            print(f"   {p['ticker']:>6} {p['qty']:>6.0f} ${float(p['avg_price']):>8.2f} "
                  f"${float(p['current_price']):>8.2f} ${float(p['market_value']):>8,.0f} {pnl:>+6.1f}%")
        print()

    # ── 2. 因子分析 ──
    print("🔬 8 因子分析中...")
    all_scores, vix, signal_series, price_data = run_factor_analysis(store)

    # 最新价格
    price_matrix = pd.concat({t: df["Close"] for t, df in price_data.items()}, axis=1).ffill()
    latest_prices = price_matrix.iloc[-1]

    # VIX 状态
    vix_label = "⚠️ 恐慌" if vix > 30 else "🟡 警戒" if vix > 20 else "🟢 正常"
    print(f"   VIX: {vix:.2f} ({vix_label})")
    print()

    # ── 3. 止损止盈检查（用因子结果判断半卖/全卖） ──
    top_tickers_set = set(all_scores.head(MAX_HOLD).index)
    urgent_close_all: list[str] = []
    urgent_close_half: list[str] = []

    if not positions.empty:
        sl_hits, tp_hits = risk.check_sl_tp(positions)
        if sl_hits or tp_hits:
            print(f"⚠️  风控触发: {'止损 ' + ', '.join(sl_hits) if sl_hits else ''}"
                  f"{' 止盈 ' + ', '.join(tp_hits) if tp_hits else ''}")
            closed_full, closed_half = handle_sl_tp(
                client, risk, positions, top_tickers_set, dry_run=args.dry_run,
            )
            urgent_close_all = closed_full
            urgent_close_half = closed_half
            # 刷新状态
            if not args.dry_run and (closed_full or closed_half):
                time.sleep(1)
                cash, equity, positions = get_current_positions(client)
                # 仓位变化后重新跑因子（可选：如果变化不大可跳过）
                if closed_full:
                    print("   🔬 仓位变化，重新分析...")
                    all_scores, vix, signal_series, price_data = run_factor_analysis(store)
                    price_matrix = pd.concat({t: df["Close"] for t, df in price_data.items()}, axis=1).ffill()
                    latest_prices = price_matrix.iloc[-1]
                    top_tickers_set = set(all_scores.head(MAX_HOLD).index)
            print()
        else:
            print("   ✅ 无 SL/TP 触发")

    # ── 4. 排名展示 ──
    top_n = all_scores.head(MAX_HOLD)
    current_set = set(positions["ticker"]) if not positions.empty else set()
    # 合并 SL/TP 全部平仓的股票
    sl_tp_hits = urgent_close_all + urgent_close_half
    print(f"   📋 Top {MAX_HOLD} 信号排名:")
    print(f"   {'排名':>4} {'股票':>6} {'信号':>8} {'状态':>10} {'现价':>8} {'强勢因子':>30}")
    print("   " + "-" * 66)
    for _, row in top_n.iterrows():
        status = "✅ 持有" if row.name in current_set else "🆕 新标的"
        if row.name in urgent_close_half:
            status = "🎯 半仓"
        print(f"   {row['rank']:>4} {row.name:>6} {row['signal']:>8.3f} "
              f"{status:>10} ${row['price']:>7.2f} {row['top_factors']:>30}")
    print()

    # 当前持股排名
    if not positions.empty:
        target_set = set(top_n.index)
        print(f"   📋 当前持股诊断:")
        print(f"   {'股票':>6} {'排名':>6} {'信号':>8} {'决策':>12}")
        print("   " + "-" * 34)
        for _, p in positions.iterrows():
            t = p["ticker"]
            if t in all_scores.index:
                rank = all_scores.loc[t, "rank"]
                sig = all_scores.loc[t, "signal"]
                if t in urgent_close_all:
                    decision = "🛑 强制平仓"
                elif t in urgent_close_half:
                    decision = "🎯 半仓锁利"
                elif t in target_set:
                    decision = "✅ 保留"
                else:
                    decision = "❌ 替换"
                print(f"   {t:>6} {int(rank):>6} {sig:>8.3f} {decision:>12}")
            else:
                print(f"   {t:>6} {'N/A':>6} {'N/A':>8} {'❌ 无信号':>12}")
        print()

    # ── 5. 宏观覆盖 ──
    macro_mult = risk.get_macro_multiplier(dt=date.today(), vix=vix)
    if macro_mult < 1.0:
        print(f"📊 宏观仓位系数: {macro_mult:.0%}")
        print()

    # ── 6. 调仓决策 ──
    state = load_state()
    idle_days = days_since_last()
    previous_holdings = set(state.get("holdings_snapshot", []))

    should, reason = risk.should_rebalance(
        current_holdings=current_set,
        new_top=set(top_n.index),
        days_since_last=idle_days,
        sl_tp_hits=sl_tp_hits,
    )

    print(f"🔍 调仓判断: {'✅ 调仓' if should else '⏸️  跳过'} — {reason}")
    print(f"   上次调仓: {state.get('last_rebalance', '从未')} ({idle_days} 天前)")
    print()

    if not should:
        print("✅ 持倉已最优，跳过调仓")
        return

    # ── 6. 生成调仓计划 ──
    print("=" * 64)
    print("  📋 调仓计划")
    print("=" * 64)

    target_top2 = all_scores.head(MAX_HOLD)
    target_tickers = set(target_top2.index)
    sell_candidates = [t for t in current_set if t not in target_tickers]

    # 仓位分配预览
    planned = risk.size_positions(
        target_top2["signal"], equity, latest_prices, macro_mult=macro_mult, cash_buffer=CASH_BUFFER,
    )

    sell_value = 0
    if not positions.empty:
        sell_value = sum(
            float(positions[positions["ticker"] == t]["market_value"].iloc[0])
            for t in sell_candidates if t in positions["ticker"].values
        )

    buy_value = sum(p["target_value"] for p in planned)

    print(f"   卖 {len(sell_candidates)} 档: {', '.join(sell_candidates) if sell_candidates else '无'} "
          f"(≈ ${sell_value:,.0f})")
    print(f"   买 {len(planned)} 档 (信号加权):")
    for p in planned:
        held = "📌" if p["ticker"] in current_set else "🆕"
        print(f"     {held} {p['ticker']:>6}: {p['shares']:>4}股 × ${p['price']:<8.2f} "
              f"= ${p['target_value']:>8,.0f} (权重 {p['weight']:.1%}, 信号 {p['signal']:+.3f})")
    print(f"   预估交易成本: ${(sell_value + buy_value) * 0.001:,.2f} (0.1%)")
    print()

    # ── 7. 执行 ──
    if args.dry_run:
        print("🔍 仅分析模式，不执行交易")
        return

    if args.auto:
        print("🤖 自动模式：执行调仓")
    else:
        confirm = input("是否执行调仓？(y/n): ").strip().lower()
        if confirm != "y":
            print("❌ 取消调仓")
            return

    # 记录调仓前状态
    positions_before = positions.copy() if not positions.empty else pd.DataFrame()

    result = execute_rebalance(
        client, risk, positions, cash, all_scores, latest_prices, macro_mult,
    )

    # 保存状态 + 活动日志
    save_state(result["new_holdings"])
    fresh_cash, fresh_equity, fresh_positions = get_current_positions(client)
    save_activity(
        action="rebalance",
        details={
            "equity": round(fresh_equity, 2),
            "cash": round(fresh_cash, 2),
            "vix": round(float(vix), 2),
            "macro_mult": round(macro_mult, 2),
            "sell_tickers": [t for t in current_set if t not in set(top_n.index)],
            "buy_tickers": [t for t in set(top_n.index) if t not in current_set],
            "tp_hits": urgent_close_half,
            "sl_hits": urgent_close_all,
            "orders": result["sell_count"] + result["buy_count"],
        },
        positions_before=positions_before,
        positions_after=fresh_positions,
    )

    print()
    print("=" * 64)
    print(f"  ✅ 调仓完成 — 卖 {result['sell_count']} 档 | 买 {result['buy_count']} 档")
    print(f"  新持仓: {', '.join(result['new_holdings'])}")
    print(f"  下次检查: 下一个交易日")
    print("=" * 64)


if __name__ == "__main__":
    main()
