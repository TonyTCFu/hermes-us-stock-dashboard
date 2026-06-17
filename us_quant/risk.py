"""风控模块：个券止损止盈 + 宏观事件覆盖 + 信号强度仓位分配"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ── 宏观事件日历 ──
# 格式: "YYYY-MM-DD" → event_type
# event_type: fomc / cpi / nfp
# 需定期更新（FOMC 每年 8 次，CPI 每月中，NFP 每月第一个周五）
MACRO_CALENDAR: dict[str, str] = {
    # 2026 Q3-Q4 (预估，实际日期以美联储公告为准)
    "2026-06-17": "fomc",
    "2026-07-10": "cpi",
    "2026-07-15": "nfp",
    "2026-07-29": "fomc",
    "2026-08-07": "nfp",
    "2026-08-12": "cpi",
    "2026-09-04": "nfp",
    "2026-09-11": "cpi",
    "2026-09-23": "fomc",
    "2026-10-02": "nfp",
    "2026-10-13": "cpi",
    "2026-11-06": "fomc",
    "2026-11-06": "nfp",
    "2026-11-13": "cpi",
    "2026-12-04": "nfp",
    "2026-12-11": "cpi",
    "2026-12-16": "fomc",
}

# 宏观事件仓位系数
MACRO_MULTIPLIER: dict[str, float] = {
    "fomc": 0.50,  # FOMC 日仓位减半
    "cpi": 0.75,   # CPI 日仓位 75%
    "nfp": 0.75,   # NFP 日仓位 75%
}


class RiskManager:
    """交易风控管理器。

    - 个券止损 5% / 止盈 10%
    - 宏观事件日降仓
    - VIX 市场状态调节
    - 信号强度加权仓位分配
    """

    def __init__(
        self,
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.10,
        max_holding: int = 8,
    ):
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_holding = max_holding

    # ── 止损止盈 ──

    def check_sl_tp(self, positions: pd.DataFrame) -> tuple[list[str], list[str]]:
        """检查所有持仓的止损止盈触发。

        Returns
        -------
        (stop_loss_hits, take_profit_hits): 各为 ticker list
        """
        sl_hits: list[str] = []
        tp_hits: list[str] = []

        for _, p in positions.iterrows():
            pnl_pct = (float(p["current_price"]) - float(p["avg_price"])) / float(p["avg_price"])
            ticker = p["ticker"]

            if pnl_pct <= -self.stop_loss_pct:
                sl_hits.append(ticker)
                logger.warning("🛑 %s 触发止损: %.1f%% (成本 $%.2f → 现价 $%.2f)",
                               ticker, pnl_pct * 100, float(p["avg_price"]), float(p["current_price"]))
            elif pnl_pct >= self.take_profit_pct:
                tp_hits.append(ticker)
                logger.info("🎯 %s 触发止盈: %.1f%% (成本 $%.2f → 现价 $%.2f)",
                            ticker, pnl_pct * 100, float(p["avg_price"]), float(p["current_price"]))

        return sl_hits, tp_hits

    # ── 宏观覆盖 ──

    def get_macro_multiplier(self, dt: Optional[date] = None, vix: float = 0) -> float:
        """综合宏观仓位系数 (0.0 ~ 1.0)。

        取宏观事件和 VIX 调节中较严格的那个。
        """
        if dt is None:
            dt = date.today()

        dt_str = dt.strftime("%Y-%m-%d")

        # 宏观事件
        event_type = MACRO_CALENDAR.get(dt_str)
        event_mult = MACRO_MULTIPLIER.get(event_type, 1.0) if event_type else 1.0

        # VIX 调节（来自 Hermes 原有逻辑）
        if vix > 30:
            vix_mult = 0.50
        elif vix > 25:
            vix_mult = 0.65
        elif vix > 20:
            vix_mult = max(0.70, 1.0 - (vix - 20) / 30)
        else:
            vix_mult = 1.0

        multiplier = min(event_mult, vix_mult)

        if multiplier < 1.0:
            reasons = []
            if event_type:
                reasons.append(f"{event_type.upper()}: ×{event_mult:.0%}")
            if vix_mult < 1.0:
                reasons.append(f"VIX {vix:.1f}: ×{vix_mult:.0%}")
            logger.info("📊 宏观系数: %.0f%%（%s）", multiplier * 100, ", ".join(reasons))

        return multiplier

    # ── 仓位分配 ──

    def size_positions(
        self,
        signals: pd.Series,
        total_capital: float,
        prices: pd.Series,
        macro_mult: float = 1.0,
        cash_buffer: float = 0.95,
    ) -> list[dict]:
        """根据信号强度分配仓位权重。

        信号越强的股票分配越多资金。保证每只入选股票至少有最低权重。
        """
        top = signals.nlargest(self.max_holding)
        if top.empty:
            return []

        n = len(top)

        # 信号强度加权：softmax 风格
        # weight ∝ exp(signal / temperature)，temperature=2 保持适度集中
        centered = top - top.mean()
        exp_scores = (centered / 2).clip(lower=-5, upper=5).apply(np.exp)
        raw_weights = exp_scores / exp_scores.sum()

        # 保证最低权重 = 等权的 50%（避免最低信号股票分到 0）
        floor = 1.0 / (n * 2)
        weights = raw_weights.copy()
        deficit = 0.0
        for t in weights.index:
            if weights[t] < floor:
                deficit += floor - weights[t]
                weights[t] = floor
        # 从高于 floor 的股票中扣除 deficit
        above = weights[weights > floor]
        if deficit > 0 and not above.empty:
            trim_total = (above - floor).sum()
            for t in above.index:
                weights[t] -= deficit * (weights[t] - floor) / trim_total

        weights = weights / weights.sum()  # 最终归一化

        # 应用宏观系数
        deployable = total_capital * macro_mult * cash_buffer

        positions = []
        for ticker in top.index:
            w = weights[ticker]
            target_value = deployable * w
            price = prices.get(ticker, 0)
            if price <= 0:
                continue
            shares = max(int(target_value / price), 1)
            actual_value = shares * price
            positions.append({
                "ticker": ticker,
                "target_value": round(actual_value, 2),
                "weight": round(w, 4),
                "shares": shares,
                "price": round(float(price), 2),
                "signal": round(float(top[ticker]), 4),
            })

        return positions

    # ── 调仓触发判断 ──

    def should_rebalance(
        self,
        current_holdings: set[str],
        new_top: set[str],
        days_since_last: int,
        sl_tp_hits: list[str],
    ) -> tuple[bool, str]:
        """判断是否需要调仓。

        Returns
        -------
        (should_rebalance, reason)
        """
        # 1. 止损/止盈触发 → 必须调
        if sl_tp_hits:
            return True, f"止损/止盈触发: {', '.join(sl_tp_hits)}"

        # 2. Top-N 名单变化 ≥ 2 檔 → 调仓
        changed = len(current_holdings.symmetric_difference(new_top))
        if changed >= 2:
            added = new_top - current_holdings
            removed = current_holdings - new_top
            parts = []
            if added:
                parts.append(f"新增 {', '.join(sorted(added))}")
            if removed:
                parts.append(f"移除 {', '.join(sorted(removed))}")
            return True, "; ".join(parts)

        # 3. 超过 3 个交易日未调仓 → 强制检查
        if days_since_last >= 3:
            return True, f"距上次调仓 {days_since_last} 天，强制刷新"

        return False, f"无变化（已过 {days_since_last} 天）"


def load_macro_calendar_from_csv(path: str) -> None:
    """从 CSV 加载宏观事件日历（格式: date,event_type）。"""
    import csv
    try:
        with open(path) as f:
            for row in csv.DictReader(f):
                MACRO_CALENDAR[row["date"]] = row["event_type"]
        logger.info("已载入宏观日历: %d 个事件", len(MACRO_CALENDAR))
    except Exception as e:
        logger.warning("载入宏观日历失败: %s", e)
