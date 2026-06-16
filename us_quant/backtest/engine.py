"""回測引擎：模擬多因子策略的時間序列回測"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """回測結果。"""
    portfolio_value: pd.Series
    """每日總值"""

    positions: pd.DataFrame
    """每日持倉比例 (ticker × date)"""

    trades: pd.DataFrame
    """交易記錄"""

    returns: pd.Series
    """每日報酬率"""

    benchmark_returns: pd.Series | None = None
    """對應的基準報酬率"""

    metrics: dict = field(default_factory=dict)
    """績效指標"""

    holdings_detail: list[dict] = field(default_factory=list)
    """最新持倉明細：[{ticker, shares, buy_date, buy_price, current_price, cost, market_value, pnl}]"""


class BacktestEngine:
    """多因子加權信號回測引擎。

    支援：
    - 定期再平衡（月 / 季 / 年）
    - 交易成本與滑價
    - 做多限制（不支援放空）
    - 最大持倉數量限制
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
        max_holding: int = 20,
        rebalance_freq: str = "monthly",
    ):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.max_holding = max_holding
        self.rebalance_freq = rebalance_freq

    def run(
        self,
        price_data: dict[str, pd.DataFrame],
        signal: pd.Series,
        benchmark: Optional[pd.Series] = None,
    ) -> BacktestResult:
        """執行回測。

        Parameters
        ----------
        price_data : dict[str, pd.DataFrame]
            股價資料（需含 'Close' 欄位）
        signal : pd.Series
            MultiIndex (ticker, date) → Z-score 綜合信號
        benchmark : pd.Series, optional
            基準指數的每日報酬率

        Returns
        -------
        BacktestResult
        """
        # 建立價格矩陣
        price_matrix = self._build_price_matrix(price_data)
        signal_matrix = signal.unstack(level=0)  # date × ticker

        # 對齊日期
        common_dates = price_matrix.index.intersection(signal_matrix.index)
        if common_dates.empty:
            raise ValueError("價格資料與信號資料沒有重疊的日期區間")

        price_matrix = price_matrix.loc[common_dates]
        signal_matrix = signal_matrix.loc[common_dates]

        # 決定再平衡日
        rebalance_dates = self._get_rebalance_dates(common_dates)

        # 模擬（以股數追蹤持倉，每日依收盤價重新估值）
        n = len(common_dates)
        n_stocks = len(price_matrix.columns)
        tickers = price_matrix.columns.tolist()

        cash = self.initial_capital
        shares = np.zeros(n_stocks)  # 各股持有股數
        cost_basis = np.zeros(n_stocks)   # 各股成本均價
        buy_dates = [None] * n_stocks     # 最後買入日期
        portfolio_values = np.full(n, np.nan)
        positions_log = np.zeros((n, n_stocks))
        trades_log = []

        for i, date in enumerate(common_dates):
            prices = price_matrix.iloc[i].values  # 當日收盤價

            # 持倉市值 = 股數 × 當日股價
            position_value = shares * prices
            total_value = cash + position_value.sum()
            portfolio_values[i] = total_value

            # 持倉權重
            if total_value > 0:
                positions_log[i] = position_value / total_value

            # 再平衡：賣光舊倉 → 付費 → 買新倉
            if date in rebalance_dates and i > 0:
                sig = signal_matrix.loc[date].dropna()
                selected = sig.nlargest(self.max_holding)
                n_selected = len(selected)

                if n_selected == 0 or total_value <= 0:
                    continue

                # 賣出所有舊倉 — 以昨日收盤價計算
                prev_prices = price_matrix.iloc[i - 1].values
                sell_value = (shares * prev_prices).sum()
                sell_cost = sell_value * self.transaction_cost

                # 重新計算當日資產（用當日價格，扣除賣出成本）
                cash = cash + sell_value - sell_cost
                shares = np.zeros(n_stocks)
                cost_basis = np.zeros(n_stocks)
                buy_dates = [None] * n_stocks

                # 買入新倉 — 等權重
                target_per_stock = total_value / n_selected
                for ticker in selected.index:
                    idx = tickers.index(ticker)
                    price = prices[idx]
                    if pd.isna(price) or price <= 0:
                        continue
                    buy_value = target_per_stock
                    slippage_cost = buy_value * self.slippage
                    net_buy = buy_value - slippage_cost
                    shares[idx] = net_buy / price
                    cost_basis[idx] = price
                    buy_dates[idx] = date
                    cash -= buy_value

                    trades_log.append({
                        "date": date,
                        "ticker": ticker,
                        "action": "rebalance",
                        "weight": 1.0 / n_selected,
                        "price": float(price),
                    })

        # 整理結果
        portfolio_value = pd.Series(portfolio_values, index=common_dates, name="portfolio_value")
        positions = pd.DataFrame(positions_log, index=common_dates, columns=price_matrix.columns)
        trades = pd.DataFrame(trades_log) if trades_log else pd.DataFrame()

        # 計算每日報酬率
        ret_series = portfolio_value.pct_change().fillna(0)

        # 整理最新持倉明細
        last_prices = price_matrix.iloc[-1].values
        holdings_detail = []
        for i, t in enumerate(tickers):
            if shares[i] > 0.001 and cost_basis[i] > 0:
                market_value = shares[i] * last_prices[i]
                cost_value = shares[i] * cost_basis[i]
                holdings_detail.append({
                    "ticker": t,
                    "shares": round(shares[i], 2),
                    "buy_date": str(buy_dates[i].date()) if buy_dates[i] is not None else "",
                    "buy_price": round(float(cost_basis[i]), 2),
                    "current_price": round(float(last_prices[i]), 2),
                    "cost": round(float(cost_value), 2),
                    "market_value": round(float(market_value), 2),
                    "pnl": round(float(market_value - cost_value), 2),
                    "pnl_pct": round(float((market_value - cost_value) / cost_value * 100), 2),
                })

        # 計算績效指標
        metrics = self._calc_metrics(ret_series, portfolio_value)

        if benchmark is not None:
            benchmark_aligned = benchmark.reindex(common_dates).fillna(0)
        else:
            benchmark_aligned = None

        return BacktestResult(
            portfolio_value=portfolio_value,
            positions=positions,
            trades=trades,
            returns=ret_series,
            benchmark_returns=benchmark_aligned,
            metrics=metrics,
            holdings_detail=holdings_detail,
        )

    def _build_price_matrix(self, price_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """將 dict 價量資料轉為 date × ticker 矩陣。"""
        dfs = {}
        for ticker, df in price_data.items():
            if "Close" in df.columns:
                dfs[ticker] = df["Close"].rename(ticker)
        if not dfs:
            raise ValueError("無有效的價格資料")
        return pd.concat(dfs.values(), axis=1).sort_index().ffill().bfill()

    def _get_rebalance_dates(self, dates: pd.DatetimeIndex) -> set[pd.Timestamp]:
        """根據頻率計算再平衡日期。"""
        if self.rebalance_freq == "monthly":
            # 每月第一個交易日
            months = dates.to_period("M").unique()
            result = set()
            for m in months:
                mask = dates.to_period("M") == m
                if mask.any():
                    result.add(dates[mask][0])
            return result
        elif self.rebalance_freq == "biweekly":
            # 每兩週第一個交易日
            week_offset = (dates - dates[0]).days // 14
            seen = set()
            result = set()
            for i, d in enumerate(dates):
                key = week_offset[i]
                if key not in seen:
                    seen.add(key)
                    result.add(d)
            return result
        elif self.rebalance_freq == "quarterly":
            quarters = dates.to_period("Q").unique()
            result = set()
            for q in quarters:
                mask = dates.to_period("Q") == q
                if mask.any():
                    result.add(dates[mask][0])
            return result
        elif self.rebalance_freq == "yearly":
            years = dates.to_period("Y").unique()
            result = set()
            for y in years:
                mask = dates.to_period("Y") == y
                if mask.any():
                    result.add(dates[mask][0])
            return result
        return set(dates)

    @staticmethod
    def _calc_metrics(returns: pd.Series, value: pd.Series) -> dict:
        """計算基本績效指標。"""
        daily_rf = 0.05 / 252
        excess = returns - daily_rf
        total_return = (value.iloc[-1] / value.iloc[0]) - 1
        n_years = len(returns) / 252
        ann_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

        ann_vol = returns.std() * np.sqrt(252)
        sharpe = (ann_return - 0.05) / ann_vol if ann_vol > 0 else 0

        cummax = value.cummax()
        drawdown = (value - cummax) / cummax
        max_dd = drawdown.min()

        calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

        # 勝率
        win_rate = (returns > 0).sum() / max(len(returns), 1)

        return {
            "total_return": float(total_return),
            "ann_return": float(ann_return),
            "ann_volatility": float(ann_vol),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_dd),
            "calmar_ratio": float(calmar),
            "win_rate": float(win_rate),
            "num_trading_days": len(returns),
            "start_date": str(value.index[0].date()),
            "end_date": str(value.index[-1].date()),
        }
