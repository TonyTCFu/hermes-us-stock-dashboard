"""績效指標：詳盡的投資組合績效分析"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class PerformanceAnalyzer:
    """投資組合績效分析工具。

    支援：
    - 年化報酬 / 波動
    - Sharpe / Sortino / Calmar 比率
    - 最大回撤與回撤期間
    - Alpha / Beta / 資訊比率
    - 滾動指標
    """

    def __init__(self, risk_free_rate: float = 0.05, trading_days: int = 252):
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days

    def analyze(self, portfolio_returns: pd.Series, benchmark_returns: Optional[pd.Series] = None) -> dict:
        """全面分析投資組合績效。

        Parameters
        ----------
        portfolio_returns : pd.Series
            投資組合每日報酬率（以日期為索引）
        benchmark_returns : pd.Series, optional
            基準指數每日報酬率

        Returns
        -------
        dict
            各項績效指標
        """
        ret = portfolio_returns.dropna()
        if len(ret) < 5:
            return {"error": "數據不足"}

        daily_rf = self.risk_free_rate / self.trading_days
        excess = ret - daily_rf

        # ── 基本指標 ──
        n_years = len(ret) / self.trading_days
        cum_ret = (1 + ret).prod() - 1
        ann_ret = (1 + cum_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
        ann_vol = ret.std() * np.sqrt(self.trading_days)

        sharpe = (ann_ret - self.risk_free_rate) / ann_vol if ann_vol > 0 else 0

        # ── Sortino ──
        downside = ret[ret < 0].std() * np.sqrt(self.trading_days)
        sortino = (ann_ret - self.risk_free_rate) / downside if downside > 0 else 0

        # ── 回撤 ──
        cum_value = (1 + ret).cumprod()
        running_max = cum_value.cummax()
        drawdown = (cum_value - running_max) / running_max
        max_dd = drawdown.min()
        max_dd_start = drawdown.idxmin() if not drawdown.empty else None
        # 回撤持續時間
        dd_duration = 0
        if not drawdown.empty:
            underwater = drawdown < 0
            dd_duration = underwater.sum()

        # ── Calmar ──
        calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.inf

        # ── 勝率 ──
        win_rate = (ret > 0).sum() / len(ret)
        avg_win = ret[ret > 0].mean() if (ret > 0).any() else 0
        avg_loss = ret[ret < 0].mean() if (ret < 0).any() else 0
        profit_factor = abs((ret[ret > 0].sum()) / (ret[ret < 0].sum())) if (ret < 0).sum() != 0 else np.inf

        # ── Alpha / Beta（如果有基準） ──
        alpha = beta = info_ratio = None
        if benchmark_returns is not None:
            bench = benchmark_returns.reindex(ret.index).dropna()
            ret_aligned = ret.reindex(bench.index)
            common = ret_aligned.notna() & bench.notna()
            r = ret_aligned[common]
            b = bench[common]

            if len(r) > 30:
                cov = np.cov(r, b)
                beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0
                alpha = (r.mean() - b.mean() * beta) * self.trading_days
                tracking_error = (r - b).std() * np.sqrt(self.trading_days)
                info_ratio = (r.mean() - b.mean()) / (r - b).std() if tracking_error > 0 else 0

        # ── 滾動夏普 ──
        rolling_sharpe_60d = None
        if len(ret) >= 63:
            roll_ret = ret.rolling(63).mean() * self.trading_days
            roll_vol = ret.rolling(63).std() * np.sqrt(self.trading_days)
            rolling_sharpe_60d = ((roll_ret - self.risk_free_rate) / roll_vol).dropna()
            rolling_sharpe_60d = rolling_sharpe_60d.to_dict()

        return {
            "total_return": float(cum_ret),
            "annualized_return": float(ann_ret),
            "annualized_volatility": float(ann_vol),
            "sharpe_ratio": round(sharpe, 3),
            "sortino_ratio": round(sortino, 3),
            "calmar_ratio": round(calmar, 3),
            "max_drawdown": float(max_dd),
            "max_drawdown_date": str(max_dd_start.date()) if max_dd_start is not None else "N/A",
            "drawdown_duration_days": int(dd_duration),
            "win_rate": float(win_rate),
            "avg_win": float(avg_win) if avg_win != 0 else 0,
            "avg_loss": float(avg_loss) if avg_loss != 0 else 0,
            "profit_factor": round(profit_factor, 3),
            "alpha": round(alpha, 4) if alpha is not None else None,
            "beta": round(beta, 4) if beta is not None else None,
            "information_ratio": round(info_ratio, 3) if info_ratio is not None else None,
            "num_trading_days": len(ret),
            "date_range": f"{ret.index[0].date()} ~ {ret.index[-1].date()}",
        }

    def summary_table(self, results: dict) -> pd.DataFrame:
        """將績效指標轉為可視化 DataFrame。"""
        categories = {
            "報酬": ["total_return", "annualized_return"],
            "風險": ["annualized_volatility", "max_drawdown", "max_drawdown_date", "drawdown_duration_days"],
            "風險調整後報酬": ["sharpe_ratio", "sortino_ratio", "calmar_ratio", "profit_factor"],
            "交易統計": ["win_rate", "avg_win", "avg_loss", "num_trading_days"],
            "市場比較": ["alpha", "beta", "information_ratio"],
        }
        labels = {
            "total_return": "累積報酬",
            "annualized_return": "年化報酬",
            "annualized_volatility": "年化波動",
            "max_drawdown": "最大回撤",
            "max_drawdown_date": "回撤發生日",
            "drawdown_duration_days": "回撤天數",
            "sharpe_ratio": "夏普比率",
            "sortino_ratio": "索提諾比率",
            "calmar_ratio": "卡爾瑪比率",
            "profit_factor": "獲利因子",
            "win_rate": "勝率",
            "avg_win": "平均獲利",
            "avg_loss": "平均虧損",
            "num_trading_days": "交易日數",
            "alpha": "Alpha",
            "beta": "Beta",
            "information_ratio": "資訊比率",
        }
        rows = []
        for cat, keys in categories.items():
            for k in keys:
                v = results.get(k)
                if v is not None and v != "N/A" and v != "":
                    rows.append({"類別": cat, "指標": labels.get(k, k), "數值": v})
        return pd.DataFrame(rows)
