"""信號合成：將多個因子加權合併為單一交易信號"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from ..factors import FactorResult


class SignalCombiner:
    """將多個 FactorResult 加權平均為綜合信號。

    支援 Z-score 加權平均與排名加權平均兩種模式。
    """

    def __init__(self, weights: dict[str, float]):
        """
        Parameters
        ----------
        weights : dict[str, float]
            { 因子名稱: 權重 }，如 {"momentum": 0.20, "value": 0.15}
        """
        total = sum(weights.values())
        self.weights = {k: v / total for k, v in weights.items()}

    def combine(self, factor_results: list[FactorResult]) -> pd.Series:
        """合併多個因子為綜合信號。

        Returns
        -------
        pd.Series
            MultiIndex (ticker, date) → 綜合 Z-score
        """
        combined = pd.DataFrame()
        for fr in factor_results:
            w = self.weights.get(fr.name, 0)
            if w == 0 or fr.scores.empty:
                continue
            col = fr.scores.copy()
            col.name = fr.name
            combined = pd.concat([combined, col.to_frame()], axis=1)

        if combined.empty:
            return pd.Series(dtype=float)

        # 等權重填補缺失值（同一 ticker 某因子缺失時用其他因子均值補）
        combined = combined.T.fillna(combined.T.mean()).T

        weights_series = pd.Series(self.weights)
        available = [c for c in combined.columns if c in weights_series.index]
        if not available:
            return pd.Series(dtype=float)

        w = weights_series[available]
        w = w / w.sum()  # 重新正規化

        signal = combined[available] @ w
        return signal.sort_index()


def regime_adjust_weights(signal: pd.Series, vix: pd.Series) -> pd.Series:
    """VIX 市場狀態調整：根據 VIX 水平調整信號強度。

    - VIX > 30 → 降低 50% 信號強度（市場恐慌，減倉）
    - VIX 20-30 → 線性遞減調降
    - VIX < 15 → 正常（無調整）
    - VIX 15-20 → 輕微遞增

    Parameters
    ----------
    signal : pd.Series
        MultiIndex (ticker, date) → Z-score
    vix : pd.Series
        index=date → VIX close

    Returns
    -------
    pd.Series
        調整後的信號
    """
    if vix.empty:
        return signal

    adj = signal.copy()
    dates = signal.index.get_level_values("date").unique()

    for d in dates:
        d_str = d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
        if d_str not in vix.index:
            continue
        v = vix[d_str]

        # 非線性調整因子
        if v > 30:
            factor = 0.50  # 恐慌 → 減半
        elif v > 25:
            factor = 0.65
        elif v > 20:
            factor = max(0.70, 1.0 - (v - 20) / 30)  # 20~30 線性遞減
        elif v < 15:
            factor = 1.0  # 正常
        else:
            factor = 1.0  # 15-20 正常

        mask = adj.index.get_level_values("date") == d
        adj.loc[mask] = adj.loc[mask] * factor

    return adj
