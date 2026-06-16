"""因子基礎類別與結果結構"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class FactorResult:
    """單一因子的計算結果。"""
    name: str
    """因子名稱（如 momentum）"""

    scores: pd.Series
    """標準化後的分數 (Z-score)，index=(ticker, date)，值域 ~[-3, 3]"""

    raw: pd.Series | None = None
    """原始計算值（未標準化）"""

    description: str = ""
    """因子說明"""

    metadata: dict = field(default_factory=dict)
    """額外資訊（如參數設定）"""


class FactorBase(ABC):
    """所有因子的抽象基底類別。"""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description

    @abstractmethod
    def compute(self, price_data: dict[str, pd.DataFrame], **params) -> FactorResult:
        """計算因子分數。

        Parameters
        ----------
        price_data : dict[str, pd.DataFrame]
            ticker → OHLCV DataFrame，來自 fetcher.fetch_price_data()

        Returns
        -------
        FactorResult
            scores 為 MultiIndex Series: (ticker, date) → Z-score
        """
        ...

    @staticmethod
    def _zscore(s: pd.Series | pd.DataFrame) -> pd.Series:
        """橫截面 Z-score 標準化（移除極端值）。"""
        arr = s.values if isinstance(s, (pd.DataFrame, pd.Series)) else s
        lo = float(pd.Series(arr.flatten()).quantile(0.01))
        hi = float(pd.Series(arr.flatten()).quantile(0.99))
        clipped = pd.Series(arr.flatten()).clip(lower=lo, upper=hi)
        std = float(clipped.std())
        if std == 0 or pd.isna(std):
            return pd.Series(0.0, index=s.index)
        result = (clipped - float(clipped.mean())) / std
        if isinstance(s, pd.Series):
            return pd.Series(result.values, index=s.index)
        return pd.Series(result.values, index=s.index)

    @staticmethod
    def _rank(s: pd.Series) -> pd.Series:
        """百分位排名 [0, 1]。"""
        return s.rank(pct=True)
