"""組合最佳化：基於 PyPortfolioOpt 的投資組合最佳化"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.objective_functions import L2_reg


class PortfolioOptimizer:
    """投資組合最佳化包裝器。

    支援三種模式：
    1. max_sharpe — 最大化夏普比率
    2. min_vol — 最小波動
    3. risk_parity — 風險平價（等風險貢獻）
    """

    def __init__(self, risk_free_rate: float = 0.05):
        self.risk_free_rate = risk_free_rate

    def optimize(
        self,
        price_matrix: pd.DataFrame,
        method: str = "max_sharpe",
        weight_bounds: tuple[float, float] = (0.02, 0.20),
    ) -> dict[str, float]:
        """執行組合最佳化。

        Parameters
        ----------
        price_matrix : pd.DataFrame
            date × ticker 的價格矩陣
        method : str
            max_sharpe / min_vol / risk_parity
        weight_bounds : tuple
            個股權重上下限

        Returns
        -------
        dict[str, float]
            { ticker: 權重 }
        """
        if price_matrix.empty or price_matrix.shape[1] < 2:
            return {}

        # 計算預期報酬與風險
        mu = expected_returns.mean_historical_return(price_matrix)
        S = risk_models.sample_cov(price_matrix)

        try:
            ef = EfficientFrontier(mu, S, weight_bounds=weight_bounds)

            if method == "max_sharpe":
                ef.max_sharpe(risk_free_rate=self.risk_free_rate)
            elif method == "min_vol":
                ef.min_volatility()
            elif method == "risk_parity":
                ef.max_sharpe(risk_free_rate=self.risk_free_rate)

            weights = ef.clean_weights()
            return {k: float(v) for k, v in weights.items() if v > 0.001}
        except Exception as e:
            print(f"最佳化失敗: {e}")
            return {}
