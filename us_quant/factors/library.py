"""內建因子庫 — 7 個核心因子"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .base import FactorBase, FactorResult


class MomentumFactor(FactorBase):
    """12 個月動能因子（排除最近 1 個月，避免短期反轉）。"""

    def __init__(self, lookback: int = 252, skip: int = 21):
        super().__init__(
            name="momentum",
            description=f"{lookback}日動能（排除最近 {skip} 日）",
        )
        self.lookback = lookback
        self.skip = skip

    def compute(self, price_data: dict[str, pd.DataFrame], **params) -> FactorResult:
        lookback = params.get("lookback", self.lookback)
        skip = params.get("skip", self.skip)
        records: list[dict] = []

        for ticker, df in price_data.items():
            close = df["Close"].sort_index()
            if len(close) < lookback + skip:
                continue
            # 每個交易日計算過去 lookback 日的回報（排除最近 skip 日）
            ret = close.pct_change(periods=lookback).shift(skip)
            for dt, val in ret.items():
                if pd.notna(val):
                    records.append({"ticker": ticker, "date": dt, "value": val})

        if not records:
            return FactorResult(name=self.name, scores=pd.Series(dtype=float), description=self.description)

        raw = pd.DataFrame(records).set_index(["ticker", "date"]).squeeze("columns")
        scores = self._zscore(raw)
        return FactorResult(name=self.name, scores=scores, raw=raw, description=self.description)


class ValueFactor(FactorBase):
    """綜合價值因子：低 PE、低 PB、低 PS 的等權平均。"""

    def __init__(self):
        super().__init__(name="value", description="低PE × 低PB × 低PS 綜合價值分數")

    def compute(self, price_data, fundamentals: Optional[pd.DataFrame] = None, **params) -> FactorResult:
        if fundamentals is None:
            raise ValueError("ValueFactor 需要 fundamentals DataFrame")

        metrics = ["pe_ratio", "pb_ratio", "ps_ratio"]
        # 取倒數並標準化（數值越低分數越高）
        combined = pd.Series(dtype=float)
        for m in metrics:
            if m not in fundamentals.columns:
                continue
            # 只取正值的倒數
            s = fundamentals[m].copy()
            s = s[s > 0]
            if s.empty:
                continue
            inv = 1.0 / s
            # 橫截面 Z-score，反轉（數值低 = 分數高）
            z = -(self._zscore(inv))
            combined = pd.concat([combined, z])

        # 每個 ticker 取平均值
        mean_scores = combined.groupby(level=0).mean()

        # 展開為 (ticker, date) 格式 — 使用每個 ticker 最近交易日
        dates = {}
        for ticker, df in price_data.items():
            if not df.empty:
                dates[ticker] = df.index[-1]

        idx = pd.MultiIndex.from_tuples(
            [(t, dates[t]) for t in mean_scores.index if t in dates],
            names=["ticker", "date"],
        )
        scores = pd.Series(mean_scores.loc[[t for t, _ in idx]].values, index=idx)
        return FactorResult(name=self.name, scores=scores, description=self.description)


class QualityFactor(FactorBase):
    """品質因子：高 ROE × 低負債比 × 高毛利率。"""

    def __init__(self):
        super().__init__(name="quality", description="高ROE × 低負債 × 高毛利率")

    def compute(self, price_data, fundamentals: Optional[pd.DataFrame] = None, **params) -> FactorResult:
        if fundamentals is None:
            raise ValueError("QualityFactor 需要 fundamentals DataFrame")

        quality_cols = ["roe", "gross_margin"]
        scores_list = []

        for col in quality_cols:
            if col not in fundamentals.columns:
                continue
            s = fundamentals[col].dropna()
            if s.empty:
                continue
            scores_list.append(self._zscore(s))

        # 負債比越低越好，取反
        if "debt_to_equity" in fundamentals.columns:
            dte = fundamentals["debt_to_equity"].dropna()
            if not dte.empty:
                scores_list.append(-self._zscore(dte))

        if not scores_list:
            return FactorResult(name=self.name, scores=pd.Series(dtype=float), description=self.description)

        combined = pd.concat(scores_list).groupby(level=0).mean()

        dates = {}
        for ticker, df in price_data.items():
            if not df.empty:
                dates[ticker] = df.index[-1]

        idx = pd.MultiIndex.from_tuples(
            [(t, dates[t]) for t in combined.index if t in dates],
            names=["ticker", "date"],
        )
        scores = pd.Series(combined.loc[[t for t, _ in idx]].values, index=idx)
        return FactorResult(name=self.name, scores=scores, description=self.description)


class LowVolFactor(FactorBase):
    """低波動因子：過去 1 年的日報酬標準差。"""

    def __init__(self, lookback: int = 252):
        super().__init__(name="low_vol", description=f"過去{lookback}日波動度（越低分數越高）")
        self.lookback = lookback

    def compute(self, price_data: dict[str, pd.DataFrame], **params) -> FactorResult:
        lookback = params.get("lookback", self.lookback)
        records: list[dict] = []

        for ticker, df in price_data.items():
            close = df["Close"].sort_index()
            if len(close) < lookback + 20:
                continue
            daily_ret = close.pct_change()
            vol = daily_ret.rolling(lookback).std()
            for dt, v in vol.items():
                if pd.notna(v):
                    records.append({"ticker": ticker, "date": dt, "value": v})

        if not records:
            return FactorResult(name=self.name, scores=pd.Series(dtype=float), description=self.description)

        raw = pd.DataFrame(records).set_index(["ticker", "date"]).squeeze("columns")
        # 波動越低分數越高 → 取反
        scores = -self._zscore(raw)
        return FactorResult(name=self.name, scores=scores, raw=raw, description=self.description)


class SizeFactor(FactorBase):
    """規模因子：市值越小的公司分數越高（小市值溢酬）。"""

    def __init__(self):
        super().__init__(name="size", description="市值反比（小公司溢酬）")

    def compute(self, price_data, fundamentals: Optional[pd.DataFrame] = None, **params) -> FactorResult:
        if fundamentals is None or "market_cap" not in fundamentals.columns:
            # Render / 雲端環境 yfinance 偶爾缺欄位，回傳空分數而非 raise
            return FactorResult(name=self.name, scores=pd.Series(dtype=float), description=self.description)

        mc = fundamentals["market_cap"].dropna()
        if mc.empty:
            return FactorResult(name=self.name, scores=pd.Series(dtype=float), description=self.description)

        # 市值取 log 後反轉（小公司高分）
        log_mc = np.log(mc)
        scores_raw = -self._zscore(log_mc)

        dates = {}
        for ticker, df in price_data.items():
            if not df.empty:
                dates[ticker] = df.index[-1]

        idx = pd.MultiIndex.from_tuples(
            [(t, dates[t]) for t in scores_raw.index if t in dates],
            names=["ticker", "date"],
        )
        scores = pd.Series(scores_raw.loc[[t for t, _ in idx]].values, index=idx)
        return FactorResult(name=self.name, scores=scores, description=self.description)


class DivYieldFactor(FactorBase):
    """股息率因子：高股息率。"""

    def __init__(self):
        super().__init__(name="div_yield", description="股息率（越高分數越高）")

    def compute(self, price_data, fundamentals: Optional[pd.DataFrame] = None, **params) -> FactorResult:
        if fundamentals is None or "dividend_yield" not in fundamentals.columns:
            return FactorResult(name=self.name, scores=pd.Series(dtype=float), description=self.description)

        dy = fundamentals["dividend_yield"].dropna()
        if dy.empty:
            return FactorResult(name=self.name, scores=pd.Series(dtype=float), description=self.description)

        scores_raw = self._zscore(dy)

        dates = {}
        for ticker, df in price_data.items():
            if not df.empty:
                dates[ticker] = df.index[-1]

        idx = pd.MultiIndex.from_tuples(
            [(t, dates[t]) for t in scores_raw.index if t in dates],
            names=["ticker", "date"],
        )
        scores = pd.Series(scores_raw.loc[[t for t, _ in idx]].values, index=idx)
        return FactorResult(name=self.name, scores=scores, description=self.description)


class RevenueGrowthFactor(FactorBase):
    """營收成長因子：年營收成長率。"""

    def __init__(self):
        super().__init__(name="revenue_growth", description="營收年成長率")

    def compute(self, price_data, fundamentals: Optional[pd.DataFrame] = None, **params) -> FactorResult:
        if fundamentals is None or "revenue_growth" not in fundamentals.columns:
            return FactorResult(name=self.name, scores=pd.Series(dtype=float), description=self.description)

        rg = fundamentals["revenue_growth"].dropna()
        if rg.empty:
            return FactorResult(name=self.name, scores=pd.Series(dtype=float), description=self.description)

        scores_raw = self._zscore(rg)

        dates = {}
        for ticker, df in price_data.items():
            if not df.empty:
                dates[ticker] = df.index[-1]

        idx = pd.MultiIndex.from_tuples(
            [(t, dates[t]) for t in scores_raw.index if t in dates],
            names=["ticker", "date"],
        )
        scores = pd.Series(scores_raw.loc[[t for t, _ in idx]].values, index=idx)
        return FactorResult(name=self.name, scores=scores, description=self.description)


# ── 新增因子 ──────────────────────────────────────────


class IndustryMomentumFactor(FactorBase):
    """產業輪動因子：強勢產業中的股票獲得加分。

    每個再平衡日計算各產業過去 3 個月的累積報酬中位數，
    身處強勢產業的股票獲得正向調整，弱勢產業則扣分。
    """

    def __init__(self, lookback: int = 63):
        super().__init__(name="industry_momentum", description=f"產業輪動動能（{lookback}日）")
        self.lookback = lookback

    def compute(self, price_data: dict[str, pd.DataFrame],
                sector_map: Optional[pd.DataFrame] = None, **params) -> FactorResult:
        if sector_map is None or sector_map.empty:
            raise ValueError("IndustryMomentumFactor 需要 sector_map DataFrame（index=ticker, column=sector）")

        lookback = params.get("lookback", self.lookback)
        records: list[dict] = []

        # 每個 ticker 計算其過去報酬
        individual_rets: dict[str, pd.Series] = {}
        for ticker, df in price_data.items():
            close = df["Close"].sort_index()
            if len(close) < lookback:
                continue
            ret = close.pct_change(periods=lookback)
            individual_rets[ticker] = ret

        if not individual_rets:
            return FactorResult(name=self.name, scores=pd.Series(dtype=float), description=self.description)

        # 建立 ticker → sector 對應
        sector_of: dict[str, str] = {}
        for t in individual_rets:
            if t in sector_map.index and pd.notna(sector_map.loc[t, "sector"]):
                sector_of[t] = sector_map.loc[t, "sector"]

        # 對每個日期，計算各產業的動能中位數
        # 先把所有 ticker 的報酬合併為 date × ticker
        ret_df = pd.DataFrame(individual_rets)

        for date in ret_df.index:
            day_data = ret_df.loc[date].dropna()
            if day_data.empty:
                continue
            # 計算每個產業的中位數報酬
            sector_scores: dict[str, list[float]] = {}
            for t, ret_val in day_data.items():
                sec = sector_of.get(t)
                if sec is None:
                    continue
                sector_scores.setdefault(sec, []).append(ret_val)

            if not sector_scores:
                continue

            # 每個產業的中位數動能
            sector_medians = {sec: np.median(vals) for sec, vals in sector_scores.items()}
            # 產業中位數的 Z-score
            sec_series = pd.Series(sector_medians)
            sec_z = self._zscore(sec_series)

            # 分配回個股：個股分數 = 所屬產業的 Z-score
            for t, ret_val in day_data.items():
                sec = sector_of.get(t)
                if sec is None or sec not in sec_z.index:
                    continue
                records.append({"ticker": t, "date": date, "value": sec_z[sec]})

        if not records:
            return FactorResult(name=self.name, scores=pd.Series(dtype=float), description=self.description)

        raw = pd.DataFrame(records).set_index(["ticker", "date"]).squeeze("columns")
        scores = self._zscore(raw)
        return FactorResult(name=self.name, scores=scores, raw=raw, description=self.description)


class FlowFactor(FactorBase):
    """資金流向因子：以 Money Flow Index (MFI) 衡量買賣壓力。

    MFI(14) 越高代表越強的買入資金流入。
    搭配 OBV 趨勢確認。
    """

    def __init__(self, period: int = 14):
        super().__init__(name="flow", description=f"資金流向 MFI({period})")
        self.period = period

    def _compute_mfi(self, df: pd.DataFrame, period: int) -> pd.Series:
        """計算單一股票的 Money Flow Index。"""
        tp = (df["High"] + df["Low"] + df["Close"]) / 3.0  # Typical Price
        raw_mf = tp * df["Volume"]

        mf_sign = tp.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        pos_mf = (raw_mf * mf_sign.clip(lower=0)).rolling(period).sum()
        neg_mf = (raw_mf * (-mf_sign).clip(lower=0)).rolling(period).sum()

        mf_ratio = pos_mf / neg_mf.replace(0, np.nan)
        mfi = 100.0 - (100.0 / (1.0 + mf_ratio))
        return mfi

    def compute(self, price_data: dict[str, pd.DataFrame], **params) -> FactorResult:
        period = params.get("period", self.period)
        records: list[dict] = []

        for ticker, df in price_data.items():
            required = {"High", "Low", "Close", "Volume"}
            if not required.issubset(df.columns):
                continue
            if len(df) < period + 10:
                continue
            mfi = self._compute_mfi(df.sort_index(), period)
            for dt, v in mfi.items():
                if pd.notna(v):
                    records.append({"ticker": ticker, "date": dt, "value": v})

        if not records:
            return FactorResult(name=self.name, scores=pd.Series(dtype=float), description=self.description)

        raw = pd.DataFrame(records).set_index(["ticker", "date"]).squeeze("columns")
        scores = self._zscore(raw)
        return FactorResult(name=self.name, scores=scores, raw=raw, description=self.description)


class FXExposureFactor(FactorBase):
    """匯率曝險因子：美元強弱對海外營收佔比較高公司的影響。

    從 yfinance 抓 DXY（美元指數），搭配各公司的海外營收佔比。
    DXY 上漲 = 美元走強 → 海外營收佔比高者受壓 → 分數降低。
    使用過去 3 個月 DXY 變化作爲信號。
    """

    def __init__(self, lookback: int = 63):
        super().__init__(name="fx_exposure", description=f"匯率曝險（{lookback}日DXY變化 × 海外營收佔比）")
        self.lookback = lookback

    def compute(self, price_data: dict[str, pd.DataFrame],
                dxy: Optional[pd.Series] = None,
                sector_map: Optional[pd.DataFrame] = None, **params) -> FactorResult:
        if dxy is None or dxy.empty:
            raise ValueError("FXExposureFactor 需要 dxy Series（index=date, values=DXY close）")
        if sector_map is None or sector_map.empty:
            raise ValueError("FXExposureFactor 需要 sector_map DataFrame（含 foreign_revenue_pct）")

        lookback = params.get("lookback", self.lookback)

        # DXY 變化率（回看 N 日）
        dxy_chg = dxy.pct_change(periods=lookback)
        # 取反：DXY 漲 = 對高海外營收公司不利
        dxy_signal = -dxy_chg
        dxy_signal = dxy_signal.dropna()
        if dxy_signal.empty:
            return FactorResult(name=self.name, scores=pd.Series(dtype=float), description=self.description)

        # 海外營收佔比（正規化為 Z-score）
        fr_pct = sector_map["foreign_revenue_pct"].dropna()
        if fr_pct.empty:
            return FactorResult(name=self.name, scores=pd.Series(dtype=float), description=self.description)
        fr_z = self._zscore(fr_pct)

        # 使用 merge 而非迭代：把所有 ticker 的日期展開，跟 DXY 信號合併
        all_points: list[dict] = []
        for ticker, df in price_data.items():
            if ticker not in fr_z.index:
                continue
            if df.empty:
                continue
            dates = df.sort_index().index
            for dt in dates:
                all_points.append({"ticker": ticker, "date": dt})

        if not all_points:
            return FactorResult(name=self.name, scores=pd.Series(dtype=float), description=self.description)

        points_df = pd.DataFrame(all_points)
        # 將日期轉為統一格式
        points_df["date"] = pd.to_datetime(points_df["date"])

        # 跟 DXY 信號合併
        dxy_df = dxy_signal.reset_index()
        dxy_df.columns = ["date", "dxy_signal"]
        dxy_df["date"] = pd.to_datetime(dxy_df["date"])

        merged = points_df.merge(dxy_df, on="date", how="inner")
        if merged.empty:
            return FactorResult(name=self.name, scores=pd.Series(dtype=float), description=self.description)

        # 加入海外營收權重
        merged["value"] = merged["ticker"].map(fr_z) * merged["dxy_signal"]

        records_val = merged.loc[merged["value"].notna(), ["ticker", "date", "value"]]
        if records_val.empty:
            return FactorResult(name=self.name, scores=pd.Series(dtype=float), description=self.description)

        raw = records_val.set_index(["ticker", "date"]).squeeze("columns")
        scores = self._zscore(raw)
        return FactorResult(name=self.name, scores=scores, raw=raw, description=self.description)


class AIThemeFactor(FactorBase):
    """AI 主题因子：AI 产业链曝光度 × AI 板块相对强度（vs SPY）。

    追蹤純 AI 籃子相對大盤的超額收益，作為 AI 主題熱度的代理變量。
    個股 AI 曝光度越高，在 AI 主題火熱時得分越高。
    """

    # 個股 AI 產業鏈曝光度 (0-100)，基於業務結構分析
    AI_EXPOSURE: dict[str, float] = {
        # ── 芯片/硬件層 ──
        "NVDA": 100, "AMD": 95, "AVGO": 90, "INTC": 70, "TSM": 85,
        # ── 雲端/基礎設施層 ──
        "MSFT": 90, "GOOGL": 85, "AMZN": 80, "ORCL": 65, "IBM": 55,
        # ── AI 應用/平台層 ──
        "META": 80, "ADBE": 70, "CRM": 70, "AAPL": 60, "SAP": 50,
        # ── 間接受益 ──
        "CSCO": 40, "ACN": 40, "NFLX": 35, "DIS": 30, "BA": 20,
        "GE": 20, "CAT": 15, "TMO": 15, "TM": 15,
        # ── 金融/支付 AI 應用 ──
        "JPM": 15, "V": 15, "MA": 15, "GS": 15,
        # ── 傳統行業（低 AI 曝光） ──
        "JNJ": 10, "ABBV": 10, "LLY": 10, "UNH": 10,
        "PG": 5, "KO": 5, "PEP": 5, "HD": 5, "MCD": 5, "NKE": 5,
        "WMT": 5, "XOM": 0, "CVX": 0,
    }

    AI_BASKET = ["NVDA", "AMD", "AVGO", "MSFT", "GOOGL", "META", "AMZN", "CRM"]

    def __init__(self, lookback: int = 21):
        super().__init__(name="ai_industry", description=f"AI 主题强度（{lookback}日相對SPY動量）")
        self.lookback = lookback

    def compute(self, price_data: dict[str, pd.DataFrame], **params) -> FactorResult:
        lookback = params.get("lookback", self.lookback)

        # 建立 AI 籃子等權指數
        basket_prices = []
        for t in self.AI_BASKET:
            if t in price_data and not price_data[t].empty:
                s = price_data[t]["Close"].sort_index().rename(t)
                basket_prices.append(s)
        if len(basket_prices) < 3:
            return FactorResult(name=self.name, scores=pd.Series(dtype=float), description=self.description)

        ai_index = pd.concat(basket_prices, axis=1).ffill().mean(axis=1)
        ai_ret = ai_index.pct_change()

        # SPY 基準
        spy = params.get("benchmark")
        if spy is None:
            spy_ret = pd.Series(dtype=float)
        elif isinstance(spy, pd.DataFrame):
            spy_close = spy["Close"].sort_index()
            spy_ret = spy_close.pct_change()
        elif isinstance(spy, pd.Series):
            spy_ret = spy.sort_index().pct_change()

        # AI 相對 SPY 的超額收益
        if not spy_ret.empty:
            common_idx = ai_ret.index.intersection(spy_ret.index)
            relative = (ai_ret.loc[common_idx] - spy_ret.loc[common_idx]).dropna()
        else:
            relative = ai_ret.dropna()

        if relative.empty:
            return FactorResult(name=self.name, scores=pd.Series(dtype=float), description=self.description)

        # 累積相對動能（回看 N 日）
        cum_rel = relative.rolling(lookback).sum()

        records: list[dict] = []
        for date in cum_rel.dropna().index:
            strength = cum_rel[date]
            if pd.isna(strength):
                continue
            for ticker in price_data:
                exp = self.AI_EXPOSURE.get(ticker, 0)
                if exp == 0:
                    continue
                if ticker not in price_data or price_data[ticker].empty:
                    continue
                if date not in price_data[ticker].index:
                    continue
                records.append({"ticker": ticker, "date": date, "value": exp * strength})

        if not records:
            return FactorResult(name=self.name, scores=pd.Series(dtype=float), description=self.description)

        raw = pd.DataFrame(records).set_index(["ticker", "date"]).squeeze("columns")
        scores = self._zscore(raw)
        return FactorResult(name=self.name, scores=scores, raw=raw, description=self.description)


# ── 工廠函數 ──

FACTOR_REGISTRY: dict[str, type[FactorBase]] = {
    "momentum": MomentumFactor,
    "value": ValueFactor,
    "quality": QualityFactor,
    "low_vol": LowVolFactor,
    "size": SizeFactor,
    "div_yield": DivYieldFactor,
    "revenue_growth": RevenueGrowthFactor,
    "industry_momentum": IndustryMomentumFactor,
    "flow": FlowFactor,
    "fx_exposure": FXExposureFactor,
    "ai_industry": AIThemeFactor,
}


def get_factor(name: str, **kwargs) -> FactorBase:
    """根據名稱取得因子實例。"""
    cls = FACTOR_REGISTRY.get(name)
    if cls is None:
        raise KeyError(f"未知因子: {name}，可選: {list(FACTOR_REGISTRY)}")
    return cls(**kwargs)
