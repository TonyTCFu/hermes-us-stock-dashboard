"""
美股量化分析 Dashboard

啟動方式：
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

# 確保專案在 import path 上
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from us_quant.config import (
    STOCK_UNIVERSE,
    FACTOR_WEIGHTS,
    INITIAL_CAPITAL,
    TRANSACTION_COST,
    SLIPPAGE,
    REBALANCE_FREQUENCY,
    RISK_FREE_RATE,
    BENCHMARK_TICKER,
    DB_PATH,
)
from us_quant.data import (
    fetch_price_data, fetch_all_fundamentals, fetch_sector_info,
    fetch_dxy, fetch_vix, ensure_price_data, DataStore,
)
from us_quant.factors import get_factor, FactorResult
from us_quant.signals import SignalCombiner, regime_adjust_weights
from us_quant.backtest import BacktestEngine
from us_quant.reporting import PerformanceAnalyzer
from pypfopt import expected_returns, risk_models, EfficientFrontier
from us_quant.portfolio import PortfolioOptimizer
from us_quant.broker.alpaca import AlpacaBroker

logger = logging.getLogger(__name__)

# ── 頁面設定 ──
st.set_page_config(
    page_title="【Claude Code】美股量化分析平台",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<link rel="icon" type="image/png" sizes="32x32" href="./static/favicon-32.png">
<link rel="icon" type="image/png" sizes="192x192" href="./static/favicon-192.png">
<link rel="apple-touch-icon" sizes="180x180" href="./static/favicon-180.png">
<link rel="manifest" href="./static/manifest.json">
<meta name="theme-color" content="#0f172a">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="Claude Code 量化">
<style>
    .metric-card { background: #1a1a2e; padding: 1rem; border-radius: 0.5rem; border: 1px solid #2d2d4e; }
    .stTabs [data-baseweb="tab-list"] { gap: 1rem; }
    .stTabs [data-baseweb="tab"] { padding: 0.5rem 1rem; }
</style>
""", unsafe_allow_html=True)


# ── 輔助函數 ──

@st.cache_data(ttl=3600)
def fmt_pct(v, default="N/A"):
    """格式化百分比，v 為小數 (如 0.297 → '29.70%')。"""
    if v is None or v == "N/A":
        return default
    return f"{float(v)*100:.2f}%"

def fmt_num(v, default="N/A"):
    if v is None or v == "N/A":
        return default
    return f"{float(v):.2f}"


@st.cache_resource
def get_store() -> DataStore:
    """單例 DataStore（Streamlit 生命週期內只初始化一次）。"""
    return DataStore(DB_PATH)


@st.cache_data(ttl=86400)
def load_data(tickers, start, end):
    """載入股價與基本面資料（增量更新 + SQLite 快取）。

    第一次：下載全部並寫入 SQLite。
    之後：只下載本地缺失的資料，再從 SQLite 讀取。
    """
    store = get_store()
    price_data = ensure_price_data(tickers, store, start=start)
    price_data = {k: v for k, v in price_data.items() if not v.empty}
    fundamentals = fetch_all_fundamentals(list(price_data.keys()))
    store.save_fundamentals(fundamentals)
    return price_data, fundamentals


def compute_signal(price_data, fundamentals, weights, sector_map=None, dxy=None, vix=None, rebalance_date=None):
    """計算綜合因子信號（支援 10 因子 + VIX 調整）。"""
    factor_results = []
    for factor_name, weight in weights.items():
        if weight == 0:
            continue
        try:
            factor = get_factor(factor_name)
            params = {}
            if factor_name in ("value", "quality", "size", "div_yield", "revenue_growth"):
                params["fundamentals"] = fundamentals
            if factor_name == "industry_momentum":
                params["sector_map"] = sector_map
            if factor_name == "fx_exposure":
                params["dxy"] = dxy
                params["sector_map"] = sector_map
            result = factor.compute(price_data, **params)
            if not result.scores.empty:
                factor_results.append(result)
        except Exception as e:
            st.warning(f"因子 {factor_name} 計算失敗: {e}")

    if not factor_results:
        return None, factor_results

    combiner = SignalCombiner(weights)
    signal = combiner.combine(factor_results)

    # VIX 市場狀態調整
    if vix is not None and not vix.empty:
        signal = regime_adjust_weights(signal, vix)

    return signal, factor_results


# ── Sidebar ──

st.sidebar.title("📊 量化分析控制台")

st.sidebar.header("📌 股票池")
ticker_input = st.sidebar.text_area(
    "Ticker 列表（逗號分隔）",
    value=", ".join(STOCK_UNIVERSE[:20]),
    height=100,
)
selected_tickers = [t.strip().upper() for t in ticker_input.replace("\n", ",").split(",") if t.strip()]

st.sidebar.header("📅 時間範圍")
default_start = "2018-01-01"
default_end = datetime.today().strftime("%Y-%m-%d")
start_date = st.sidebar.date_input("開始日期", datetime.strptime(default_start, "%Y-%m-%d"))
end_date = st.sidebar.date_input("結束日期", datetime.today())

st.sidebar.header("⚙️ 因子權重")
factor_weights = {}
for name, default_weight in FACTOR_WEIGHTS.items():
    w = st.sidebar.slider(
        name,
        min_value=0.0, max_value=1.0,
        value=default_weight, step=0.05,
        key=f"fw_{name}",
    )
    factor_weights[name] = w

st.sidebar.header("🔄 回測設定")
rebalance_freq = st.sidebar.selectbox(
    "再平衡頻率",
    ["biweekly", "monthly", "quarterly", "yearly"],
    index=["biweekly", "monthly", "quarterly", "yearly"].index(REBALANCE_FREQUENCY),
)
max_holdings = st.sidebar.slider("最大持倉數量", 5, 50, 20)
tx_cost = st.sidebar.slider("交易成本 (%)", 0.0, 1.0, TRANSACTION_COST * 100, 0.05) / 100
init_capital = st.sidebar.number_input("初始資金 ($)", min_value=10_000, value=int(INITIAL_CAPITAL), step=100_000)

optimize_method = st.sidebar.selectbox(
    "組合最佳化方法",
    ["max_sharpe", "min_vol", "risk_parity"],
    index=0,
)

run_btn = st.sidebar.button("🚀 執行分析", type="primary", use_container_width=True)

# ── Main ──

st.header("📈 【Claude Code】美股量化投資分析平台")
st.caption(
    f"股票池: {len(selected_tickers)} 檔 | "
    f"時間: {start_date} ~ {end_date} | "
    f"再平衡: {rebalance_freq} | "
    f"最大持倉: {max_holdings} 檔"
)

if not run_btn:
    st.info("👈 左側設定好參數後點擊「執行分析」開始")
    st.stop()

# ── 載入資料 ──
with st.spinner("📥 載入股價與基本面資料..."):
    price_data, fundamentals = load_data(selected_tickers, str(start_date), str(end_date))

if not price_data:
    st.error("無法取得股價資料，請檢查 Ticker 是否正確")
    st.stop()

st.success(f"✅ 成功載入 {len(price_data)} 檔股票的資料")

# ── 計算信號 ──
with st.spinner("🧮 計算因子分數（10 因子 + VIX 調整）..."):
    store = get_store()
    # 載入行業分類、DXY、VIX（如有快取則用快取）
    sector_map = store.load_sector_mapping()
    if sector_map.empty:
        sector_map = fetch_sector_info(list(price_data.keys()))
        store.save_sector_mapping(sector_map)

    dxy = store.load_macro("DXY")
    if dxy.empty:
        dxy = fetch_dxy(start=str(start_date))
        store.save_macro("DXY", dxy.to_frame("value").reset_index())

    vix = store.load_macro("VIX")
    if vix.empty:
        vix = fetch_vix(start=str(start_date))
        store.save_macro("VIX", vix.to_frame("value").reset_index())

    signal, factor_results = compute_signal(
        price_data, fundamentals, factor_weights,
        sector_map=sector_map, dxy=dxy, vix=vix,
    )

if signal is None or signal.empty:
    st.error("無法計算綜合信號，請檢查因子設定")
    st.stop()

# ── 獲取基準資料 ──
with st.spinner("📊 載入基準指數 (SPY)..."):
    bench_data = fetch_price_data([BENCHMARK_TICKER], str(start_date), str(end_date))
    if bench_data and BENCHMARK_TICKER in bench_data:
        bench_close = bench_data[BENCHMARK_TICKER]["Close"]
        bench_returns = bench_close.pct_change().fillna(0)
    else:
        bench_returns = None

# ── 回測 ──
with st.spinner("🏃 執行回測..."):
    engine = BacktestEngine(
        initial_capital=init_capital,
        transaction_cost=tx_cost,
        slippage=0.0005,
        max_holding=max_holdings,
        rebalance_freq=rebalance_freq,
    )
    bt_result = engine.run(price_data, signal, benchmark=bench_returns)

# ── 績效分析 ──
analyzer = PerformanceAnalyzer(risk_free_rate=RISK_FREE_RATE)
perf = analyzer.analyze(bt_result.returns, benchmark_returns=bench_returns)

# ── Tab 切換 ──
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["📊 資料概覽", "🧬 因子分析", "📈 回測結果", "📋 持倉分析", "🎯 組合最佳化", "💼 Alpaca 帳戶"]
)

# ════════════════════ Tab 1: 資料概覽 ════════════════════
with tab1:
    st.subheader("📊 股價走勢")
    col1, col2 = st.columns([3, 1])
    with col1:
        plot_ticker = st.selectbox("選擇股票查看走勢", sorted(price_data.keys()), key="plot_ticker")
    with col2:
        ma_period = st.number_input("移動平均線", min_value=5, max_value=200, value=50, step=5)

    fig = go.Figure()
    if plot_ticker in price_data:
        df = price_data[plot_ticker]
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name=f"{plot_ticker} 收盤價"))
        if len(df) > ma_period:
            ma = df["Close"].rolling(ma_period).mean()
            fig.add_trace(go.Scatter(x=df.index, y=ma, mode="lines", name=f"{ma_period}日均線", line=dict(dash="dash")))
    fig.update_layout(height=450, margin=dict(l=0, r=0, t=20, b=0), template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 基本面快照")
    if fundamentals is not None and not fundamentals.empty:
        disp = fundamentals.copy()
        disp.columns = [c.replace("_", " ").title() for c in disp.columns]
        st.dataframe(disp.style.format("{:.2f}"), use_container_width=True)
    else:
        st.info("尚無基本面資料")

# ════════════════════ Tab 2: 因子分析 ════════════════════
with tab2:
    st.subheader("🧬 因子分數分布（10 因子）")

    # 顯示 VIX 狀態
    vix_val = vix.iloc[-1] if vix is not None and not vix.empty else None
    if vix_val is not None:
        vix_label = "⚠️ 恐慌" if vix_val > 30 else ("🟡 警戒" if vix_val > 20 else "🟢 正常")
        st.caption(f"當前 VIX: {vix_val:.2f}（{vix_label}）— 信號已依市場狀態調整")

    # 因子指標：2 列 × N 行
    n_factors = len(factor_results)
    n_cols = 5
    rows_needed = (n_factors + n_cols - 1) // n_cols
    for r in range(rows_needed):
        cols = st.columns(n_cols)
        for c in range(n_cols):
            idx = r * n_cols + c
            if idx < n_factors:
                fr = factor_results[idx]
                with cols[c]:
                    st.metric(
                        f"{fr.name}",
                        f"{fr.scores.mean():.3f}",
                        f"σ={fr.scores.std():.3f}",
                    )

    # 因子相關性矩陣
    st.subheader("因子相關性")
    if len(factor_results) >= 2:
        factor_df = pd.DataFrame({fr.name: fr.scores for fr in factor_results})
        corr = factor_df.corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto")
        fig.update_layout(height=500, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    # 各因子 Top/Bottom 股票（可展開）
    st.subheader("📌 各因子 Top 5 股票")
    with st.expander("展開查看所有因子的 Top 5 股票", expanded=True):
        top_cols = st.columns(min(3, n_factors))
        for i, fr in enumerate(factor_results):
            col_idx = i % 3
            if col_idx == 0 and i > 0:
                top_cols = st.columns(min(3, n_factors - i))
            if i < len(top_cols):
                with top_cols[col_idx]:
                    top = fr.scores.groupby(level="ticker").mean().nlargest(5)
                    st.markdown(f"**{fr.name}**")
                    for ticker, score in top.items():
                        st.write(f"  {ticker}: {score:.3f}")

# ════════════════════ Tab 3: 回測結果 ════════════════════
with tab3:
    st.subheader("📈 權益曲線")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("累積報酬", fmt_pct(perf.get("total_return")))
    col2.metric("年化報酬", fmt_pct(perf.get("annualized_return")))
    col3.metric("年化波動", fmt_pct(perf.get("annualized_volatility")))
    col4.metric("夏普比率", fmt_num(perf.get("sharpe_ratio")))

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("最大回撤", fmt_pct(perf.get("max_drawdown")))
    col6.metric("卡爾瑪比率", fmt_num(perf.get("calmar_ratio")))
    col7.metric("勝率", fmt_pct(perf.get("win_rate")))
    col8.metric("交易日數", perf.get("num_trading_days", "N/A"))

    # 權益曲線圖
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("權益曲線", "回撤曲線"),
        row_heights=[0.65, 0.35],
    )

    # 權益曲線
    norm_value = bt_result.portfolio_value / bt_result.portfolio_value.iloc[0]
    fig.add_trace(go.Scatter(
        x=bt_result.portfolio_value.index, y=bt_result.portfolio_value.values,
        mode="lines", name="策略", line=dict(color="#00b4d8", width=2),
    ), row=1, col=1)

    # 基準（如果有）
    if bench_returns is not None:
        bench_value = (1 + bench_returns).cumprod() * init_capital
        fig.add_trace(go.Scatter(
            x=bench_value.index, y=bench_value.values,
            mode="lines", name=f"{BENCHMARK_TICKER}",
            line=dict(color="#adb5bd", width=1.5, dash="dot"),
        ), row=1, col=1)

    # 回撤曲線
    cum_value = bt_result.portfolio_value
    running_max = cum_value.cummax()
    drawdown = (cum_value - running_max) / running_max * 100
    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.values,
        mode="lines", name="回撤 (%)",
        fill="tozeroy", line=dict(color="#e63946", width=1.5),
    ), row=2, col=1)

    fig.update_layout(height=600, showlegend=True, template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0))
    fig.update_yaxes(title_text="組合市值 ($)", row=1, col=1)
    fig.update_yaxes(title_text="回撤 (%)", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # 績效報表
    st.subheader("📋 詳細績效指標")
    perf_df = analyzer.summary_table(perf)
    if perf_df is not None and not perf_df.empty:
        st.dataframe(perf_df, use_container_width=True, hide_index=True)

    # 月報酬熱圖
    st.subheader("📅 月報酬熱圖")
    monthly_ret = bt_result.returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    monthly_table = monthly_ret.to_frame("報酬")
    monthly_table["年"] = monthly_table.index.year
    monthly_table["月"] = monthly_table.index.month_name().str[:3]
    heatmap_data = monthly_table.pivot_table(index="年", columns="月", values="報酬", aggfunc="first")

    if not heatmap_data.empty:
        fig = px.imshow(
            heatmap_data * 100,
            text_auto=".1f",
            color_continuous_scale="RdYlGn",
            aspect="auto",
            labels=dict(x="月", y="年", color="月報酬 (%)"),
        )
        fig.update_layout(height=350, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

# ════════════════════ Tab 4: 回測持倉分析 ════════════════════
with tab4:
    st.subheader("📋 回測持倉分析（歷史回測，非實際 Alpaca 持倉）")
    st.caption(
        "💡 此 Tab 顯示**模型回測**下的持倉權重、權重歷史與交易記錄。"
        "**實際的 Alpaca 真實持倉**（含真實買入均價、損益）請看 Tab 6。"
    )

    latest_positions = bt_result.positions.iloc[-1] if not bt_result.positions.empty else pd.Series()
    holdings = latest_positions[latest_positions > 0].sort_values(ascending=False)

    if not holdings.empty:
        col1, col2 = st.columns([1, 1.5])
        with col1:
            fig = go.Figure(data=[go.Pie(
                labels=holdings.index[:10].tolist(),
                values=holdings.values[:10].tolist(),
                hole=0.4,
                textinfo="label+percent",
            )])
            fig.update_layout(
                height=400,
                template="plotly_dark",
                title=f"Top 10 持倉（共 {len(holdings)} 檔）",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.dataframe(
                holdings.to_frame("權重").style.format("{:.4%}"),
                use_container_width=True,
            )

        # 持倉權重變化
        st.subheader("📈 持倉權重歷史")
        top5 = holdings.nlargest(5).index.tolist()
        fig = go.Figure()
        colors = px.colors.qualitative.Set2
        for i, ticker in enumerate(top5):
            if ticker in bt_result.positions.columns:
                fig.add_trace(go.Scatter(
                    x=bt_result.positions.index,
                    y=bt_result.positions[ticker],
                    mode="lines",
                    name=ticker,
                    line=dict(color=colors[i % len(colors)], width=2),
                ))
        fig.update_layout(
            height=400,
            template="plotly_dark",
            yaxis_title="權重",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("無回測持倉資料")

    # 交易記錄（回測）
    st.subheader("📝 回測交易記錄")
    st.caption("這是回測期間產生的模擬交易，與 Alpaca 實際下單無關。")
    if bt_result.trades is not None and not bt_result.trades.empty:
        st.dataframe(bt_result.trades.tail(20), use_container_width=True, hide_index=True)
    else:
        st.info("無交易記錄")

# ════════════════════ Tab 5: 組合最佳化 ════════════════════
with tab5:
    st.subheader(f"🎯 組合最佳化 — {optimize_method}")

    # 建立價格矩陣
    price_matrix = pd.concat(
        {t: df["Close"] for t, df in price_data.items()},
        axis=1,
    )
    price_matrix = price_matrix.sort_index().ffill().bfill()

    # 使用最近 2 年資料做最佳化
    opt_start = end_date - timedelta(days=365 * 2)
    opt_price = price_matrix.loc[str(opt_start):str(end_date)]

    if not opt_price.empty and opt_price.shape[1] >= 5:
        optimizer = PortfolioOptimizer(risk_free_rate=RISK_FREE_RATE)
        opt_weights = optimizer.optimize(opt_price, method=optimize_method)

        if opt_weights:
            col1, col2 = st.columns([1, 1.5])
            with col1:
                fig = go.Figure(data=[go.Pie(
                    labels=list(opt_weights.keys()),
                    values=list(opt_weights.values()),
                    hole=0.4,
                    textinfo="label+percent",
                )])
                fig.update_layout(height=400, template="plotly_dark", title="最佳化權重")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                w_df = pd.DataFrame(list(opt_weights.items()), columns=["Ticker", "權重"])
                w_df["權重%"] = w_df["權重"].apply(lambda x: f"{x*100:.2f}%")
                st.dataframe(w_df, use_container_width=True, hide_index=True)

            # 預期風險-報酬
            if len(opt_weights) >= 3:
                st.subheader("📊 效率前緣模擬")
                try:
                    mu = expected_returns.mean_historical_return(opt_price)
                    S = risk_models.sample_cov(opt_price)
                    ef = EfficientFrontier(mu, S, weight_bounds=(0.02, 0.20))

                    # 模擬隨機組合
                    n_sim = 500
                    sim_returns = []
                    sim_vol = []
                    n_assets = len(mu)
                    for _ in range(n_sim):
                        w = np.random.dirichlet(np.ones(n_assets))
                        sim_returns.append(mu.values @ w * 252)
                        sim_vol.append(np.sqrt(w @ S.values @ w) * np.sqrt(252))
                        sim_vol[-1] = sim_vol[-1] if sim_vol[-1] > 0 else 0.001
                    sim_sharpe = [(r - 0.05) / v for r, v in zip(sim_returns, sim_vol)]

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=sim_vol, y=sim_returns,
                        mode="markers",
                        marker=dict(color=sim_sharpe, colorscale="Viridis", size=4, showscale=True,
                                    colorbar=dict(title="Sharpe")),
                        name="隨機組合",
                    ))
                    # 最佳組合
                    opt_ret = mu[list(opt_weights.keys())].values @ list(opt_weights.values()) * 252
                    opt_vol = np.sqrt(np.array(list(opt_weights.values())) @ S.loc[list(opt_weights.keys()), list(opt_weights.keys())].values @ np.array(list(opt_weights.values()))) * np.sqrt(252)
                    fig.add_trace(go.Scatter(
                        x=[opt_vol], y=[opt_ret],
                        mode="markers",
                        marker=dict(color="red", size=12, symbol="star"),
                        name="最佳組合",
                    ))
                    fig.update_layout(
                        height=500,
                        template="plotly_dark",
                        xaxis_title="年化波動",
                        yaxis_title="年化報酬",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"效率前緣繪製失敗: {e}")
        else:
            st.warning("最佳化無結果，可能資產數量不足")
    else:
        st.info(f"資料不足（需要至少 5 檔股票、2 年歷史資料），目前 {opt_price.shape[1]} 檔")

# ════════════════════ Tab 6: Alpaca 帳戶 & 活動 ════════════════════
with tab6:
    st.subheader("💼 Alpaca Paper Trading — 即時持倉與活動")

    try:
        alpaca = AlpacaBroker()
        account = alpaca.get_account()
        positions = alpaca.get_positions()
        open_orders = alpaca.get_open_orders()
    except Exception as e:
        st.error(f"無法連線 Alpaca: {e}")
        st.info("請確認環境變數已設定 ALPACA_API_KEY 和 ALPACA_SECRET_KEY")
        st.stop()

    # ── 帳戶摘要 ──
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("💰 總權益", f"${account['equity']:,.0f}")
    col2.metric("💵 現金", f"${account['cash']:,.0f}")
    col3.metric("📈 購買力", f"${account['buying_power']:,.0f}")
    day_str = f"{account['day_change_pct']*100:+.2f}%" if account['day_change_pct'] != 0 else "0.00%"
    col4.metric("📊 今日損益", day_str)
    col5.metric("📋 持倉數", f"{len(positions)} 檔")

    st.divider()

    # ── 持倉明細（增強版） ──
    if positions:
        st.subheader(f"📋 目前持倉（{len(positions)} 檔）")

        # 計算總數據
        total_mv = sum(p["market_value"] for p in positions)
        total_cost = sum(p["cost_basis"] for p in positions)
        total_pl = sum(p["unrealized_pl"] for p in positions)

        df = pd.DataFrame(positions)
        df["weight"] = df["market_value"] / total_mv * 100

        # 構建顯示用的 DataFrame
        display = pd.DataFrame({
            "股票": df["ticker"],
            "股數": df["qty"].apply(lambda x: f"{x:.0f}"),
            "均價": df["avg_entry_price"].apply(lambda x: f"${x:.2f}"),
            "現價": df["current_price"].apply(lambda x: f"${x:.2f}"),
            "市值": df["market_value"].apply(lambda x: f"${x:,.0f}"),
            "權重": df["weight"].apply(lambda x: f"{x:.1f}%"),
            "損益$": df["unrealized_pl"].apply(
                lambda x: f"{'🟢' if x>=0 else '🔴'} ${x:+,.0f}"),
            "損益%": df["unrealized_pl_pct"].apply(
                lambda x: f"{'🟢' if x>=0 else '🔴'} {x:+.1f}%"),
        })
        st.dataframe(display, use_container_width=True, hide_index=True)

        # 持倉權重圓餅圖 + 損益長條圖
        c1, c2 = st.columns(2)
        with c1:
            fig_pie = px.pie(
                df, values="market_value", names="ticker", title="持倉權重分佈",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_pie, use_container_width=True)
        with c2:
            df_sorted = df.sort_values("unrealized_pl")
            colors = ['#2ecc71' if x >= 0 else '#e74c3c' for x in df_sorted["unrealized_pl"]]
            fig_bar = go.Figure(data=[
                go.Bar(x=df_sorted["ticker"], y=df_sorted["unrealized_pl"],
                       marker_color=colors, text=df_sorted["unrealized_pl"].apply(lambda x: f"${x:+,.0f}"),
                       textposition='outside')
            ])
            fig_bar.update_layout(
                title="未實現損益分佈", height=350, margin=dict(l=0, r=0, t=30, b=0),
                xaxis_title="", yaxis_title="損益 (USD)",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # 損益彙總
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("總成本", f"${total_cost:,.0f}")
        c2.metric("總市值", f"${total_mv:,.0f}")
        pl_pct = (total_pl / total_cost * 100) if total_cost > 0 else 0
        c3.metric("未實現損益",
                  f"{'🟢' if total_pl>=0 else '🔴'} ${total_pl:+,.0f} ({pl_pct:+.2f}%)")
        # 計算今日變動
        today_change = sum(p.get("day_change", 0) for p in positions)
        c4.metric("今日持倉變動", f"{'🟢' if today_change>=0 else '🔴'} ${today_change:+,.0f}")
    else:
        st.info("目前無持倉")

    st.divider()

    # ── 未成交訂單 ──
    if open_orders:
        st.subheader(f"📝 待成交訂單（{len(open_orders)} 筆）")
        odf = pd.DataFrame(open_orders)
        odf = odf[["ticker", "side", "qty", "limit_price", "status", "created_at"]]
        odf.columns = ["股票", "方向", "數量", "限價", "狀態", "下單時間"]
        odf["限價"] = odf["限價"].apply(lambda x: f"${x:.2f}" if x else "-")
        st.dataframe(odf, use_container_width=True, hide_index=True)
    else:
        st.success("✅ 無待成交訂單，所有交易已完成")

    st.divider()

    # ── 最近活動記錄 ──
    st.subheader("📜 最近調倉活動")
    activity_path = Path(__file__).resolve().parent / "data" / "activity.json"
    if activity_path.exists():
        try:
            import json
            activities = json.loads(activity_path.read_text())
            if activities:
                for act in reversed(activities[-10:]):  # 最近 10 筆
                    act_date = act.get("date", "")
                    act_type = act.get("action", "")
                    det = act.get("details", {})

                    # 活動標題
                    icon = "🔄" if act_type == "rebalance" else "📊"
                    expander_label = f"{icon} {act_date} — {act_type}"
                    if act_type == "rebalance":
                        expander_label += f" | 權益 ${act.get('equity', 0):,.0f} | VIX {act.get('vix', 0):.1f}"
                    with st.expander(expander_label, expanded=(len(activities) - activities.index(act) <= 2)):
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("權益", f"${act.get('equity', 0):,.0f}")
                        col_b.metric("現金", f"${act.get('cash', 0):,.0f}")
                        col_c.metric("VIX / 宏觀", f"{act.get('vix', 0):.1f} / ×{act.get('macro_mult', 1.0):.0%}")

                        # 買賣摘要
                        before = act.get("positions_before", [])
                        after = act.get("positions_after", [])
                        if before:
                            before_tickers = {p["ticker"] for p in before}
                            after_tickers = {p["ticker"] for p in after} if after else set()
                            sold = before_tickers - after_tickers
                            bought = after_tickers - before_tickers if after else set()

                            if sold or bought:
                                cols = st.columns(2)
                                with cols[0]:
                                    if sold:
                                        st.markdown("**📉 賣出:** " + ", ".join(sorted(sold)))
                                    if bought:
                                        st.markdown("**📈 買入:** " + ", ".join(sorted(bought)))
                                with cols[1]:
                                    tp = det.get("tp_hits", [])
                                    sl = det.get("sl_hits", [])
                                    if tp:
                                        st.markdown(f"🎯 止盈: {', '.join(tp)}")
                                    if sl:
                                        st.markdown(f"🛑 止损: {', '.join(sl)}")

                            # 持倉變化
                            st.caption(f"調倉前: {', '.join(p['ticker'] + f'({p[\"pnl_pct\"]:+.1f}%)' for p in before)}")
                            if after:
                                st.caption(f"調倉後: {', '.join(p['ticker'] + f'({p[\"pnl_pct\"]:+.1f}%)' for p in after)}")
                        st.caption(f"訂單數: {det.get('orders', 0)}")
            else:
                st.info("尚無活動記錄")
        except Exception as e:
            st.warning(f"讀取活動記錄失敗: {e}")
    else:
        st.info("尚無活動記錄（調倉後自動產生）")

    st.divider()

    # ── 策略狀態摘要 ──
    st.subheader("🎯 策略狀態")
    try:
        from us_quant.risk import RiskManager, MACRO_CALENDAR
        from datetime import date

        rm = RiskManager(stop_loss_pct=0.05, take_profit_pct=0.10, max_holding=8)

        # VIX
        try:
            vix_data = fetch_vix(start="2026-01-01")
            current_vix = float(vix_data.iloc[-1]) if not vix_data.empty else 0
        except Exception:
            current_vix = 0

        macro_mult = rm.get_macro_multiplier(dt=date.today(), vix=current_vix)

        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("因子數", "8")
        sc2.metric("最大持倉", "8 檔")
        sc3.metric("風控", "SL -5% / TP +10%")
        sc4.metric("調倉模式", "信號驅動")

        sc5, sc6, sc7, sc8 = st.columns(4)
        vix_emoji = "🔴" if current_vix > 30 else "🟡" if current_vix > 20 else "🟢"
        sc5.metric("VIX", f"{vix_emoji} {current_vix:.1f}")
        sc6.metric("宏觀係數", f"×{macro_mult:.0%}")
        today_str = date.today().strftime("%Y-%m-%d")
        event_today = MACRO_CALENDAR.get(today_str)
        sc7.metric("今日事件", event_today.upper() if event_today else "無")
        sc8.metric("因子權重",
                   f"momentum .20 | ai .20 | quality .15")
    except Exception as e:
        st.caption(f"策略狀態加載中... ({e})")

# ── Footer ──
st.divider()
st.caption(
    f"⚠️ 免責聲明：本工具僅供研究與教育用途，不構成投資建議。"
    f"歷史績效不代表未來表現。"
    f" | 最後更新: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
)
