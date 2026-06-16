"""全域設定 — Render 環境變數驅動，本地 .env 為輔

設計：
- 統一從環境變數讀取（Render 部署標準）
- 本地開發有 .env 會自動載入（透過 env.py）
- 不再直接依賴 .env 檔案存在
"""
import os
from pathlib import Path

from .env import get_env, get_env_float

# ── 目錄 ──
# Render 容器每次重啟會清空非 /tmp 目錄，所以用 /tmp 持久化 DB
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(get_env("DATA_DIR", str(_PROJECT_ROOT / "data")))
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── 股票池 ──
STOCK_UNIVERSE = [
    s.strip()
    for s in get_env(
        "STOCK_UNIVERSE",
        "AAPL,MSFT,GOOGL,AMZN,META,NVDA,TSLA,JPM,V,MA,"
        "UNH,JNJ,PG,KO,PEP,HD,MCD,NKE,DIS,BA,CAT,GE,"
        "XOM,CVX,ABBV,LLY,TMO,AVGO,ADBE,CRM,NFLX,ACN,CSCO,INTC,AMD,IBM,ORCL,SAP,TM",
    ).split(",")
]

# ── 回測預設 ──
BENCHMARK_TICKER = get_env("BENCHMARK_TICKER", "SPY")
RISK_FREE_RATE = get_env_float("RISK_FREE_RATE", 0.05)
INITIAL_CAPITAL = get_env_float("INITIAL_CAPITAL", 1_000_000)
TRANSACTION_COST = get_env_float("TRANSACTION_COST", 0.001)
SLIPPAGE = get_env_float("SLIPPAGE", 0.0005)
REBALANCE_FREQUENCY = get_env("REBALANCE_FREQUENCY", "biweekly")
MAX_HOLDING = int(get_env_float("MAX_HOLDING", 20))

# ── 因子設定 ──
# 格式: 因子名:權重,逗號分隔
FACTOR_WEIGHTS = {
    k.strip(): float(v.strip())
    for item in get_env(
        "FACTOR_WEIGHTS",
        "momentum:0.15,value:0.10,quality:0.15,low_vol:0.10,size:0.05,"
        "div_yield:0.05,revenue_growth:0.10,"
        "industry_momentum:0.10,flow:0.10,fx_exposure:0.10",
    ).split(",")
    for k, v in [item.split(":")]
}

# ── 資料庫 ──
DB_PATH = DATA_DIR / "market.db"