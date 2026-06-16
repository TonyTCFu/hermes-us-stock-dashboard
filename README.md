# 📈 【Hermes】美股量化分析平台

> 美股多因子量化選股 + 回測 + 組合最佳化 + Alpaca Paper Trading 整合

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red)](https://streamlit.io)

## ✨ 功能

- **📦 資料層** — yfinance 抓取股價 & 基本面、SQLite 本地快取、行業分類、DXY/VIX 宏觀資料
- **🧬 10 因子引擎**
  - 技術面：momentum（12 月動能）、low_vol（年化波動）
  - 基本面：value（PE/PB/PS）、quality（ROE/毛利率/負債）、size（市值）、div_yield、revenue_growth
  - 進階：**industry_momentum**（產業輪動）、**flow**（MFI 資金流向）、**fx_exposure**（DXY × 海外營收曝險）
- **🛡️ VIX 市場狀態調節** — VIX > 30 信號砍半、20-30 線性遞減
- **📊 回測引擎** — biweekly / monthly / quarterly / yearly 自動再平衡
- **🎯 組合最佳化** — max_sharpe / min_vol / risk_parity
- **💼 Alpaca Paper Trading** — 即時帳戶、持倉、未成交訂單

## 🚀 部署到 Render（一鍵）

1. **建立 GitHub repo**（公開，名字任意，例如 `hermes-us-stock-dashboard`）
2. **把這個 repo push 上去**
3. 去 [render.com](https://render.com) → 用 GitHub 登入
4. 點 **New +** → **Web Service** → 選你的 repo
5. Render 會自動讀 `render.yaml`，確認設定：
   - Runtime: Python
   - Build Command: （留空，自動跑 `pip install -r requirements.txt`）
   - Start Command: `streamlit run app.py --server.port $PORT ...`
   - Plan: **Free**
   - Region: **Singapore**（亞洲節點，台灣延遲最低）
6. **設定環境變數**（在 Render 後台 Environment 頁面）：
   - `ALPACA_API_KEY`：你的 paper trading key
   - `ALPACA_SECRET_KEY`：你的 paper trading secret
   - （其他 render.yaml 已寫好）
7. 點 **Deploy Web Service**，等 5-10 分鐘
8. Render 會給你一個 `https://xxx.onrender.com` 的網址

> ⚠️ **免費方案 15 分鐘 idle 會 sleep**，下次訪問要等 30-60 秒冷啟動

## 🛠️ 本地開發

```bash
# 用 uv 安裝（推薦）
pip install uv
uv venv --python 3.12
uv pip install -e .

# 或用 pip
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 啟動 Dashboard
streamlit run app.py
```

## 📁 專案結構

```
hermes-us-stock-dashboard/
├── app.py                  # Streamlit 入口（Render 部署用）
├── requirements.txt        # 依賴清單
├── render.yaml             # Render 部署配置
├── us_quant/               # 核心套件
│   ├── env.py              # 統一環境變數讀取（Render + 本地）
│   ├── config.py           # 配置（環境變數驅動）
│   ├── data/               # 資料層（yfinance + SQLite）
│   ├── factors/            # 10 因子引擎
│   ├── signals/            # Z-score 合成 + VIX 調節
│   ├── backtest/           # 回測引擎
│   ├── portfolio/          # PyPortfolioOpt 最佳化
│   ├── reporting/          # 績效指標
│   └── broker/             # Alpaca API 封裝
├── scripts/                # CLI 工具（rebalance.py 等）
└── static/                 # PWA 圖示
```

## 🔑 環境變數

| 變數 | 必填 | 說明 |
|------|:----:|------|
| `ALPACA_API_KEY` | ✅ | Alpaca Paper Trading API Key |
| `ALPACA_SECRET_KEY` | ✅ | Alpaca Paper Trading Secret |
| `ALPACA_BASE_URL` | ❌ | 預設 `https://paper-api.alpaca.markets` |
| `STOCK_UNIVERSE` | ❌ | 39 檔大型美股，預設值見 `render.yaml` |
| `FACTOR_WEIGHTS` | ❌ | 10 因子權重，預設值見 `render.yaml` |
| `REBALANCE_FREQUENCY` | ❌ | biweekly / monthly / quarterly |
| `DATA_DIR` | ❌ | SQLite DB 路徑（Render 預設 `/tmp/data`） |

## ⚠️ 免責聲明

本工具僅供研究與教育用途，不構成任何投資建議。歷史績效不代表未來表現。