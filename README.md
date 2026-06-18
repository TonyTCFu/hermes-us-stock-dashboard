# 📈 【Claude Code】美股量化分析平台

> 8因子信号驱动 + AI主题 | 激进短期 | SL5%/TP10% | Alpaca Paper Trading

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red)](https://streamlit.io)

## ✨ 功能

- **📦 数据层** — yfinance 抓取股价 & 基本面、SQLite 本地快取、行业分类、DXY/VIX 数据
- **🧬 8因子引擎**
  - 技术面：momentum（12月动能）、low_vol（年化波动）、flow（MFI资金流向）
  - 基本面：value（PE/PB/PS）、quality（ROE/毛利率/负债）、revenue_growth（营收增长）
  - 进阶：**industry_momentum**（行业轮动）、**ai_industry**（AI主题强度）
- **🛡️ 风控模块** — SL5%/TP10%个券止损、FOMC/CPI/NFP宏观事件降仓、VIX调节
- **📊 回测引擎** — biweekly / monthly / quarterly / yearly 自动再平衡
- **💼 Alpaca Paper Trading** — 实时账户、持仓、信号排名、活动记录

## 🚀 公网访问

- Dashboard: http://cc-us-stock-dashboard.futienchun.com

## 🛠️ 本地开发

```bash
cd ~/hermes-workplace/us-stock-quant

# 启动 Dashboard
PYTHONPATH=.venv/lib/python3.12/site-packages:. .venv/bin/streamlit run app.py

# 运行调仓脚本
PYTHONPATH=.venv/lib/python3.12/site-packages:. .venv/bin/python scripts/rebalance_v2.py

# 生成 HTML Dashboard
PYTHONPATH=.venv/lib/python3.12/site-packages:. .venv/bin/python scripts/build_dashboard_html.py
```

## 📁 项目结构

```
us-stock-quant/
├── app.py                      # Streamlit 入口
├── us_quant/                   # 核心套件
│   ├── config.py               # 环境变量驱动配置
│   ├── env.py                  # 统一环境变量读取
│   ├── risk.py                 # 风控模块（SL/TP + 宏观事件 + 仓位分配）
│   ├── data/                   # 数据层（yfinance + SQLite）
│   ├── factors/                # 8因子引擎（含AI主题因子）
│   ├── signals/                # Z-score合成 + VIX调节
│   ├── backtest/               # 回测引擎
│   ├── portfolio/              # PyPortfolioOpt 最优化
│   ├── reporting/              # 绩效指标
│   └── broker/                 # Alpaca API 封装
├── scripts/                    # CLI 工具
│   ├── rebalance_v2.py         # 信号驱动调仓
│   └── build_dashboard_html.py # 静态 HTML Dashboard 生成
└── static/                     # PWA 图标
```

## 🔑 环境变量

| 变量 | 必填 | 说明 |
|------|:----:|------|
| `ALPACA_API_KEY` | ✅ | Alpaca Paper Trading API Key |
| `ALPACA_SECRET_KEY` | ✅ | Alpaca Paper Trading Secret |
| `ALPACA_BASE_URL` | ❌ | 默认 `https://paper-api.alpaca.markets` |
| `STOCK_UNIVERSE` | ❌ | 39档大型美股 |
| `FACTOR_WEIGHTS` | ❌ | 8因子权重（momentum .20, ai_industry .20, quality .15, ...） |

## ⚠️ 免责声明

本工具仅供研究与教育用途，不构成任何投资建议。历史绩效不代表未来表现。
