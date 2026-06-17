#!/usr/bin/env python3
"""生成静态 HTML Dashboard → GitHub Pages (cc-us-stock-dashboard.futienchun.com)

用法：
    python scripts/build_dashboard_html.py           # 生成 + 部署
    python scripts/build_dashboard_html.py --local   # 仅生成到本地 /tmp/dashboard.html
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import os
import subprocess
import time
import logging
from datetime import date, datetime

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── 配置 ──
DEPLOY_REPO = Path("/tmp/cc-us-stock-dashboard")
DEPLOY_REMOTE = "https://github.com/TonyTCFu/cc-us-stock-dashboard.git"
PUBLIC_URL = "http://cc-us-stock-dashboard.futienchun.com"

# ── Alpaca ──
try:
    from dotenv import load_dotenv
    _env = Path(__file__).resolve().parent.parent / ".env"
    if _env.exists():
        load_dotenv(_env)
except ImportError:
    pass

from alpaca.trading.client import TradingClient


def get_alpaca_data():
    """获取 Alpaca 账户和持仓数据。"""
    client = TradingClient(
        os.getenv("ALPACA_API_KEY"),
        os.getenv("ALPACA_SECRET_KEY"),
        paper=True,
    )
    account = client.get_account()
    positions = client.get_all_positions()

    data = {
        "equity": round(float(account.equity), 2),
        "cash": round(float(account.cash), 2),
        "buying_power": round(float(account.buying_power), 2),
        "day_change_pct": round(
            (float(account.equity) - float(account.last_equity)) / float(account.last_equity) * 100, 2
        ) if float(account.last_equity) > 0 else 0,
        "positions": [],
    }

    total_mv = 0
    for p in positions:
        mv = float(p.market_value)
        total_mv += mv
        data["positions"].append({
            "ticker": p.symbol,
            "qty": float(p.qty),
            "avg_price": float(p.avg_entry_price),
            "current_price": float(p.current_price),
            "market_value": mv,
            "unrealized_pl": float(p.unrealized_pl),
            "unrealized_pl_pct": round(float(p.unrealized_plpc) * 100, 2),
        })

    data["total_market_value"] = round(total_mv, 2)
    data["total_cost"] = round(total_mv - sum(p["unrealized_pl"] for p in data["positions"]), 2)
    data["total_pl"] = round(sum(p["unrealized_pl"] for p in data["positions"]), 2)
    data["position_count"] = len(positions)

    return data


def get_signal_data():
    """运行因子分析获取信号排名。"""
    try:
        from us_quant.data import (
            ensure_price_data, fetch_all_fundamentals, fetch_sector_info,
            fetch_dxy, fetch_vix, DataStore, fetch_price_data,
        )
        from us_quant.factors import get_factor
        from us_quant.signals import SignalCombiner
        from us_quant.config import DB_PATH, STOCK_UNIVERSE, FACTOR_WEIGHTS, MAX_HOLDING

        store = DataStore(DB_PATH)
        price_data = ensure_price_data(STOCK_UNIVERSE[:30], store, start="2018-01-01")
        price_data = {k: v for k, v in price_data.items() if not v.empty}
        tickers = list(price_data.keys())

        fundamentals = fetch_all_fundamentals(tickers)
        sector_map = fetch_sector_info(tickers)
        spy_raw = fetch_price_data(["SPY"], start="2018-01-01")
        spy_df = spy_raw.get("SPY", pd.DataFrame())

        dxy = fetch_dxy(start="2018-01-01")
        vix_raw = fetch_vix(start="2026-01-01")
        current_vix = round(float(vix_raw.iloc[-1]), 2) if not vix_raw.empty else 0

        params = {
            "momentum": {},
            "value": {"fundamentals": fundamentals},
            "quality": {"fundamentals": fundamentals},
            "low_vol": {},
            "revenue_growth": {"fundamentals": fundamentals},
            "industry_momentum": {"sector_map": sector_map},
            "flow": {},
            "ai_industry": {"benchmark": spy_df},
        }

        factor_results = []
        for name in FACTOR_WEIGHTS:
            if name not in params:
                continue
            try:
                r = get_factor(name).compute(price_data, **params[name])
                if not r.scores.empty:
                    factor_results.append(r)
            except Exception:
                continue

        signal = SignalCombiner(FACTOR_WEIGHTS).combine(factor_results)
        latest = signal.groupby(level=0).last().sort_values(ascending=False)

        top_n = []
        for rank, (ticker, score) in enumerate(latest.head(MAX_HOLDING).items(), 1):
            top_f = []
            for fr in factor_results:
                w = FACTOR_WEIGHTS.get(fr.name, 0)
                if w == 0 or fr.scores.empty:
                    continue
                try:
                    val = fr.scores.xs(ticker, level="ticker").iloc[-1]
                    top_f.append((fr.name, round(val * w, 3)))
                except (KeyError, IndexError):
                    continue
            top_f.sort(key=lambda x: -x[1])
            top_n.append({
                "rank": rank, "ticker": ticker, "signal": round(float(score), 3),
                "top_factors": ", ".join([f"{n}({v:.2f})" for n, v in top_f[:2]]),
            })

        return {"top_n": top_n, "vix": current_vix, "factor_count": len(factor_results)}

    except Exception as e:
        logger.warning("因子分析失败: %s", e)
        return {"top_n": [], "vix": 0, "factor_count": 0, "error": str(e)}


def get_activity():
    """读取最近活动记录。"""
    path = Path(__file__).resolve().parent.parent / "data" / "activity.json"
    if path.exists():
        try:
            return json.loads(path.read_text())[-10:]  # 最近 10 条
        except (json.JSONDecodeError, OSError):
            pass
    return []


def get_macro_info():
    """获取宏观环境信息。"""
    try:
        from us_quant.risk import MACRO_CALENDAR
        today_str = date.today().strftime("%Y-%m-%d")
        event = MACRO_CALENDAR.get(today_str)
    except Exception:
        event = None
    return {"event_today": event.upper() if event else None}


def render_html(alpaca, signals, activities, macro):
    """渲染完整 HTML Dashboard。"""
    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M") + " ET"
    refresh_str = now.strftime("%Y-%m-%d %H:%M")

    pos_rows = ""
    for p in sorted(alpaca["positions"], key=lambda x: -x["market_value"]):
        pl_class = "bull" if p["unrealized_pl"] >= 0 else "bear"
        pos_rows += f"""<tr>
            <td><b>{p['ticker']}</b></td>
            <td class="num">{p['qty']:.0f}</td>
            <td class="num">${p['avg_price']:.2f}</td>
            <td class="num">${p['current_price']:.2f}</td>
            <td class="num">${p['market_value']:,.0f}</td>
            <td class="num {pl_class}">${p['unrealized_pl']:+,.0f}</td>
            <td class="num {pl_class}">{p['unrealized_pl_pct']:+.1f}%</td>
        </tr>"""

    signal_rows = ""
    for s in signals.get("top_n", []):
        signal_rows += f"""<tr>
            <td class="num">{s['rank']}</td>
            <td><b>{s['ticker']}</b></td>
            <td class="num">{s['signal']:+.3f}</td>
            <td style="font-size:10px;color:#8b949e">{s['top_factors']}</td>
        </tr>"""

    activity_html = ""
    for act in reversed(activities[-6:]):
        det = act.get("details", {})
        sold = det.get("sell_tickers", [])
        tp = det.get("tp_hits", [])
        sl = det.get("sl_hits", [])
        tags = ""
        if tp:
            tags += " ".join([f'<span class="badge" style="background:#1b3a1b;color:#3fb950">🎯 止盈 {t}</span>' for t in tp])
        if sl:
            tags += " ".join([f'<span class="badge" style="background:#3a1b1b;color:#f85149">🛑 止损 {t}</span>' for t in sl])
        if sold:
            tags += f' <span class="badge" style="background:#1a2a3a;color:#58a6ff">📉 卖 {",".join(sold)}</span>'

        activity_html += f"""<div class="card">
            <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:4px">
                <span style="font-size:12px">🔄 <b>{act['date']}</b></span>
                <span style="font-size:10px;color:#8b949e">权益 ${act.get('equity',0):,.0f} | VIX {act.get('vix',0)} | 宏观 ×{act.get('macro_mult',1.0):.0%}</span>
            </div>
            <div style="margin-top:6px;font-size:11px">{tags}</div>
        </div>"""

    vix = signals.get("vix", 0)
    vix_emoji = "🔴" if vix > 30 else "🟡" if vix > 20 else "🟢"
    vix_label = "恐慌" if vix > 30 else "警戒" if vix > 20 else "正常"
    macro_event = macro.get("event_today") or "无"

    total_pl = alpaca["total_pl"]
    pl_emoji = "🟢" if total_pl >= 0 else "🔴"
    day_emoji = "🟢" if alpaca["day_change_pct"] >= 0 else "🔴"

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0,maximum-scale=1.0,user-scalable=no">
<meta http-equiv="refresh" content="900">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="Claude Code Quant">
<link rel="apple-touch-icon" sizes="192x192" href="icon-192.png">
<link rel="icon" type="image/png" sizes="192x192" href="icon-192.png">
<link rel="manifest" href="manifest.json">
<title>【Claude Code】美股量化 Dashboard</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0f1117;color:#e1e4e8;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;padding:10px;line-height:1.5;-webkit-text-size-adjust:100%}}
h1{{font-size:17px;color:#58a6ff;margin-bottom:2px}}
h2{{font-size:14px;color:#58a6ff;margin:18px 0 6px;padding-bottom:4px;border-bottom:2px solid #30363d}}
.subtitle{{color:#8b949e;font-size:11px;margin:2px 0 12px;line-height:1.5}}
.card{{background:#1a1d27;border:1px solid #30363d;border-radius:8px;padding:10px;margin-bottom:8px;overflow-x:auto}}
.metrics-bar{{display:flex;align-items:center;gap:6px;flex-wrap:wrap;font-size:11px;color:#8b949e;padding:6px 0 10px;line-height:1.7}}
.metrics-bar b{{color:#f0f3f5;font-size:12px}}
.m-sep{{color:#30363d}}
table{{width:100%;border-collapse:collapse;font-size:11px}}
th{{text-align:left;padding:5px 6px;border-bottom:2px solid #30363d;color:#8b949e;font-weight:600;font-size:10px;white-space:nowrap}}
td{{padding:4px 6px;border-bottom:1px solid #30363d;white-space:nowrap}}
.num{{text-align:right;font-variant-numeric:tabular-nums}}
.bull{{color:#3fb950}} .bear{{color:#f85149}}
.nav{{display:flex;gap:6px;margin-bottom:10px;font-size:10px;flex-wrap:wrap}}
.nav a{{color:#58a6ff;text-decoration:none}}
.footer{{margin-top:14px;padding:8px 0;border-top:1px solid #30363d;color:#8b949e;font-size:9px}}
.badge{{display:inline-block;padding:1px 6px;border-radius:8px;font-size:9px;font-weight:bold;margin:1px}}
.grid2{{display:flex;flex-direction:column;gap:8px}}
@media(min-width:768px){{
    body{{padding:16px}} h1{{font-size:18px}} h2{{font-size:15px}}
    .card{{padding:14px;margin-bottom:12px}} .metrics-bar{{font-size:13px;gap:10px}}
    .grid2{{flex-direction:row}} .grid2 .card{{flex:1}}
}}
</style>
</head>
<body>

<div class="nav">
  <span style="color:#58a6ff">📈 <b>Claude Code 美股量化</b></span>
  <span style="color:#8b949e">| 刷新: {refresh_str}</span>
  <span style="color:#8b949e">| 每 15 分钟自动刷新</span>
</div>

<h1>📈 【Claude Code】美股量化模型 — 即时持仓</h1>
<div class="subtitle">
  $106K Alpaca Paper | 8因子 + AI主题 | SL5%/TP10% | 信号驱动调仓 | 2026-06-12 起运行
</div>

<div class="metrics-bar">
    <span>权益 <b>${alpaca['equity']:,.0f}</b></span>
    <span class="m-sep">|</span>
    <span>现金 <b>${alpaca['cash']:,.0f}</b></span>
    <span class="m-sep">|</span>
    <span>未实现 <b style="color:{'#3fb950' if total_pl>=0 else '#f85149'}">${total_pl:+,.0f}</b></span>
    <span class="m-sep">|</span>
    <span>今日 <b style="color:{'#3fb950' if alpaca['day_change_pct']>=0 else '#f85149'}">{alpaca['day_change_pct']:+.2f}%</b></span>
    <span class="m-sep">|</span>
    <span>持仓 <b>{alpaca['position_count']} 档</b></span>
    <span class="m-sep">|</span>
    <span>VIX <b>{vix_emoji} {vix}</b> ({vix_label})</span>
</div>

<div class="grid2">
    <div class="card">
        <h2>💼 即时持仓 ({alpaca['position_count']} 档)</h2>
        <table>
            <tr><th>股票</th><th class="num">股数</th><th class="num">成本</th><th class="num">现价</th><th class="num">市值</th><th class="num">盈亏$</th><th class="num">盈亏%</th></tr>
            {pos_rows}
            <tr style="font-weight:bold;border-top:2px solid #58a6ff">
                <td>合计</td><td class="num">-</td><td class="num">-</td><td class="num">-</td>
                <td class="num">${alpaca['total_market_value']:,.0f}</td>
                <td class="num {pl_class}">${total_pl:+,.0f}</td><td class="num {pl_class}">-</td>
            </tr>
        </table>
    </div>

    <div class="card">
        <h2>🎯 8因子信号排名 (Top {len(signals.get('top_n', []))})</h2>
        <table>
            <tr><th>#</th><th>股票</th><th class="num">信号</th><th>强势因子</th></tr>
            {signal_rows}
        </table>
        <div style="margin-top:6px;font-size:10px;color:#8b949e">
            权重: momentum .20 | ai_industry .20 | quality .15 | flow/value/revenue_growth/industry_momentum .10 | low_vol .05
        </div>
    </div>
</div>

<h2>📜 最近调仓活动</h2>
{activity_html if activity_html else '<div class="card"><span style="color:#8b949e">暂无记录</span></div>'}

<h2>🌍 策略状态</h2>
<div class="card">
    <div class="metrics-bar" style="padding:0">
        <span>📊 因子 <b>8个</b></span><span class="m-sep">|</span>
        <span>🎯 最大持仓 <b>8 档</b></span><span class="m-sep">|</span>
        <span>🛡️ 风控 <b>SL -5% / TP +10%</b></span><span class="m-sep">|</span>
        <span>🔄 调仓 <b>信号驱动</b></span><span class="m-sep">|</span>
        <span>📅 今日事件 <b>{macro_event}</b></span><span class="m-sep">|</span>
        <span>🕐 下次定时检查 <b>明天 5:15 BJT</b></span>
    </div>
</div>

<div class="footer">
    ⚠️ 免责声明：本工具仅供研究用途，不构成投资建议。历史绩效不代表未来表现。<br>
    生成: {refresh_str} | Alpaca Paper Trading | GitHub Pages 部署 | 定时更新: 北京时间周二~六 5:15 AM
</div>

</body>
</html>"""
    return html


def deploy(html_content: str):
    """将 HTML 推送到 GitHub Pages。"""
    if DEPLOY_REPO.exists():
        subprocess.run(["git", "-C", str(DEPLOY_REPO), "pull", "origin", "main"], capture_output=True)
    else:
        subprocess.run(["git", "clone", DEPLOY_REMOTE, str(DEPLOY_REPO)], check=True)

    # 写 index.html
    index_path = DEPLOY_REPO / "index.html"
    index_path.write_text(html_content)

    # 提交并推送
    subprocess.run(["git", "-C", str(DEPLOY_REPO), "add", "index.html"], check=True)
    result = subprocess.run(
        ["git", "-C", str(DEPLOY_REPO), "commit", "-m", f"Update {datetime.now().strftime('%Y-%m-%d %H:%M')} ET"],
        capture_output=True, text=True,
    )
    if "nothing to commit" not in result.stdout + result.stderr:
        subprocess.run(["git", "-C", str(DEPLOY_REPO), "push", "origin", "main"], check=True)
        logger.info("✅ Dashboard 已部署到 %s", PUBLIC_URL)
    else:
        logger.info("⏭️  无变更，跳过部署")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="生成静态 HTML Dashboard")
    parser.add_argument("--local", "-l", action="store_true", help="仅生成到本地，不部署")
    parser.add_argument("--output", "-o", type=str, default="/tmp/dashboard.html", help="输出路径")
    args = parser.parse_args()

    print("📊 生成 Dashboard...")

    # 1. Alpaca 数据
    print("   📡 Alpaca 持仓...")
    alpaca = get_alpaca_data()
    print(f"      权益 ${alpaca['equity']:,.0f} | {alpaca['position_count']} 档持仓")

    # 2. 信号排名
    print("   🔬 因子分析...")
    signals = get_signal_data()
    if signals.get("error"):
        print(f"      ⚠️ 因子分析失败: {signals['error']}")
    else:
        print(f"      VIX {signals['vix']} | Top {len(signals['top_n'])} 信号就绪")

    # 3. 活动记录
    activities = get_activity()
    print(f"   📜 活动记录: {len(activities)} 条")

    # 4. 宏观
    macro = get_macro_info()

    # 5. 渲染
    html = render_html(alpaca, signals, activities, macro)

    # 6. 输出
    output_path = Path(args.output)
    output_path.write_text(html)
    print(f"   ✅ HTML 已生成: {output_path} ({len(html):,} bytes)")

    # 7. 部署
    if not args.local:
        print("   🚀 部署到 GitHub Pages...")
        deploy(html)
        print(f"   🌐 公网: {PUBLIC_URL}")
    else:
        print(f"   📄 本地预览: file://{output_path}")


if __name__ == "__main__":
    main()
