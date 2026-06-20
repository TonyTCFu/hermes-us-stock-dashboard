#!/bin/bash
# 盘中每小时刷新 Dashboard — launchd 定时调用
# 仅在美股盘中时段运行 (ET 9:30-16:00 → BJT 21:30-04:00)

set -e
LOG="/tmp/quant-dashboard-hourly.log"
PROJ="/Users/tonyfu/hermes-workplace/us-stock-quant"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 刷新 Dashboard..." >> "$LOG"
cd "$PROJ"
PYTHONPATH=.venv/lib/python3.12/site-packages:. .venv/bin/python scripts/build_dashboard_html.py >> "$LOG" 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 完成" >> "$LOG"
