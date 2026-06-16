#!/bin/bash
# 一鍵啟動：美股量化 Dashboard + Auth Proxy + 公開隧道
# 用法: bash start_dashboard.sh

set -e

cd "$(dirname "$0")"

echo "========================================"
echo " 美股量化 Dashboard — 啟動"
echo "========================================"

# 1. 清除舊服務
echo ">> 清除舊服務..."
lsof -ti :8501 2>/dev/null | xargs kill -9 2>/dev/null || true
lsof -ti :9001 2>/dev/null | xargs kill -9 2>/dev/null || true
sleep 1

# 2. 啟動 Streamlit
echo ">> 啟動 Streamlit (port 8501)..."
.venv/bin/streamlit run dashboard/app.py --server.port 8501 --server.headless true &
ST_PID=$!
sleep 3

# 檢查 Streamlit 是否活著
if ! kill -0 $ST_PID 2>/dev/null; then
    echo "❌ Streamlit 啟動失敗"
    exit 1
fi
echo "   ✅ Streamlit (PID $ST_PID)"

# 3. 啟動 Auth Proxy (Node.js)
echo ">> 啟動 Auth Proxy (port 9001)..."
node scripts/auth_proxy_v2.js &
AP_PID=$!
sleep 1

if ! kill -0 $AP_PID 2>/dev/null; then
    echo "❌ Auth Proxy 啟動失敗"
    exit 1
fi
echo "   ✅ Auth Proxy (PID $AP_PID)"

# 4. 啟動 SSH Tunnel
echo ">> 啟動 SSH Tunnel (localhost.run)..."
ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=60 \
    -R 80:localhost:9001 nokey@localhost.run 2>&1 |
    while IFS= read -r line; do
        echo "$line"
        # 抓取 URL
        if [[ "$line" =~ https?://[a-z0-9]+\.lhr\.life ]]; then
            URL="${BASH_REMATCH[0]}"
            echo ""
            echo "========================================"
            echo "  ✅ 公開網址: $URL"
            echo "  帳號: tony"
            echo "  密碼: quant2024"
            echo "========================================"
            echo ""
            echo "按 Ctrl+C 停止所有服務"
        fi
    done

echo ">>  Tunnel 中斷，清理服務..."
kill $ST_PID $AP_PID 2>/dev/null || true
