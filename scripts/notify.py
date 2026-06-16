#!/usr/bin/env python3
"""notify.py — Supervisor / monitor 的通知工具

支持两种渠道（按 .env 配）：
  - macOS 通知中心：默认开，0 配置
  - iCloud SMTP：可选，配 SMTP_APP_PASSWORD 才生效

使用：
  python scripts/notify.py --level info --title "..." --body "..."
  python scripts/notify.py --level error --title "..." --body "..." --stdout-log /path/to/log

退出码：
  0 = 至少一个渠道成功（或没人订阅）
  1 = 配了 SMTP 但发信失败
"""

import argparse
import os
import smtplib
import ssl
import subprocess
import sys
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

# ── 加载 .env（从项目根找）──
try:
    from dotenv import load_dotenv
    _ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
    if _ENV_PATH.exists():
        load_dotenv(_ENV_PATH)
except ImportError:
    pass  # 没装 dotenv 就用裸环境变量

# ── 从 .env 读 SMTP 配置（可选）──
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.mail.me.com")  # iCloud SMTP
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")                  # 你的 iCloud 邮箱
SMTP_PASS = os.getenv("SMTP_APP_PASSWORD", "")          # appleid.apple.com 生成的 app-specific password
NOTIFY_TO = os.getenv("NOTIFY_TO", SMTP_USER)           # 默认发给自己

# macOS 通知中心常开（不需要任何配置）
ENABLE_MACOS_NOTIFY = os.getenv("ENABLE_MACOS_NOTIFY", "1") == "1"

# SMTP 关闭开关：iCloud SMTP 在 Tony 所在 IP 段（台湾 111.246.x）被 Apple 拒连
# 见 .loop/findings.md F-009。设 1 = 完全不走 SMTP，只用 macOS 通知中心
SMTP_DISABLED = os.getenv("SMTP_DISABLED", "1") == "1"


def send_macos_notification(title: str, body: str, subtitle: str = "") -> bool:
    """通过 osascript 调 macOS 通知中心。"""
    if not ENABLE_MACOS_NOTIFY:
        return False
    # 拼接消息（osascript 的 display notification）
    # 注意：body 用 \\n 换行
    full_body = f"{subtitle}\\n{body}" if subtitle else body
    # AppleScript 转义：双引号变 \\\"
    def esc(s):
        return s.replace('\\', '\\\\').replace('"', '\\"')
    script = f'display notification "{esc(full_body)}" with title "{esc(title)}"'
    try:
        subprocess.run(
            ["osascript", "-e", script],
            check=True,
            capture_output=True,
            timeout=5,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"⚠️ macOS 通知中心失败: {e}", file=sys.stderr)
        return False


def send_smtp_email(title: str, body: str, level: str, stdout_log: str = "") -> bool:
    """通过 iCloud SMTP 发邮件。如果没配 SMTP_USER/PASS 或 SMTP_DISABLED=1 直接跳过（不报错）。"""
    if SMTP_DISABLED:
        return False  # SMTP 已显式关闭
    if not (SMTP_USER and SMTP_PASS and NOTIFY_TO):
        return False  # 静默跳过 = SMTP 没配

    msg = MIMEMultipart("alternative")
    msg["From"] = SMTP_USER
    msg["To"] = NOTIFY_TO
    msg["Subject"] = f"[quant-{level}] {title}"

    # 邮件正文（HTML + 纯文本双版本）
    text_body = f"""{title}

{body}

时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    if stdout_log:
        text_body += f"\n详细 log: {stdout_log}\n"

    html_body = f"""
<html><body>
<h2 style="color: {'#d00' if level == 'error' else '#06c' if level == 'info' else '#888'};">{title}</h2>
<pre style="font-family: -apple-system, sans-serif; white-space: pre-wrap;">{body}</pre>
<p style="color: #888; font-size: 0.9em;">⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
{"<p>📝 详细 log: <code>" + stdout_log + "</code></p>" if stdout_log else ""}
</body></html>
"""
    msg.attach(MIMEText(text_body, "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    try:
        context = ssl.create_default_context()
        # 465 = SMTPS (SSL from start), 587 = SMTP+STARTTLS
        # iCloud 同时支持；某些网络封 587 → 试 465
        if SMTP_PORT == 465:
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=context, timeout=10) as server:
                server.login(SMTP_USER, SMTP_PASS)
                server.sendmail(SMTP_USER, NOTIFY_TO, msg.as_string())
        else:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
                server.ehlo()
                server.starttls(context=context)
                server.ehlo()
                server.login(SMTP_USER, SMTP_PASS)
                server.sendmail(SMTP_USER, NOTIFY_TO, msg.as_string())
        return True
    except (smtplib.SMTPException, OSError) as e:
        print(f"⚠️ SMTP 发信失败: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="us-stock-quant 通知工具")
    parser.add_argument("--level", choices=["info", "warn", "error"], default="info",
                        help="通知级别（影响邮件主题颜色）")
    parser.add_argument("--title", required=True, help="通知标题")
    parser.add_argument("--body", required=True, help="通知正文")
    parser.add_argument("--subtitle", default="", help="macOS 通知副标题")
    parser.add_argument("--stdout-log", default="", help="详细 log 文件路径（邮件用）")
    parser.add_argument("--no-macos", action="store_true", help="不发 macOS 通知")
    parser.add_argument("--no-smtp", action="store_true", help="不发邮件")
    args = parser.parse_args()

    print(f"[notify] level={args.level} title={args.title!r}")

    success_count = 0
    if not args.no_macos:
        if send_macos_notification(args.title, args.body, args.subtitle):
            success_count += 1
            print("  ✅ macOS 通知中心 OK")
    if not args.no_smtp:
        if send_smtp_email(args.title, args.body, args.level, args.stdout_log):
            success_count += 1
            print(f"  ✅ SMTP 邮件 OK (to={NOTIFY_TO})")
        elif SMTP_DISABLED:
            # 显式关掉了
            print("  ⏭️  SMTP 已禁用（设 SMTP_DISABLED=0 启用）")
        elif SMTP_USER and SMTP_PASS:
            # 配了但失败 → exit 1
            print("  ❌ SMTP 配了但发送失败", file=sys.stderr)
            sys.exit(1)
        else:
            print("  ⏭️  SMTP 未配（设置 SMTP_USER + SMTP_APP_PASSWORD 启用）")

    if success_count == 0:
        print("⚠️ 没有任何渠道成功", file=sys.stderr)
    sys.exit(0)


if __name__ == "__main__":
    main()
