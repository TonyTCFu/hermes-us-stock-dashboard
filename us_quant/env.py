"""統一讀取環境變數的 helper。

設計目標：
1. 本地開發時讀 .env（如果存在）
2. Render 部署時直接讀環境變數
3. 移除 .env 依賴後，所有敏感值只能從環境變數讀
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# 嘗試載入 .env（本地開發用，Render 沒有這個檔）
try:
    from dotenv import load_dotenv
    _ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
    if _ENV_FILE.exists():
        load_dotenv(_ENV_FILE)
except ImportError:
    pass  # dotenv 不存在就跳過


def get_env(key: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    """讀環境變數，找不到且 required=True 就 raise。

    Parameters
    ----------
    key : str
        環境變數名稱
    default : str, optional
        找不到時的預設值
    required : bool
        True 表示必須存在，否則 raise ValueError
    """
    val = os.getenv(key, default)
    if required and (val is None or val == ""):
        raise ValueError(
            f"❌ 缺少環境變數 {key}\n"
            f"   本地開發：在 .env 設定\n"
            f"   Render 部署：在 dashboard 後台 Environment 設定"
        )
    return val


def get_env_float(key: str, default: float) -> float:
    val = os.getenv(key)
    if val is None or val == "":
        return default
    try:
        return float(val)
    except ValueError:
        return default


def get_env_int(key: str, default: int) -> int:
    val = os.getenv(key)
    if val is None or val == "":
        return default
    try:
        return int(val)
    except ValueError:
        return default


def get_env_bool(key: str, default: bool = False) -> bool:
    """讀布林，'true'/'1'/'yes' → True，其他 → False。"""
    val = os.getenv(key, "").strip().lower()
    if val in ("true", "1", "yes", "on"):
        return True
    if val in ("false", "0", "no", "off"):
        return False
    return default