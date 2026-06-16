"""Alpaca Broker 整合：查帳戶、持倉、下單"""
from __future__ import annotations

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import OrderSide, QueryOrderStatus

logger = logging.getLogger(__name__)


class AlpacaBroker:
    """Alpaca Paper Trading API 封裝。"""

    def __init__(self):
        key = os.getenv("ALPACA_API_KEY")
        secret = os.getenv("ALPACA_SECRET_KEY")
        if not key or not secret:
            raise ValueError("請在 .env 設定 ALPACA_API_KEY 和 ALPACA_SECRET_KEY")
        self.client = TradingClient(key, secret, paper=True)

    def get_account(self) -> dict:
        """取得帳戶摘要。"""
        acc = self.client.get_account()
        return {
            "cash": float(acc.cash),
            "equity": float(acc.equity),
            "buying_power": float(acc.buying_power),
            "status": acc.status.value,
            "day_change_pct": (float(acc.equity) - float(acc.last_equity)) / float(acc.last_equity) if float(acc.last_equity) > 0 else 0,
        }

    def get_positions(self) -> list[dict]:
        """取得所有持倉明細。"""
        positions = self.client.get_all_positions()
        result = []
        for p in positions:
            market_value = float(p.market_value)
            cost_basis = float(p.cost_basis)
            result.append({
                "ticker": p.symbol,
                "qty": float(p.qty),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "cost_basis": cost_basis,
                "market_value": market_value,
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_pl_pct": float(p.unrealized_plpc) * 100,
                "day_change": float(p.change_today),
                "day_change_pct": float(p.unrealized_intraday_plpc) * 100 if hasattr(p, 'unrealized_intraday_plpc') and p.unrealized_intraday_plpc else 0,
            })
        return result

    def get_open_orders(self) -> list[dict]:
        """取得所有未成交訂單。"""
        orders = self.client.get_orders()
        open_orders = [o for o in orders if o.status.value == "open"]
        return [
            {
                "id": o.id,
                "ticker": o.symbol,
                "side": o.side.value,
                "qty": float(o.qty),
                "type": o.type.value,
                "limit_price": float(o.limit_price) if o.limit_price else None,
                "status": o.status.value,
                "created_at": str(o.created_at)[:19],
            }
            for o in open_orders
        ]
