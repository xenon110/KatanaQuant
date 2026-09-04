"""
Telegram Notification Dispatcher for Real-Time Trade & Risk Alerts.
"""
import logging
from typing import Optional
import httpx

from src.config.settings import settings

logger = logging.getLogger(__name__)


class TelegramNotifier:
    def __init__(self, bot_token: Optional[str] = None, chat_id: Optional[str] = None):
        self.bot_token = bot_token or settings.telegram_bot_token
        self.chat_id = chat_id or settings.telegram_chat_id

    @property
    def is_configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    async def send_message(self, text: str) -> bool:
        if not self.is_configured:
            return False

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "Markdown"
        }

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                res = await client.post(url, json=payload)
                return res.status_code == 200
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")
            return False

    async def notify_order_filled(self, symbol: str, side: str, qty: int, price: float, broker: str = "Alpaca"):
        msg = (
            f"🚀 *TRADE FILLED ({broker})*\n\n"
            f"• *Symbol:* `{symbol}`\n"
            f"• *Side:* `{side}`\n"
            f"• *Quantity:* `{qty} shares`\n"
            f"• *Fill Price:* `${price:.2f}`\n"
            f"• *Est. Value:* `${(qty * price):,.2f}`\n"
            f"• *Status:* `CONFIRMED`"
        )
        return await self.send_message(msg)

    async def notify_risk_rejection(self, symbol: str, reasons: list):
        reasons_str = "\\n".join([f"  - {r}" for r in reasons])
        msg = (
            f"🛑 *RISK GATE REJECTION*\n\n"
            f"• *Symbol:* `{symbol}`\n"
            f"• *Blocked By:* `Deterministic Risk Gate`\n"
            f"• *Reasons:*\n{reasons_str}"
        )
        return await self.send_message(msg)

    async def notify_kill_switch(self, reason: str):
        msg = (
            f"🚨 *CIRCUIT BREAKER TRIGGERED*\n\n"
            f"• *Emergency Action:* `ALL TRADING HALTED`\n"
            f"• *Reason:* `{reason}`"
        )
        return await self.send_message(msg)
