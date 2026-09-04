"""
Supabase Cloud Database Client.
Provides cloud persistence and real-time syncing for orders, trade audit logs,
signals, and risk decisions via Supabase PostgREST API.
"""
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import httpx

from src.config.settings import settings

logger = logging.getLogger(__name__)


class SupabaseManager:
    def __init__(
        self,
        url: Optional[str] = None,
        key: Optional[str] = None
    ):
        self.url = (url or settings.supabase_url or "").rstrip("/")
        self.key = key or settings.supabase_service_role_key or settings.supabase_anon_key

    @property
    def is_configured(self) -> bool:
        return bool(self.url and self.key)

    def _headers(self) -> Dict[str, str]:
        return {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation"
        }

    async def insert_record(self, table: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Inserts a record into the specified Supabase table."""
        if not self.is_configured:
            return None

        endpoint = f"{self.url}/rest/v1/{table}"
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                res = await client.post(endpoint, headers=self._headers(), json=data)
                if res.status_code in (200, 201):
                    return res.json()
                else:
                    logger.debug(f"Supabase write to {table} (status {res.status_code}): {res.text}")
                    return None
        except Exception as e:
            logger.debug(f"Supabase connection notice: {e}")
            return None

    async def log_order(
        self,
        order_id: str,
        client_order_id: str,
        symbol: str,
        side: str,
        quantity: int,
        price: Optional[float] = None,
        status: str = "SUBMITTED",
        order_type: str = "MARKET"
    ):
        payload = {
            "order_id": order_id,
            "client_order_id": client_order_id,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "fill_price": price,
            "status": status,
            "order_type": order_type,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        return await self.insert_record("orders", payload)

    async def record_order(self, data: Dict[str, Any]):
        """Helper to insert order payload into Supabase orders table."""
        return await self.insert_record("orders", data)

    async def log_signal(
        self,
        symbol: str,
        direction: str,
        confidence: float,
        reasoning: str
    ):
        payload = {
            "symbol": symbol,
            "direction": direction,
            "confidence": confidence,
            "reasoning": reasoning,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        return await self.insert_record("signal_logs", payload)

    async def log_risk_decision(
        self,
        symbol: str,
        approved: bool,
        allowed_quantity: int,
        violations: List[str],
        reasons: List[str]
    ):
        payload = {
            "symbol": symbol,
            "approved": approved,
            "allowed_quantity": allowed_quantity,
            "violations": violations,
            "reasons": reasons,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        return await self.insert_record("risk_decision_logs", payload)
