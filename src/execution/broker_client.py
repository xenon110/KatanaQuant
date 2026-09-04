"""
Abstract Broker Client Interface.
Standardizes order routing and account state fetching across Dry-Run, Paper, and Live modes.
"""
from abc import ABC, abstractmethod
from typing import List, Optional
from src.core.models import Order, AccountState, Position, ProposedTrade, RiskGateDecision


class BaseBrokerClient(ABC):
    @abstractmethod
    async def get_account(self) -> AccountState:
        """Fetch real-time account state (equity, cash, buying power, PDT status)."""
        pass

    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """Fetch all currently open positions."""
        pass

    @abstractmethod
    async def submit_order(
        self,
        trade: ProposedTrade,
        decision: RiskGateDecision
    ) -> Order:
        """
        Submit an approved or modified order to the broker.
        Throws error if decision is not approved.
        """
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending open order."""
        pass

    @abstractmethod
    async def cancel_all_orders(self) -> int:
        """Cancel all open orders (used by Emergency Kill Switch)."""
        pass
