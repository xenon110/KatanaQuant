"""
Dry-Run / Simulation Broker.
Logs intended orders and maintains virtual state without routing real funds.
"""
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Optional
import logging

from src.core.models import (
    Order,
    AccountState,
    Position,
    ProposedTrade,
    RiskGateDecision
)
from src.core.enums import (
    OrderStatus,
    OrderSide,
    AccountType,
    RiskGateStatus
)
from src.execution.broker_client import BaseBrokerClient

logger = logging.getLogger(__name__)


class DryRunBroker(BaseBrokerClient):
    def __init__(
        self,
        initial_equity: float = 30000.0,
        account_type: AccountType = AccountType.MARGIN
    ):
        self.account_type = account_type
        self.initial_equity = initial_equity
        self.cash = initial_equity
        self.settled_cash = initial_equity
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.realized_pnl = 0.0

    async def get_account(self) -> AccountState:
        # Dynamically recompute position market values and unrealized PnL
        unrealized_pnl = 0.0
        positions_market_val = 0.0
        for pos in self.positions.values():
            if pos.quantity > 0:
                pos.unrealized_pnl = (pos.current_price - pos.avg_entry_price) * pos.quantity
                pos.unrealized_pnl_pct = (pos.current_price - pos.avg_entry_price) / pos.avg_entry_price if pos.avg_entry_price > 0 else 0.0
                pos.market_value = pos.quantity * pos.current_price
            elif pos.quantity < 0:
                pos.unrealized_pnl = (pos.avg_entry_price - pos.current_price) * abs(pos.quantity)
                pos.unrealized_pnl_pct = (pos.avg_entry_price - pos.current_price) / pos.avg_entry_price if pos.avg_entry_price > 0 else 0.0
                pos.market_value = pos.quantity * pos.current_price
            unrealized_pnl += pos.unrealized_pnl
            positions_market_val += pos.market_value

        equity = self.cash + positions_market_val
        buying_power = max(0.0, equity * 2.0 if self.account_type == AccountType.MARGIN else self.settled_cash)

        return AccountState(
            account_id="DRY_RUN_ACC_001",
            account_type=self.account_type,
            equity=round(equity, 2),
            cash=round(self.cash, 2),
            settled_cash=round(self.settled_cash, 2),
            unsettled_cash=0.0,
            buying_power=round(buying_power, 2),
            day_trade_count=0,
            is_pdt=False,
            is_trading_blocked=False,
            daily_realized_pnl=round(self.realized_pnl, 2),
            daily_unrealized_pnl=round(unrealized_pnl, 2),
            starting_daily_equity=round(self.initial_equity, 2),
            last_updated=datetime.now(timezone.utc)
        )

    async def get_positions(self) -> List[Position]:
        for pos in self.positions.values():
            if pos.quantity > 0:
                pos.unrealized_pnl = round((pos.current_price - pos.avg_entry_price) * pos.quantity, 2)
                pos.unrealized_pnl_pct = (pos.current_price - pos.avg_entry_price) / pos.avg_entry_price if pos.avg_entry_price > 0 else 0.0
                pos.market_value = round(pos.quantity * pos.current_price, 2)
            elif pos.quantity < 0:
                pos.unrealized_pnl = round((pos.avg_entry_price - pos.current_price) * abs(pos.quantity), 2)
                pos.unrealized_pnl_pct = (pos.avg_entry_price - pos.current_price) / pos.avg_entry_price if pos.avg_entry_price > 0 else 0.0
                pos.market_value = round(pos.quantity * pos.current_price, 2)
        return list(self.positions.values())

    async def submit_order(
        self,
        trade: ProposedTrade,
        decision: RiskGateDecision
    ) -> Order:
        if not decision.approved:
            raise ValueError(f"Cannot submit unapproved trade: {decision.rejection_reasons}")

        effective_qty = decision.allowed_quantity
        order_id = f"dry_{uuid.uuid4().hex[:12]}"
        now_dt = datetime.now(timezone.utc)
        client_id = f"cli_{trade.symbol}_{int(now_dt.timestamp())}"

        order = Order(
            order_id=order_id,
            client_order_id=client_id,
            symbol=trade.symbol,
            side=trade.side,
            quantity=effective_qty,
            filled_quantity=effective_qty,  # Simulate instantaneous fill in dry run
            order_type=trade.order_type,
            time_in_force=trade.time_in_force,
            status=OrderStatus.FILLED,
            submitted_at=now_dt,
            filled_at=now_dt,
            avg_fill_price=trade.price
        )
        self.orders[order_id] = order

        fill_val = effective_qty * trade.price
        existing = self.positions.get(trade.symbol)

        if trade.side == OrderSide.BUY:
            if existing and existing.quantity < 0:
                # Closing or reducing a SHORT position
                short_qty = abs(existing.quantity)
                closing_qty = min(short_qty, effective_qty)
                trade_pnl = (existing.avg_entry_price - trade.price) * closing_qty
                self.realized_pnl += trade_pnl
                self.cash -= closing_qty * trade.price
                self.settled_cash -= closing_qty * trade.price
                remaining_short = short_qty - closing_qty

                if remaining_short == 0:
                    del self.positions[trade.symbol]
                    # If remaining buy quantity exceeds short, flip to LONG
                    rem_buy = effective_qty - closing_qty
                    if rem_buy > 0:
                        self.cash -= rem_buy * trade.price
                        self.settled_cash -= rem_buy * trade.price
                        self.positions[trade.symbol] = Position(
                            symbol=trade.symbol,
                            quantity=rem_buy,
                            avg_entry_price=trade.price,
                            current_price=trade.price,
                            market_value=rem_buy * trade.price,
                            unrealized_pnl=0.0,
                            unrealized_pnl_pct=0.0,
                            updated_at=now_dt
                        )
                else:
                    existing.quantity = -remaining_short
                    existing.current_price = trade.price
                    existing.market_value = -remaining_short * trade.price
                    existing.updated_at = now_dt
            else:
                # Increasing or opening a LONG position
                self.cash -= fill_val
                self.settled_cash -= fill_val
                if existing:
                    new_qty = existing.quantity + effective_qty
                    avg_p = ((existing.quantity * existing.avg_entry_price) + fill_val) / new_qty
                    existing.quantity = new_qty
                    existing.avg_entry_price = avg_p
                    existing.current_price = trade.price
                    existing.market_value = new_qty * trade.price
                    existing.updated_at = now_dt
                else:
                    self.positions[trade.symbol] = Position(
                        symbol=trade.symbol,
                        quantity=effective_qty,
                        avg_entry_price=trade.price,
                        current_price=trade.price,
                        market_value=fill_val,
                        unrealized_pnl=0.0,
                        unrealized_pnl_pct=0.0,
                        updated_at=now_dt
                    )
        elif trade.side == OrderSide.SELL:
            if existing and existing.quantity > 0:
                # Closing or reducing a LONG position
                long_qty = existing.quantity
                closing_qty = min(long_qty, effective_qty)
                trade_pnl = (trade.price - existing.avg_entry_price) * closing_qty
                self.realized_pnl += trade_pnl
                self.cash += closing_qty * trade.price
                remaining_long = long_qty - closing_qty

                if remaining_long == 0:
                    del self.positions[trade.symbol]
                    # If remaining sell quantity exceeds long, flip to SHORT
                    rem_short = effective_qty - closing_qty
                    if rem_short > 0:
                        self.cash += rem_short * trade.price
                        self.positions[trade.symbol] = Position(
                            symbol=trade.symbol,
                            quantity=-rem_short,
                            avg_entry_price=trade.price,
                            current_price=trade.price,
                            market_value=-rem_short * trade.price,
                            unrealized_pnl=0.0,
                            unrealized_pnl_pct=0.0,
                            updated_at=now_dt
                        )
                else:
                    existing.quantity = remaining_long
                    existing.current_price = trade.price
                    existing.market_value = remaining_long * trade.price
                    existing.updated_at = now_dt
            else:
                # Opening or increasing a SHORT position
                self.cash += fill_val
                if existing:
                    short_qty = abs(existing.quantity)
                    new_short_qty = short_qty + effective_qty
                    avg_p = ((short_qty * existing.avg_entry_price) + fill_val) / new_short_qty
                    existing.quantity = -new_short_qty
                    existing.avg_entry_price = avg_p
                    existing.current_price = trade.price
                    existing.market_value = -new_short_qty * trade.price
                    existing.updated_at = now_dt
                else:
                    self.positions[trade.symbol] = Position(
                        symbol=trade.symbol,
                        quantity=-effective_qty,
                        avg_entry_price=trade.price,
                        current_price=trade.price,
                        market_value=-fill_val,
                        unrealized_pnl=0.0,
                        unrealized_pnl_pct=0.0,
                        updated_at=now_dt
                    )

        logger.info(
            f"[DRY-RUN EXECUTION] Filled {order.side.value} {order.quantity} {order.symbol} @ ${order.avg_fill_price:.2f}"
        )
        return order

    async def cancel_order(self, order_id: str) -> bool:
        if order_id in self.orders:
            self.orders[order_id].status = OrderStatus.CANCELLED
            return True
        return False

    async def cancel_all_orders(self) -> int:
        count = 0
        for o in self.orders.values():
            if o.status in (OrderStatus.SUBMITTED, OrderStatus.APPROVED):
                o.status = OrderStatus.CANCELLED
                count += 1
        return count
