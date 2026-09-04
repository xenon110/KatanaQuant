"""
Alpaca Broker Implementation.
Routes approved orders to Alpaca Trading API (Paper / Live) using alpaca-py.
"""
from datetime import datetime, timezone
from typing import List, Optional
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
    OrderType,
    TimeInForce,
    AccountType
)
from src.execution.broker_client import BaseBrokerClient

logger = logging.getLogger(__name__)


class AlpacaBrokerClient(BaseBrokerClient):
    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper
        self._trading_client = None
        self.orders = {}

    def _get_client(self):
        if self._trading_client is None:
            try:
                from alpaca.trading.client import TradingClient
                self._trading_client = TradingClient(
                    api_key=self.api_key,
                    secret_key=self.secret_key,
                    paper=self.paper
                )
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Alpaca Trading Client: {e}")
        return self._trading_client

    async def get_account(self) -> AccountState:
        client = self._get_client()
        acc = client.get_account()

        equity = float(acc.equity or 0.0)
        cash = float(acc.cash or 0.0)
        buying_power = float(acc.buying_power or 0.0)
        day_trade_count = int(acc.daytrade_count or 0)
        is_pdt = bool(acc.pattern_day_trader or False)
        is_blocked = bool(acc.trading_blocked or False)

        # Multiplier "1" indicates Cash account, "2" or "4" indicates Margin account
        multiplier = getattr(acc, "multiplier", "2")
        acc_type = AccountType.CASH if str(multiplier) == "1" else AccountType.MARGIN

        # Starting daily equity (last_equity)
        starting_equity = float(acc.last_equity) if (hasattr(acc, "last_equity") and acc.last_equity is not None) else equity
        daily_pnl = equity - starting_equity

        return AccountState(
            account_id=str(acc.id),
            account_type=acc_type,
            equity=equity,
            cash=cash,
            settled_cash=cash, # Will be adjusted by local reconciliation ledger
            unsettled_cash=0.0,
            buying_power=buying_power,
            day_trade_count=day_trade_count,
            is_pdt=is_pdt,
            is_trading_blocked=is_blocked,
            daily_realized_pnl=daily_pnl,
            daily_unrealized_pnl=0.0,
            starting_daily_equity=starting_equity,
            last_updated=datetime.now(timezone.utc)
        )

    async def get_positions(self) -> List[Position]:
        client = self._get_client()
        alpaca_positions = client.get_all_positions()
        positions: List[Position] = []

        for p in alpaca_positions:
            positions.append(Position(
                symbol=p.symbol,
                quantity=int(p.qty),
                avg_entry_price=float(p.avg_entry_price),
                current_price=float(p.current_price),
                market_value=float(p.market_value),
                unrealized_pnl=float(p.unrealized_pl),
                unrealized_pnl_pct=float(p.unrealized_plpc),
                updated_at=datetime.now(timezone.utc)
            ))
        return positions

    async def submit_order(
        self,
        trade: ProposedTrade,
        decision: RiskGateDecision
    ) -> Order:
        from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
        from alpaca.trading.enums import OrderSide as AlpacaSide, TimeInForce as AlpacaTIF

        if not decision.approved:
            raise ValueError(f"Cannot submit unapproved order: {decision.rejection_reasons}")

        client = self._get_client()
        effective_qty = decision.allowed_quantity
        side_mapped = AlpacaSide.BUY if trade.side == OrderSide.BUY else AlpacaSide.SELL

        if trade.order_type == OrderType.MARKET:
            order_data = MarketOrderRequest(
                symbol=trade.symbol,
                qty=effective_qty,
                side=side_mapped,
                time_in_force=AlpacaTIF.DAY
            )
        else:
            order_data = LimitOrderRequest(
                symbol=trade.symbol,
                qty=effective_qty,
                side=side_mapped,
                limit_price=trade.limit_price or trade.price,
                time_in_force=AlpacaTIF.DAY
            )

        logger.info(f"Submitting order to Alpaca: {trade.side.value} {effective_qty} {trade.symbol}")
        submitted = client.submit_order(order_data=order_data)

        ord_obj = Order(
            order_id=str(submitted.id),
            client_order_id=str(submitted.client_order_id or submitted.id),
            symbol=submitted.symbol,
            side=trade.side,
            quantity=int(submitted.qty),
            order_type=trade.order_type,
            time_in_force=trade.time_in_force,
            status=OrderStatus.SUBMITTED,
            submitted_at=submitted.submitted_at.replace(tzinfo=None) if submitted.submitted_at else datetime.now(timezone.utc)
        )
        self.orders[ord_obj.order_id] = ord_obj
        return ord_obj

    async def submit_bracket_order(
        self,
        symbol: str,
        quantity: int,
        side: OrderSide,
        take_profit_price: float,
        stop_loss_price: float,
        decision: RiskGateDecision
    ) -> Order:
        """
        Submits an institutional-grade OTO Bracket Order to Alpaca.
        Automatically attaches Take-Profit Limit and Stop-Loss Stop orders.
        """
        from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
        from alpaca.trading.enums import OrderSide as AlpacaSide, TimeInForce as AlpacaTIF, OrderClass

        if not decision.approved:
            raise ValueError(f"Cannot submit unapproved order: {decision.rejection_reasons}")

        client = self._get_client()
        effective_qty = decision.allowed_quantity
        side_mapped = AlpacaSide.BUY if side == OrderSide.BUY else AlpacaSide.SELL

        tp_req = TakeProfitRequest(limit_price=round(take_profit_price, 2))
        sl_req = StopLossRequest(stop_price=round(stop_loss_price, 2))

        order_data = MarketOrderRequest(
            symbol=symbol,
            qty=effective_qty,
            side=side_mapped,
            time_in_force=AlpacaTIF.DAY,
            order_class=OrderClass.BRACKET,
            take_profit=tp_req,
            stop_loss=sl_req
        )

        logger.info(f"Submitting BRACKET order to Alpaca: {side.value} {effective_qty} {symbol} (TP: ${take_profit_price:.2f}, SL: ${stop_loss_price:.2f})")
        submitted = client.submit_order(order_data=order_data)

        ord_obj = Order(
            order_id=str(submitted.id),
            client_order_id=str(submitted.client_order_id or submitted.id),
            symbol=submitted.symbol,
            side=side,
            quantity=int(submitted.qty),
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,
            status=OrderStatus.SUBMITTED,
            submitted_at=submitted.submitted_at.replace(tzinfo=None) if submitted.submitted_at else datetime.now(timezone.utc)
        )
        self.orders[ord_obj.order_id] = ord_obj
        return ord_obj

    async def cancel_order(self, order_id: str) -> bool:
        client = self._get_client()
        try:
            client.cancel_order_by_id(order_id)
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    async def cancel_all_orders(self) -> int:
        client = self._get_client()
        res = client.cancel_orders()
        return len(res)

