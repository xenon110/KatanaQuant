"""
Reconciliation Service:
- Tracks T+1 cash settlement (Good Faith Violation prevention)
- Enforces PDT 5-day rolling window
- Reconciles broker state against internal books
"""
from datetime import datetime, date, timedelta, timezone
from typing import List, Dict, Tuple, Optional
import logging

from src.core.models import (
    AccountState,
    CashLedgerEntry,
    DayTradeRecord,
    Position,
    Order
)
from src.core.enums import AccountType, OrderSide

logger = logging.getLogger(__name__)


def calculate_t_plus_1_date(current_date: date) -> date:
    """
    Computes T+1 settlement date skipping weekends.
    (Mon->Tue, Tue->Wed, Wed->Thu, Thu->Fri, Fri->Mon).
    """
    next_day = current_date + timedelta(days=1)
    # If Saturday (5) -> advance to Monday
    if next_day.weekday() == 5:
        next_day += timedelta(days=2)
    # If Sunday (6) -> advance to Monday
    elif next_day.weekday() == 6:
        next_day += timedelta(days=1)
    return next_day


class ReconciliationService:
    def __init__(self):
        self.cash_ledger: List[CashLedgerEntry] = []
        self.day_trade_history: List[DayTradeRecord] = []
        self.intraday_fills: List[Tuple[str, OrderSide, int, datetime, str]] = [] # (symbol, side, qty, time, order_id)

    def record_trade_fill(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        price: float,
        order_id: str,
        trade_time: Optional[datetime] = None
    ) -> Optional[DayTradeRecord]:
        """
        Record a fill to update cash settlement and check for day-trade trigger.
        Returns a DayTradeRecord if a round-trip day trade was completed.
        """
        now = trade_time or datetime.now(timezone.utc)
        today = now.date()
        total_amount = quantity * price

        if side == OrderSide.SELL:
            # Sales create unsettled proceeds that settle on T+1
            settlement_dt = calculate_t_plus_1_date(today)
            entry = CashLedgerEntry(
                entry_id=f"settle_{order_id}",
                timestamp=now,
                settlement_date=settlement_dt,
                amount=total_amount,
                is_settled=False,
                source_order_id=order_id,
                description=f"Proceeds from selling {quantity} {symbol} (T+1 Settle: {settlement_dt})"
            )
            self.cash_ledger.append(entry)

        # Check for PDT day-trade match: buy followed by sell or sell followed by buy same day
        day_trade_record = None
        same_day_opposites = [
            f for f in self.intraday_fills
            if f[0] == symbol and f[1] != side and f[3].date() == today
        ]

        if same_day_opposites:
            matched_fill = same_day_opposites[0]
            day_trade_record = DayTradeRecord(
                record_id=f"dt_{order_id}_{matched_fill[4]}",
                symbol=symbol,
                trade_date=today,
                buy_order_id=order_id if side == OrderSide.BUY else matched_fill[4],
                sell_order_id=order_id if side == OrderSide.SELL else matched_fill[4],
                quantity=min(quantity, matched_fill[2]),
                realized_pnl=0.0 # Updated when calculating realized PnL
            )
            self.day_trade_history.append(day_trade_record)
            logger.warning(f"DAY TRADE DETECTED: {symbol} on {today}. Total recorded: {len(self.get_active_day_trades(today))}")

        self.intraday_fills.append((symbol, side, quantity, now, order_id))
        return day_trade_record

    def process_daily_settlements(self, current_date: date) -> float:
        """
        Transitions unsettled cash into settled cash if settlement_date <= current_date.
        Returns the total newly settled cash amount.
        """
        newly_settled = 0.0
        for entry in self.cash_ledger:
            if not entry.is_settled and entry.settlement_date <= current_date:
                entry.is_settled = True
                newly_settled += entry.amount
                logger.info(f"Settled ${entry.amount:.2f} for order {entry.source_order_id}")
        return newly_settled

    def get_unsettled_cash(self) -> float:
        return sum(entry.amount for entry in self.cash_ledger if not entry.is_settled)

    def get_active_day_trades(self, current_date: date) -> List[DayTradeRecord]:
        """
        Returns day trades completed within the rolling 5-business-day window.
        """
        cutoff_date = current_date - timedelta(days=7) # Approx 5 trading days
        return [dt for dt in self.day_trade_history if dt.trade_date >= cutoff_date]

    def reconcile_account(
        self,
        broker_account: AccountState,
        current_date: Optional[date] = None
    ) -> AccountState:
        """
        Reconciles broker account information with local T+1 settlement and PDT records.
        """
        today = current_date or datetime.now(timezone.utc).date()
        self.process_daily_settlements(today)

        active_dts = self.get_active_day_trades(today)
        unsettled = self.get_unsettled_cash()

        # Update account fields
        reconciled = broker_account.model_copy()
        reconciled.day_trade_count = len(active_dts)
        reconciled.unsettled_cash = unsettled

        if reconciled.account_type == AccountType.CASH:
            reconciled.settled_cash = max(0.0, reconciled.cash - unsettled)

        return reconciled
