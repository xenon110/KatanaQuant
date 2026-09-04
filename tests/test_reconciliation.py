"""
Unit tests for the Reconciliation Service:
Validates T+1 settlement calendar math, GFV prevention, and PDT 5-day rolling tracking.
"""
from datetime import datetime, date, timedelta
from src.core.enums import OrderSide, AccountType
from src.core.models import AccountState
from src.reconciliation.service import ReconciliationService, calculate_t_plus_1_date


def test_t_plus_1_calendar_calculation():
    # Thursday -> Friday
    thu = date(2025, 1, 9)
    assert calculate_t_plus_1_date(thu) == date(2025, 1, 10)

    # Friday -> Monday (skips Sat & Sun)
    fri = date(2025, 1, 10)
    assert calculate_t_plus_1_date(fri) == date(2025, 1, 13)


def test_unsettled_cash_tracking_and_daily_settlement():
    reconciler = ReconciliationService()
    
    # Sell on Thursday Jan 9, 2025: 100 shares @ $50 = $5,000 proceeds
    trade_dt = datetime(2025, 1, 9, 14, 30)
    reconciler.record_trade_fill(
        symbol="AAPL",
        side=OrderSide.SELL,
        quantity=100,
        price=50.0,
        order_id="ord_sell_01",
        trade_time=trade_dt
    )

    # Thursday: Cash is unsettled
    assert reconciler.get_unsettled_cash() == 5000.0
    
    # Process settlement on Thursday (Not yet T+1)
    settled = reconciler.process_daily_settlements(date(2025, 1, 9))
    assert settled == 0.0
    assert reconciler.get_unsettled_cash() == 5000.0

    # Friday Jan 10 (T+1 reached): Proceeds settle
    settled = reconciler.process_daily_settlements(date(2025, 1, 10))
    assert settled == 5000.0
    assert reconciler.get_unsettled_cash() == 0.0


def test_pdt_same_day_round_trip_detection():
    reconciler = ReconciliationService()
    now = datetime(2025, 1, 9, 10, 0)
    
    # Buy 50 AAPL in morning
    reconciler.record_trade_fill(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=50,
        price=150.0,
        order_id="ord_buy_1",
        trade_time=now
    )

    # Sell 50 AAPL in afternoon
    dt_record = reconciler.record_trade_fill(
        symbol="AAPL",
        side=OrderSide.SELL,
        quantity=50,
        price=152.0,
        order_id="ord_sell_1",
        trade_time=now + timedelta(hours=3)
    )

    assert dt_record is not None
    assert dt_record.symbol == "AAPL"
    active_dts = reconciler.get_active_day_trades(now.date())
    assert len(active_dts) == 1


def test_reconcile_account_with_broker():
    reconciler = ReconciliationService()
    today = date(2025, 1, 10)

    # Unsettled sell
    reconciler.record_trade_fill(
        symbol="MSFT",
        side=OrderSide.SELL,
        quantity=20,
        price=200.0,
        order_id="ord_sell_msft",
        trade_time=datetime(2025, 1, 10, 11, 0)
    )

    broker_acc = AccountState(
        account_id="ACC_BROKER",
        account_type=AccountType.CASH,
        equity=20000.0,
        cash=20000.0,
        settled_cash=20000.0,
        unsettled_cash=0.0,
        buying_power=20000.0,
        day_trade_count=0,
        starting_daily_equity=20000.0
    )

    reconciled = reconciler.reconcile_account(broker_acc, current_date=today)
    assert reconciled.unsettled_cash == 4000.0
    assert reconciled.settled_cash == 16000.0
