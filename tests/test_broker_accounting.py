"""
Unit tests for Broker Accounting, Short Selling, Dynamic PnL, and Position Closing.
"""
import pytest
from src.core.models import ProposedTrade, RiskGateDecision
from src.core.enums import OrderSide, OrderType, AccountType, RiskGateStatus
from src.execution.dry_run_broker import DryRunBroker


@pytest.mark.asyncio
async def test_dry_run_long_position_lifecycle():
    broker = DryRunBroker(initial_equity=30000.0)

    # 1. Buy 100 AAPL @ $150
    trade_buy = ProposedTrade(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=100,
        price=150.0,
        order_type=OrderType.MARKET
    )
    decision = RiskGateDecision(
        approved=True,
        status=RiskGateStatus.APPROVED,
        original_quantity=100,
        allowed_quantity=100
    )
    await broker.submit_order(trade_buy, decision)

    positions = await broker.get_positions()
    assert len(positions) == 1
    assert positions[0].symbol == "AAPL"
    assert positions[0].quantity == 100
    assert positions[0].avg_entry_price == 150.0
    assert broker.cash == 15000.0  # 30,000 - 15,000

    # 2. Sell 50 AAPL @ $160 (Partial Take Profit)
    trade_sell = ProposedTrade(
        symbol="AAPL",
        side=OrderSide.SELL,
        quantity=50,
        price=160.0,
        order_type=OrderType.MARKET
    )
    decision_sell = RiskGateDecision(
        approved=True,
        status=RiskGateStatus.APPROVED,
        original_quantity=50,
        allowed_quantity=50
    )
    await broker.submit_order(trade_sell, decision_sell)

    positions = await broker.get_positions()
    assert positions[0].quantity == 50
    assert broker.realized_pnl == 500.0  # (160 - 150) * 50
    assert broker.cash == 23000.0  # 15,000 + (50 * 160) = 23,000

    account = await broker.get_account()
    assert account.equity == 31000.0  # 23,000 cash + (50 * 160)


@pytest.mark.asyncio
async def test_dry_run_short_position_lifecycle():
    broker = DryRunBroker(initial_equity=30000.0)

    # 1. Open Short: Sell 50 TSLA @ $200
    trade_short = ProposedTrade(
        symbol="TSLA",
        side=OrderSide.SELL,
        quantity=50,
        price=200.0,
        order_type=OrderType.MARKET
    )
    decision = RiskGateDecision(
        approved=True,
        status=RiskGateStatus.APPROVED,
        original_quantity=50,
        allowed_quantity=50
    )
    await broker.submit_order(trade_short, decision)

    positions = await broker.get_positions()
    assert len(positions) == 1
    assert positions[0].symbol == "TSLA"
    assert positions[0].quantity == -50
    assert positions[0].avg_entry_price == 200.0
    assert broker.cash == 40000.0  # 30,000 + 10,000 (short sale proceeds)

    # 2. Price drops to $180: Close short (Buy 50 TSLA @ $180)
    trade_cover = ProposedTrade(
        symbol="TSLA",
        side=OrderSide.BUY,
        quantity=50,
        price=180.0,
        order_type=OrderType.MARKET
    )
    await broker.submit_order(trade_cover, decision)

    positions = await broker.get_positions()
    assert len(positions) == 0  # Fully closed
    assert broker.realized_pnl == 1000.0  # (200 - 180) * 50
    assert broker.cash == 31000.0  # 40,000 - 9,000 = 31,000

    account = await broker.get_account()
    assert account.equity == 31000.0
    assert account.daily_realized_pnl == 1000.0
