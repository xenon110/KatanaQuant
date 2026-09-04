"""
Unit tests for the Deterministic Risk Gate.
Validates that hard risk boundaries are strictly enforced without exception.
"""
import pytest
from datetime import datetime, timezone

from src.config.settings import Settings
from src.core.enums import OrderSide, AccountType, RiskGateStatus
from src.core.models import ProposedTrade, AccountState, Position
from src.risk.gate import DeterministicRiskGate


@pytest.fixture
def risk_gate():
    config = Settings(
        MAX_POSITION_SIZE_USD=5000.0,
        MAX_POSITION_SIZE_PCT_NAV=0.10,
        MAX_CONCURRENT_POSITIONS=3,
        MAX_DAILY_LOSS_USD=500.0,
        MAX_DAILY_LOSS_PCT=0.02,
        MAX_STALENESS_SECONDS=60,
        ENFORCE_PDT=True
    )
    return DeterministicRiskGate(config)


@pytest.fixture
def margin_account():
    return AccountState(
        account_id="ACC_001",
        account_type=AccountType.MARGIN,
        equity=30000.0,
        cash=15000.0,
        settled_cash=15000.0,
        unsettled_cash=0.0,
        buying_power=60000.0,
        day_trade_count=0,
        starting_daily_equity=30000.0
    )


def test_approve_valid_trade(risk_gate, margin_account):
    trade = ProposedTrade(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=10,
        price=150.0 # Notional: $1,500 (< 5k cap and < 10% NAV = 3k)
    )
    decision = risk_gate.evaluate(trade, margin_account, current_positions=[])
    assert decision.approved is True
    assert decision.status == RiskGateStatus.APPROVED
    assert decision.allowed_quantity == 10


def test_position_size_capped(risk_gate, margin_account):
    # NAV = $30,000. 10% NAV = $3,000 max.
    # Proposed: 30 shares @ $150 = $4,500 -> Must be trimmed to 20 shares ($3,000)
    trade = ProposedTrade(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=30,
        price=150.0
    )
    decision = risk_gate.evaluate(trade, margin_account, current_positions=[])
    assert decision.approved is True
    assert decision.status == RiskGateStatus.MODIFIED
    assert decision.allowed_quantity == 20


def test_daily_loss_limit_breach(risk_gate, margin_account):
    # Set daily realized loss to -$600 (exceeds $500 limit)
    margin_account.daily_realized_pnl = -600.0
    trade = ProposedTrade(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=10,
        price=150.0
    )
    decision = risk_gate.evaluate(trade, margin_account, current_positions=[])
    assert decision.approved is False
    assert decision.status == RiskGateStatus.REJECTED
    assert "MAX_DAILY_LOSS_USD_BREACHED" in decision.rule_violations


def test_max_concurrent_positions_limit(risk_gate, margin_account):
    # Max allowed concurrent is 3
    now = datetime.now(timezone.utc)
    existing_positions = [
        Position(symbol="AAPL", quantity=10, avg_entry_price=150.0, current_price=150.0, market_value=1500.0, unrealized_pnl=0.0, unrealized_pnl_pct=0.0, updated_at=now),
        Position(symbol="MSFT", quantity=10, avg_entry_price=300.0, current_price=300.0, market_value=3000.0, unrealized_pnl=0.0, unrealized_pnl_pct=0.0, updated_at=now),
        Position(symbol="NVDA", quantity=10, avg_entry_price=120.0, current_price=120.0, market_value=1200.0, unrealized_pnl=0.0, unrealized_pnl_pct=0.0, updated_at=now),
    ]
    # Attempting to buy a 4th new symbol (SPY)
    trade = ProposedTrade(
        symbol="SPY",
        side=OrderSide.BUY,
        quantity=5,
        price=400.0
    )
    decision = risk_gate.evaluate(trade, margin_account, current_positions=existing_positions)
    assert decision.approved is False
    assert "MAX_CONCURRENT_POSITIONS_REACHED" in decision.rule_violations


def test_pdt_limit_for_under_25k_margin_account(risk_gate):
    small_account = AccountState(
        account_id="ACC_SMALL",
        account_type=AccountType.MARGIN,
        equity=20000.0, # Under $25k
        cash=10000.0,
        settled_cash=10000.0,
        unsettled_cash=0.0,
        buying_power=20000.0,
        day_trade_count=3, # Already at 3 day trades
        starting_daily_equity=20000.0
    )
    trade = ProposedTrade(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=5,
        price=150.0
    )
    decision = risk_gate.evaluate(trade, small_account, current_positions=[], is_day_trade=True)
    assert decision.approved is False
    assert "PDT_LIMIT_REACHED" in decision.rule_violations


def test_cash_account_settled_cash_check(risk_gate):
    cash_account = AccountState(
        account_id="ACC_CASH",
        account_type=AccountType.CASH,
        equity=10000.0,
        cash=10000.0,
        settled_cash=500.0,    # Only $500 settled cash available
        unsettled_cash=9500.0,
        buying_power=500.0,
        day_trade_count=0,
        starting_daily_equity=10000.0
    )
    # Attempting to buy $1,500 worth of shares (exceeds $500 settled cash)
    trade = ProposedTrade(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=10,
        price=150.0
    )
    decision = risk_gate.evaluate(trade, cash_account, current_positions=[])
    assert decision.approved is False
    assert "INSUFFICIENT_SETTLED_CASH" in decision.rule_violations


def test_stale_data_circuit_breaker(risk_gate, margin_account):
    trade = ProposedTrade(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=10,
        price=150.0
    )
    decision = risk_gate.evaluate(
        trade,
        margin_account,
        current_positions=[],
        data_staleness_seconds=95.0 # Exceeds 60s threshold
    )
    assert decision.approved is False
    assert "DATA_FEED_STALE" in decision.rule_violations


def test_manual_kill_switch_trip(risk_gate, margin_account):
    risk_gate.trip_circuit_breaker("Emergency Operator Intervention")
    trade = ProposedTrade(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=10,
        price=150.0
    )
    decision = risk_gate.evaluate(trade, margin_account, current_positions=[])
    assert decision.approved is False
    assert "CIRCUIT_BREAKER_ACTIVE" in decision.rule_violations


def test_buy_to_cover_short_position_always_allowed(risk_gate, margin_account):
    now = datetime.now(timezone.utc)
    # 3 positions already open (at max concurrent limit), including a short in TSLA
    existing = [
        Position(symbol="AAPL", quantity=10, avg_entry_price=150.0, current_price=150.0, market_value=1500.0, unrealized_pnl=0.0, unrealized_pnl_pct=0.0, updated_at=now),
        Position(symbol="MSFT", quantity=10, avg_entry_price=300.0, current_price=300.0, market_value=3000.0, unrealized_pnl=0.0, unrealized_pnl_pct=0.0, updated_at=now),
        Position(symbol="TSLA", quantity=-20, avg_entry_price=200.0, current_price=200.0, market_value=-4000.0, unrealized_pnl=0.0, unrealized_pnl_pct=0.0, updated_at=now),
    ]
    # Buy 20 TSLA to cover short position
    trade_cover = ProposedTrade(
        symbol="TSLA",
        side=OrderSide.BUY,
        quantity=20,
        price=190.0
    )
    decision = risk_gate.evaluate(trade_cover, margin_account, current_positions=existing)
    # Must be approved as risk-reducing without being blocked by max concurrent positions
    assert decision.approved is True
    assert decision.status == RiskGateStatus.APPROVED
    assert decision.allowed_quantity == 20


def test_cash_account_blocks_naked_short_selling(risk_gate):
    cash_account = AccountState(
        account_id="ACC_CASH",
        account_type=AccountType.CASH,
        equity=30000.0,
        cash=30000.0,
        settled_cash=30000.0,
        unsettled_cash=0.0,
        buying_power=30000.0,
        day_trade_count=0,
        starting_daily_equity=30000.0
    )
    # Sell AAPL with 0 shares owned in cash account
    trade_short = ProposedTrade(
        symbol="AAPL",
        side=OrderSide.SELL,
        quantity=10,
        price=150.0
    )
    decision = risk_gate.evaluate(trade_short, cash_account, current_positions=[])
    assert decision.approved is False
    assert "INSUFFICIENT_SHARES_TO_SELL" in decision.rule_violations

