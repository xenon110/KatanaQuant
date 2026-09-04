"""
Tests for 100% Hands-Free Autonomous Trader Engine.
Verifies auto-buy on quantitative signals and auto-sell on take-profit/stop-loss.
"""
import pytest
from datetime import datetime, timezone
import pandas as pd

from src.execution.autonomous_trader import AutonomousTrader
from src.execution.dry_run_broker import DryRunBroker
from src.risk.gate import DeterministicRiskGate
from src.config.settings import settings
from src.orchestrator.pipeline import TradingOrchestrator
from src.strategy.rules import EMACrossRSIStrategy
from src.reconciliation.service import ReconciliationService
from src.data.market_data import SyntheticMarketDataProvider, BarCache
from src.notifications.telegram import TelegramNotifier
from src.storage.supabase_client import SupabaseManager
from src.core.enums import OrderSide, OrderType
from src.core.models import MarketBar, Position


@pytest.mark.asyncio
async def test_autonomous_trader_auto_exit_take_profit():
    broker = DryRunBroker(initial_equity=30000.0)
    risk_gate = DeterministicRiskGate(settings)
    strategy = EMACrossRSIStrategy()
    orchestrator = TradingOrchestrator(strategy=strategy, risk_gate=risk_gate)
    reconciler = ReconciliationService()
    data_provider = SyntheticMarketDataProvider(seed_price=106.0)
    bar_cache = BarCache()
    telegram = TelegramNotifier()
    supabase = SupabaseManager()

    auto_trader = AutonomousTrader(
        broker=broker,
        orchestrator=orchestrator,
        risk_gate=risk_gate,
        reconciler=reconciler,
        data_provider=data_provider,
        bar_cache=bar_cache,
        telegram=telegram,
        supabase=supabase
    )

    # Manually add an open position in broker: bought at $100.00
    broker.positions["AAPL"] = Position(
        symbol="AAPL",
        quantity=10,
        avg_entry_price=100.0,
        current_price=100.0,
        market_value=1000.0,
        unrealized_pnl=0.0,
        unrealized_pnl_pct=0.0,
        updated_at=datetime.now(timezone.utc)
    )

    # Simulate price surge to $106.00 (+6% gain -> exceeds 5% take-profit target)
    broker.positions["AAPL"].current_price = 106.00
    broker.positions["AAPL"].unrealized_pnl = 60.00
    broker.positions["AAPL"].unrealized_pnl_pct = 0.06

    await auto_trader.manage_open_positions()

    # Position should have been automatically liquidated to lock in profit
    positions = await broker.get_positions()
    assert len(positions) == 0


@pytest.mark.asyncio
async def test_autonomous_trader_auto_exit_stop_loss():
    broker = DryRunBroker(initial_equity=30000.0)
    risk_gate = DeterministicRiskGate(settings)
    strategy = EMACrossRSIStrategy()
    orchestrator = TradingOrchestrator(strategy=strategy, risk_gate=risk_gate)
    reconciler = ReconciliationService()
    # Price dropped to $96.00 (-4% loss -> exceeds 2.5% stop loss limit)
    data_provider = SyntheticMarketDataProvider(seed_price=96.0)
    bar_cache = BarCache()
    telegram = TelegramNotifier()
    supabase = SupabaseManager()

    auto_trader = AutonomousTrader(
        broker=broker,
        orchestrator=orchestrator,
        risk_gate=risk_gate,
        reconciler=reconciler,
        data_provider=data_provider,
        bar_cache=bar_cache,
        telegram=telegram,
        supabase=supabase
    )

    broker.positions["TSLA"] = Position(
        symbol="TSLA",
        quantity=15,
        avg_entry_price=100.0,
        current_price=96.0,
        market_value=1440.0,
        unrealized_pnl=-60.0,
        unrealized_pnl_pct=-0.04,
        updated_at=datetime.now(timezone.utc)
    )

    await auto_trader.manage_open_positions()

    # Position must be automatically sold to protect capital
    positions = await broker.get_positions()
    assert len(positions) == 0
