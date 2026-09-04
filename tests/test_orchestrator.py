"""
Unit tests for the Multi-Agent Orchestrator Pipeline.
"""
import pytest
from datetime import datetime, timedelta, timezone
import pandas as pd

from src.config.settings import Settings
from src.core.enums import SignalDirection, OrderSide, RiskGateStatus, AccountType
from src.core.models import MarketBar, AccountState
from src.strategy.rules import EMACrossRSIStrategy
from src.risk.gate import DeterministicRiskGate
from src.orchestrator.pipeline import TradingOrchestrator


@pytest.mark.asyncio
async def test_orchestrator_pipeline_execution():
    config = Settings(
        MAX_POSITION_SIZE_USD=5000.0,
        MAX_POSITION_SIZE_PCT_NAV=0.10,
        MAX_CONCURRENT_POSITIONS=4,
        MAX_DAILY_LOSS_USD=500.0,
        MAX_DAILY_LOSS_PCT=0.02,
        MAX_STALENESS_SECONDS=60,
        LLM_PROVIDER="mock"
    )
    risk_gate = DeterministicRiskGate(config)
    strategy = EMACrossRSIStrategy(fast_period=3, slow_period=6, rsi_period=5, atr_period=5)
    orchestrator = TradingOrchestrator(strategy=strategy, risk_gate=risk_gate)

    account = AccountState(
        account_id="ACC_TEST",
        account_type=AccountType.MARGIN,
        equity=30000.0,
        cash=20000.0,
        settled_cash=20000.0,
        unsettled_cash=0.0,
        buying_power=40000.0,
        day_trade_count=0,
        starting_daily_equity=30000.0
    )

    # Build bullish cross bars
    history = []
    base_price = 100.0
    start_time = datetime.now(timezone.utc) - timedelta(minutes=20)
    prices = [100.0, 99.5, 99.0, 98.5, 98.0, 97.5, 97.0, 96.8, 96.5, 96.5, 96.5, 98.6]
    for i, p in enumerate(prices):
        history.append({
            "timestamp": start_time + timedelta(minutes=i),
            "open": p - 0.2,
            "high": p + 0.5,
            "low": p - 0.5,
            "close": p,
            "volume": 2000.0
        })

    df = pd.DataFrame(history)
    current_bar = MarketBar(
        symbol="AAPL",
        timestamp=history[-1]["timestamp"],
        open=history[-1]["open"],
        high=history[-1]["high"],
        low=history[-1]["low"],
        close=history[-1]["close"],
        volume=history[-1]["volume"]
    )

    result = await orchestrator.process_bar(
        bar=current_bar,
        history_df=df,
        account=account,
        positions=[],
        data_staleness_seconds=0.5
    )

    assert result is not None
    trade, decision = result
    assert trade.symbol == "AAPL"
    assert trade.side == OrderSide.BUY
    assert trade.signal_proposal is not None
    assert trade.market_context is not None
    assert trade.risk_review is not None
    assert trade.sizing_proposal is not None
    assert decision.approved is True
    assert decision.allowed_quantity > 0
