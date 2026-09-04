"""
Tests for Market Scheduler, Real-Time Alpaca WebSocket Stream, and Multi-Strategy Suite.
"""
import pytest
import pandas as pd
from datetime import datetime, timezone
from httpx import AsyncClient, ASGITransport

from src.scheduling.market_scheduler import MarketScheduler, MarketPhase
from src.data.alpaca_stream import AlpacaStreamManager
from src.strategy.rules import MultiTimeframeConfluenceStrategy
from src.core.models import MarketBar
from src.web.app import app


def test_confluence_strategy():
    strategy = MultiTimeframeConfluenceStrategy()
    
    # Create trending series with pullbacks where Close > 5 EMA > 20 EMA > 50 EMA
    prices = [100.0 + (i * 0.8) + (0.5 if i % 2 == 0 else -0.3) for i in range(70)]
    bars_data = [
        {"timestamp": datetime.now(timezone.utc), "open": p - 0.2, "high": p + 0.5, "low": p - 0.5, "close": p, "volume": 50000.0}
        for p in prices
    ]
    df = pd.DataFrame(bars_data)
    
    current_bar = MarketBar(
        symbol="NVDA",
        timestamp=datetime.now(timezone.utc),
        open=prices[-1] - 0.5,
        high=prices[-1] + 1.0,
        low=prices[-1] - 1.0,
        close=prices[-1],
        volume=50000.0
    )
    
    signal = strategy.evaluate(df, current_bar)
    assert signal is not None
    assert signal.symbol == "NVDA"
    assert signal.direction.value == "BULLISH"
    assert signal.confidence >= 0.80


@pytest.mark.asyncio
async def test_market_scheduler_lifecycle():
    cycle_triggered = []
    
    async def mock_cycle():
        cycle_triggered.append(True)

    scheduler = MarketScheduler(on_cycle_callback=mock_cycle)
    scheduler.force_active = True
    
    status = scheduler.get_status()
    assert status["is_running"] is False
    assert status["force_active"] is True
    
    await scheduler.start(interval_seconds=1)
    assert scheduler.is_running is True
    
    # Allow 1 cycle to execute
    import asyncio
    await asyncio.sleep(1.2)
    await scheduler.stop()
    
    assert scheduler.is_running is False
    assert len(cycle_triggered) >= 1
    assert scheduler.cycle_count >= 1


@pytest.mark.asyncio
async def test_scheduler_and_strategies_api_endpoints():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # 1. Get Scheduler Status
        r_sched = await ac.get("/api/scheduler")
        assert r_sched.status_code == 200
        sched_data = r_sched.json()
        assert "scheduler" in sched_data
        assert "stream" in sched_data

        # 2. Toggle Force Trade
        r_toggle = await ac.post("/api/scheduler/toggle-force-trade")
        assert r_toggle.status_code == 200
        assert "force_active" in r_toggle.json()

        # 3. Force Cycle Trigger
        r_cycle = await ac.post("/api/scheduler/force-cycle")
        assert r_cycle.status_code == 200
        assert r_cycle.json()["status"] == "success"

        # 4. List Available Strategies
        r_strats = await ac.get("/api/strategies")
        assert r_strats.status_code == 200
        strats = r_strats.json()["strategies"]
        strat_ids = [s["id"] for s in strats]
        assert "EMA_CROSS" in strat_ids
        assert "BOLLINGER" in strat_ids
        assert "VWAP" in strat_ids
        assert "CONFLUENCE" in strat_ids

        # 5. Set Strategy to CONFLUENCE
        r_set = await ac.post("/api/strategy", json={"strategy_type": "CONFLUENCE"})
        assert r_set.status_code == 200
        assert "MultiTimeframeConfluenceStrategy" in r_set.json()["active_strategy"]

        # 6. Run Backtest with CONFLUENCE
        r_bt = await ac.post("/api/backtest", json={
            "symbol": "NVDA",
            "days": 15,
            "capital": 30000.0,
            "strategy_type": "CONFLUENCE"
        })
        assert r_bt.status_code == 200
        assert r_bt.json()["status"] == "success"
