"""
Unit & Integration tests for the AI Stock Advisor and One-Click Auto Trade Execution.
"""
import pytest
from httpx import AsyncClient, ASGITransport

from src.web.app import app
from src.strategy.ai_advisor import AIStockAdvisor
from src.core.models import AccountState


@pytest.mark.asyncio
async def test_ai_advisor_engine_basic():
    advisor = AIStockAdvisor()
    account = AccountState(
        account_id="ACC_TEST_001",
        equity=100000.0,
        cash=100000.0,
        settled_cash=100000.0,
        unsettled_cash=0.0,
        buying_power=400000.0,
        starting_daily_equity=100000.0
    )
    res = await advisor.analyze_stock("NVDA", account)
    
    assert res["symbol"] == "NVDA"
    assert "action" in res
    assert res["action"] in ["STRONG BUY", "BUY", "WAIT / HOLD", "AVOID / SELL"]
    assert res["confidence"] >= 50
    assert res["entry_price"] > 0
    assert res["target_price"] > 0
    assert res["stop_loss"] > 0
    assert res["suggested_shares"] >= 1
    assert len(res["why_buy_reasons"]) > 0
    assert len(res["warnings"]) > 0


@pytest.mark.asyncio
async def test_ai_advisor_api_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/api/ai-advisor?symbol=SPY")
        assert resp.status_code == 200
        data = resp.json()
        assert data["symbol"] == "SPY"
        assert "action" in data
        assert "suggested_shares" in data
        assert data["suggested_shares"] > 0


@pytest.mark.asyncio
async def test_ai_advisor_execute_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post("/api/ai-advisor/execute", json={
            "symbol": "SPY",
            "side": "BUY",
            "quantity": 5,
            "price": 565.00,
            "stop_loss": 555.00,
            "take_profit": 580.00,
            "reasoning": "Test AI Advisor One-Click Execution"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ["FILLED", "REJECTED_BY_RISK_GATE"]
        if data["status"] == "FILLED":
            assert data["symbol"] == "SPY"
            assert data["quantity"] == 5
