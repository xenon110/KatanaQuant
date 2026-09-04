"""
End-to-End Verification Test for all interactive buttons and API routes in the dashboard.
"""
import pytest
from httpx import AsyncClient, ASGITransport
from src.web.app import app


@pytest.mark.asyncio
async def test_all_interactive_buttons_and_routes():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        
        # 1. Main Dashboard Loading & Account State
        r_acc = await ac.get("/api/account")
        assert r_acc.status_code == 200
        acc_data = r_acc.json()
        assert "account" in acc_data
        assert "positions" in acc_data
        assert "circuit_breaker" in acc_data

        # 2. Timeframe Buttons (1D, 1W, 1M, 3M, YTD)
        for tf in ["1D", "1W", "1M", "3M", "YTD"]:
            r_tf = await ac.get(f"/api/equity-history?timeframe={tf}")
            assert r_tf.status_code == 200
            assert "data" in r_tf.json()
            assert len(r_tf.json()["data"]) > 0

        # 3. Pipeline Trigger Button ("⚡ Trigger Pipeline Cycle")
        r_sim = await ac.post("/api/simulate-step?symbol=SPY")
        assert r_sim.status_code == 200
        assert r_sim.json()["status"] == "success"

        # 4. Strategy Selector Dropdown ("EMA_CROSS", "BOLLINGER", "VWAP")
        for strat in ["EMA_CROSS", "BOLLINGER", "VWAP"]:
            r_strat = await ac.post("/api/strategy", json={"strategy_type": strat})
            assert r_strat.status_code == 200
            assert r_strat.json()["status"] == "success"

        # 5. Export Trades CSV Button ("📥 EXPORT CSV")
        r_csv = await ac.get("/api/export-trades")
        assert r_csv.status_code == 200
        assert "text/csv" in r_csv.headers["content-type"]
        assert "Order ID" in r_csv.text

        # 6. Watchlist Live Feeds & Symbol Clicking
        r_watch = await ac.get("/api/watchlist")
        assert r_watch.status_code == 200
        assert len(r_watch.json().get("items", [])) > 0

        for sym in ["NVDA", "TSLA", "AAPL", "PLTR", "MSFT"]:
            r_c = await ac.get(f"/api/candles?symbol={sym}&count=30")
            assert r_c.status_code == 200
            assert len(r_c.json().get("candles", [])) > 0

        # 7. Level 2 Orderbook
        r_ob = await ac.get("/api/orderbook?symbol=NVDA")
        assert r_ob.status_code == 200
        assert len(r_ob.json().get("bids", [])) > 0
        assert len(r_ob.json().get("asks", [])) > 0

        # 8. AI Stock Advisor Scan ("⚡ SCAN WITH AI" & Popular Pick Pills)
        for sym in ["NVDA", "TSLA", "SPY", "AAPL", "PLTR", "AMD", "META", "AMZN", "MSFT", "COIN"]:
            r_ai = await ac.get(f"/api/ai-advisor?symbol={sym}")
            assert r_ai.status_code == 200
            d = r_ai.json()
            assert d["symbol"] == sym
            assert d["action"] in ["STRONG BUY", "BUY", "WAIT / HOLD", "AVOID / SELL"]
            assert d["confidence"] >= 50
            assert d["entry_price"] > 0
            assert d["target_price"] > 0
            assert d["stop_loss"] > 0
            assert d["suggested_shares"] >= 1

        # 9. One-Click AI Auto Trade Execution ("⚡ ONE-CLICK AUTO TRADE ON ALPACA")
        r_exec = await ac.post("/api/ai-advisor/execute", json={
            "symbol": "NVDA",
            "side": "BUY",
            "quantity": 10,
            "price": 128.50,
            "stop_loss": 124.00,
            "take_profit": 137.50,
            "reasoning": "AI Advisor Auto Execution Verification"
        })
        assert r_exec.status_code == 200
        assert r_exec.json()["status"] in ["FILLED", "REJECTED_BY_RISK_GATE", "REJECTED_BY_BROKER"]

        # 10. Manual Order Ticket ("SUBMIT ORDER VIA RISK GATE")
        r_man = await ac.post("/api/manual-order", json={
            "symbol": "SPY",
            "side": "BUY",
            "quantity": 5,
            "price": 565.00,
            "order_type": "MARKET"
        })
        assert r_man.status_code == 200
        assert r_man.json()["status"] in ["FILLED", "REJECTED_BY_RISK_GATE", "REJECTED_BY_BROKER"]


        # 11. Strategy Lab Backtest Engine ("RUN HISTORICAL BACKTEST")
        r_bt = await ac.post("/api/backtest", json={
            "symbol": "SPY",
            "days": 15,
            "capital": 30000.0,
            "strategy_type": "EMA_CROSS",
            "fast_period": 9,
            "slow_period": 21,
            "atr_mult": 1.5
        })
        assert r_bt.status_code == 200
        bt_data = r_bt.json()
        assert bt_data["status"] == "success"
        assert "final_equity" in bt_data
        assert "win_rate" in bt_data
        assert "equity_curve" in bt_data

        # 12. Emergency Kill Switch ("KILL SWITCH" Trip & Reset)
        r_kill_trip = await ac.post("/api/kill-switch", json={"action": "trip", "reason": "Test button audit"})
        assert r_kill_trip.status_code == 200
        assert r_kill_trip.json()["tripped"] is True

        r_kill_reset = await ac.post("/api/kill-switch", json={"action": "reset"})
        assert r_kill_reset.status_code == 200
        assert r_kill_reset.json()["tripped"] is False
