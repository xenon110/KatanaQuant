"""
Unit & integration tests for FastAPI Web Dashboard endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from src.web.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_get_account_endpoint(client):
    response = client.get("/api/account")
    assert response.status_code == 200
    data = response.json()
    assert "account" in data
    assert "positions" in data
    assert "circuit_breaker" in data
    assert data["account"]["equity"] >= 0


def test_get_candles_endpoint(client):
    response = client.get("/api/candles?symbol=SPY&count=10")
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "SPY"
    assert "candles" in data
    assert len(data["candles"]) > 0
    assert "open" in data["candles"][0]
    assert "close" in data["candles"][0]


def test_get_orderbook_endpoint(client):
    response = client.get("/api/orderbook?symbol=NVDA")
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "NVDA"
    assert "bids" in data
    assert "asks" in data
    assert len(data["bids"]) > 0
    assert len(data["asks"]) > 0


def test_simulate_step_endpoint(client):
    response = client.post("/api/simulate-step?symbol=AAPL")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "event" in data
    assert "account" in data
    assert "positions" in data
    assert data["event"]["symbol"] == "AAPL"


def test_kill_switch_trip_and_reset(client):
    # Trip
    trip_res = client.post("/api/kill-switch", json={"action": "trip", "reason": "Test Kill Switch"})
    assert trip_res.status_code == 200
    assert trip_res.json()["tripped"] is True

    # Check account reflects tripped status
    acc_res = client.get("/api/account")
    assert acc_res.json()["circuit_breaker"]["tripped"] is True

    # Reset
    reset_res = client.post("/api/kill-switch", json={"action": "reset"})
    assert reset_res.status_code == 200
    assert reset_res.json()["tripped"] is False


def test_manual_order_endpoint(client):
    req_body = {
        "symbol": "MSFT",
        "side": "BUY",
        "order_type": "MARKET",
        "quantity": 5,
        "price": 300.0
    }
    response = client.post("/api/manual-order", json=req_body)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("FILLED", "REJECTED_BY_RISK_GATE")


def test_strategy_switching_endpoints(client):
    # Set Bollinger
    res = client.post("/api/strategy", json={"strategy_type": "BOLLINGER"})
    assert res.status_code == 200
    assert res.json()["active_strategy"] == "BollingerBandsBreakoutStrategy"

    # Get Strategy
    get_res = client.get("/api/strategy")
    assert get_res.status_code == 200
    assert get_res.json()["strategy"] == "BollingerBandsBreakoutStrategy"

    # Switch to VWAP
    res2 = client.post("/api/strategy", json={"strategy_type": "VWAP"})
    assert res2.status_code == 200
    assert res2.json()["active_strategy"] == "VWAPMeanReversionStrategy"


def test_export_trades_csv_endpoint(client):
    res = client.get("/api/export-trades")
    assert res.status_code == 200
    assert "text/csv" in res.headers["content-type"]
    assert "Order ID,Client ID,Symbol,Side" in res.text


def test_equity_history_endpoint(client):
    res = client.get("/api/equity-history?timeframe=1W")
    assert res.status_code == 200
    data = res.json()
    assert data["timeframe"] == "1W"
    assert len(data["data"]) == 20
    assert len(data["labels"]) == 20


def test_candles_with_indicator_overlays(client):
    res = client.get("/api/candles?symbol=SPY&count=20")
    assert res.status_code == 200
    candles = res.json()["candles"]
    assert len(candles) > 0
    first = candles[0]
    assert "ema9" in first
    assert "ema21" in first
    assert "bb_upper" in first
    assert "vwap" in first

