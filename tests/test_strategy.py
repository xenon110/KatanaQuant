"""
Unit tests for Technical Indicators and Deterministic Strategy Engine.
"""
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.core.enums import SignalDirection
from src.core.models import MarketBar
from src.data.indicators import calculate_ema, calculate_rsi, calculate_atr
from src.strategy.rules import EMACrossRSIStrategy
from src.backtesting.engine import BacktestEngine


def test_indicator_calculations():
    # 30 bars series
    prices = pd.Series([100.0 + i for i in range(30)])
    ema_9 = calculate_ema(prices, 9)
    assert len(ema_9) == 30
    assert ema_9.iloc[-1] > ema_9.iloc[0]

    rsi = calculate_rsi(prices, 14)
    assert len(rsi) == 30
    assert 0.0 <= rsi.iloc[-1] <= 100.0


def test_ema_crossover_strategy_bullish_signal():
    strategy = EMACrossRSIStrategy(fast_period=3, slow_period=6, rsi_period=5, atr_period=5)

    # Construct sequence where Fast EMA crosses above Slow EMA exactly at the last bar
    history = []
    base_price = 100.0
    start_time = datetime(2025, 1, 1, 9, 30)

    # Flat to mild downtrend
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
        symbol="SPY",
        timestamp=history[-1]["timestamp"],
        open=history[-1]["open"],
        high=history[-1]["high"],
        low=history[-1]["low"],
        close=history[-1]["close"],
        volume=history[-1]["volume"]
    )

    signal = strategy.evaluate(df, current_bar)
    assert signal is not None
    assert signal.direction == SignalDirection.BULLISH
    assert signal.symbol == "SPY"
    assert "stop_loss" in signal.key_levels
    assert "take_profit" in signal.key_levels


def test_backtester_engine_run():
    strategy = EMACrossRSIStrategy(fast_period=5, slow_period=10, rsi_period=7, atr_period=7)
    engine = BacktestEngine(strategy=strategy, initial_capital=30000.0)

    # Generate synthetic price series
    bars = []
    price = 150.0
    t = datetime(2025, 1, 1, 9, 30)
    for i in range(100):
        price += np.sin(i / 5.0) * 1.5 + 0.1
        bars.append(MarketBar(
            symbol="QQQ",
            timestamp=t + timedelta(minutes=i),
            open=price,
            high=price + 0.5,
            low=price - 0.5,
            close=price,
            volume=10000.0
        ))

    result = engine.run(bars)
    assert result.initial_capital == 30000.0
    assert result.final_equity > 0.0
    assert len(result.equity_curve) == 100


def test_bollinger_bands_breakout_strategy():
    from src.strategy.rules import BollingerBandsBreakoutStrategy
    strategy = BollingerBandsBreakoutStrategy(period=10, num_std=1.5)

    history = []
    start_time = datetime(2025, 1, 1, 9, 30)
    prices = [100.0 + (i * 0.1) for i in range(25)]
    # Big breakout spike
    prices[-1] = 115.0

    for i, p in enumerate(prices):
        history.append({
            "timestamp": start_time + timedelta(minutes=i),
            "open": p - 0.5,
            "high": p + 1.0,
            "low": p - 0.5,
            "close": p,
            "volume": 50000.0
        })

    df = pd.DataFrame(history)
    current_bar = MarketBar(
        symbol="NVDA",
        timestamp=history[-1]["timestamp"],
        open=history[-1]["open"],
        high=history[-1]["high"],
        low=history[-1]["low"],
        close=history[-1]["close"],
        volume=history[-1]["volume"]
    )

    signal = strategy.evaluate(df, current_bar)
    assert signal is not None
    assert signal.direction == SignalDirection.BULLISH
    assert signal.symbol == "NVDA"


def test_vwap_mean_reversion_strategy():
    from src.strategy.rules import VWAPMeanReversionStrategy
    strategy = VWAPMeanReversionStrategy(deviation_atr_mult=1.0)

    history = []
    start_time = datetime(2025, 1, 1, 9, 30)
    # High volume morning at 200, then heavy sharp dip to 190
    for i in range(25):
        p = 200.0 if i < 20 else 190.0
        history.append({
            "timestamp": start_time + timedelta(minutes=i),
            "open": p,
            "high": p + 0.5,
            "low": p - 0.5,
            "close": p,
            "volume": 100000.0 if i < 20 else 1000.0
        })

    df = pd.DataFrame(history)
    current_bar = MarketBar(
        symbol="TSLA",
        timestamp=history[-1]["timestamp"],
        open=history[-1]["open"],
        high=history[-1]["high"],
        low=history[-1]["low"],
        close=history[-1]["close"],
        volume=history[-1]["volume"]
    )

    signal = strategy.evaluate(df, current_bar)
    assert signal is not None
    assert signal.direction == SignalDirection.BULLISH
    assert signal.symbol == "TSLA"

