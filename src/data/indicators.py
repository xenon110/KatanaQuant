"""
Pure deterministic mathematical calculations for technical indicators.
Built on pandas and numpy without external black-box library dependencies.
"""
from typing import Union, Tuple
import pandas as pd
import numpy as np


def calculate_sma(series: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average."""
    return series.rolling(window=period).mean()


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    Uses Wilder's Smoothing method.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # First value is average, subsequent values are smoothed
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR).
    Expects DataFrame with columns ['high', 'low', 'close'].
    """
    high = df['high']
    low = df['low']
    close_prev = df['close'].shift(1)

    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1/period, adjust=False).mean()
    return atr


def calculate_bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.
    Returns (upper_band, middle_band, lower_band).
    """
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    return upper, middle, lower


def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Volume Weighted Average Price (VWAP).
    Expects DataFrame with columns ['high', 'low', 'close', 'volume'].
    """
    typical_price = (df['high'] + df['low'] + df['close']) / 3.0
    cum_pv = (typical_price * df['volume']).cumsum()
    cum_vol = df['volume'].cumsum()
    return cum_pv / cum_vol.replace(0, np.nan)


def calculate_macd(
    series: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD Line, Signal Line, and MACD Histogram.
    Returns (macd_line, signal_line, histogram).
    """
    ema_fast = calculate_ema(series, fast_period)
    ema_slow = calculate_ema(series, slow_period)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    return macd_line.fillna(0.0), signal_line.fillna(0.0), histogram.fillna(0.0)


def calculate_stochastic(
    df: pd.DataFrame, k_period: int = 14, d_period: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Stochastic Oscillator (%K, %D).
    Expects DataFrame with columns ['high', 'low', 'close'].
    Returns (%K, %D).
    """
    lowest_low = df['low'].rolling(window=k_period).min()
    highest_high = df['high'].rolling(window=k_period).max()
    rng = highest_high - lowest_low
    rng = rng.replace(0, np.nan)
    k = ((df['close'] - lowest_low) / rng) * 100.0
    k = k.fillna(50.0)
    d = k.rolling(window=d_period).mean().fillna(k)
    return k, d

