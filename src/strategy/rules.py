"""
Deterministic Strategy Implementations.
Rule-based entry and exit signal generators with indicator math.
"""
from typing import Optional, Dict
import pandas as pd

from src.core.models import MarketBar, SignalProposal
from src.core.enums import SignalDirection
from src.strategy.base import BaseStrategy
from src.data.indicators import calculate_ema, calculate_rsi, calculate_atr


class EMACrossRSIStrategy(BaseStrategy):
    """
    Trend-following momentum strategy with RSI confirmation and ATR-based stops.
    - Bullish: Fast EMA crosses above Slow EMA, RSI in [40, 70], price above Fast EMA.
    - Bearish: Fast EMA crosses below Slow EMA, RSI in [30, 60], price below Fast EMA.
    """

    def __init__(
        self,
        fast_period: int = 9,
        slow_period: int = 21,
        rsi_period: int = 14,
        atr_period: int = 14,
        atr_multiplier_stop: float = 1.5,
        atr_multiplier_target: float = 3.0,
    ):
        super().__init__(name="EMACrossRSIStrategy")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        self.atr_stop_mult = atr_multiplier_stop
        self.atr_target_mult = atr_multiplier_target

    def precompute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Precomputes all indicators across the entire series for instant backtesting."""
        df = df.copy()
        df['fast_ema'] = calculate_ema(df['close'], self.fast_period)
        df['slow_ema'] = calculate_ema(df['close'], self.slow_period)
        df['rsi'] = calculate_rsi(df['close'], self.rsi_period)
        df['atr'] = calculate_atr(df, self.atr_period)
        return df

    def evaluate(self, df: pd.DataFrame, current_bar: MarketBar) -> Optional[SignalProposal]:
        min_bars = max(self.slow_period, self.rsi_period, self.atr_period) + 5
        if len(df) < min_bars:
            return None

        # Calculate indicators
        fast_ema = calculate_ema(df['close'], self.fast_period)
        slow_ema = calculate_ema(df['close'], self.slow_period)
        rsi = calculate_rsi(df['close'], self.rsi_period)
        atr = calculate_atr(df, self.atr_period)

        current_fast = fast_ema.iloc[-1]
        prev_fast = fast_ema.iloc[-2]
        current_slow = slow_ema.iloc[-1]
        prev_slow = slow_ema.iloc[-2]
        current_rsi = rsi.iloc[-1]
        current_atr = atr.iloc[-1]
        close_p = current_bar.close

        # Bullish Crossover: Fast crosses above Slow
        bullish_cross = (prev_fast <= prev_slow) and (current_fast > current_slow)
        # Bearish Crossover: Fast crosses below Slow
        bearish_cross = (prev_fast >= prev_slow) and (current_fast < current_slow)

        # Institutional Volume Filter (Avoids low-volume fakeouts)
        vol_confirmed = True
        if 'volume' in df.columns and len(df['volume']) >= 20:
            avg_vol = df['volume'].iloc[-20:-1].mean()
            current_vol = current_bar.volume if current_bar.volume > 0 else df['volume'].iloc[-1]
            if avg_vol > 0 and current_vol < (avg_vol * 0.75):
                vol_confirmed = False

        if bullish_cross and vol_confirmed and (35.0 <= current_rsi <= 80.0) and (close_p > current_fast):
            stop_loss = round(close_p - (current_atr * self.atr_stop_mult), 2)
            take_profit = round(close_p + (current_atr * self.atr_target_mult), 2)
            
            return SignalProposal(
                symbol=current_bar.symbol,
                direction=SignalDirection.BULLISH,
                confidence=round(min(0.95, 0.70 + (current_rsi - 50.0) * 0.01), 2),
                reasoning=(
                    f"Fast EMA ({self.fast_period}) crossed above Slow EMA ({self.slow_period}) "
                    f"with confirmed volume expansion and RSI at {current_rsi:.1f}."
                ),
                key_levels={
                    "entry": close_p,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "atr": round(current_atr, 2)
                }
            )

        if bearish_cross and (20.0 <= current_rsi <= 65.0) and (close_p < current_fast):
            stop_loss = round(close_p + (current_atr * self.atr_stop_mult), 2)
            take_profit = round(close_p - (current_atr * self.atr_target_mult), 2)

            return SignalProposal(
                symbol=current_bar.symbol,
                direction=SignalDirection.BEARISH,
                confidence=round(min(0.95, 0.65 + (50.0 - current_rsi) * 0.01), 2),
                reasoning=(
                    f"Fast EMA ({self.fast_period}) crossed below Slow EMA ({self.slow_period}) "
                    f"with confirming RSI at {current_rsi:.1f} and price ({close_p:.2f}) < Fast EMA."
                ),
                key_levels={
                    "entry": close_p,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "atr": round(current_atr, 2)
                }
            )

        return None


class BollingerBandsBreakoutStrategy(BaseStrategy):
    """
    Volatility expansion momentum strategy:
    - Bullish: Price closes above Upper Bollinger Band with rising RSI > 55.
    - Bearish: Price closes below Lower Bollinger Band with falling RSI < 45.
    """
    def __init__(self, period: int = 20, num_std: float = 2.0, atr_stop_mult: float = 1.5):
        super().__init__(name="BollingerBandsBreakoutStrategy")
        self.period = period
        self.num_std = num_std
        self.atr_stop_mult = atr_stop_mult

    def precompute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        from src.data.indicators import calculate_bollinger_bands, calculate_rsi, calculate_atr
        upper, middle, lower = calculate_bollinger_bands(df['close'], self.period, self.num_std)
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower
        df['rsi'] = calculate_rsi(df['close'], 14)
        df['atr'] = calculate_atr(df, 14)
        return df

    def evaluate(self, df: pd.DataFrame, current_bar: MarketBar) -> Optional[SignalProposal]:
        if len(df) < self.period + 5:
            return None

        from src.data.indicators import calculate_bollinger_bands, calculate_rsi, calculate_atr
        upper, middle, lower = calculate_bollinger_bands(df['close'], self.period, self.num_std)
        rsi = calculate_rsi(df['close'], 14)
        atr = calculate_atr(df, 14)

        close_p = current_bar.close
        up_val = upper.iloc[-1]
        low_val = lower.iloc[-1]
        rsi_val = rsi.iloc[-1]
        atr_val = atr.iloc[-1]

        if close_p >= up_val and rsi_val >= 50.0:
            return SignalProposal(
                symbol=current_bar.symbol,
                direction=SignalDirection.BULLISH,
                confidence=0.82,
                reasoning=f"Bollinger Band upside breakout: Price (${close_p:.2f}) >= Upper Band (${up_val:.2f}) with RSI at {rsi_val:.1f}.",
                key_levels={
                    "entry": close_p,
                    "stop_loss": round(close_p - (atr_val * self.atr_stop_mult), 2),
                    "take_profit": round(close_p + (atr_val * self.atr_stop_mult * 2), 2),
                    "atr": round(atr_val, 2)
                }
            )

        if close_p <= low_val and rsi_val <= 50.0:
            return SignalProposal(
                symbol=current_bar.symbol,
                direction=SignalDirection.BEARISH,
                confidence=0.82,
                reasoning=f"Bollinger Band downside breakdown: Price (${close_p:.2f}) <= Lower Band (${low_val:.2f}) with RSI at {rsi_val:.1f}.",
                key_levels={
                    "entry": close_p,
                    "stop_loss": round(close_p + (atr_val * self.atr_stop_mult), 2),
                    "take_profit": round(close_p - (atr_val * self.atr_stop_mult * 2), 2),
                    "atr": round(atr_val, 2)
                }
            )

        return None


class VWAPMeanReversionStrategy(BaseStrategy):
    """
    Intraday institutional mean-reversion strategy:
    - Bullish: Price deviates > 1.0 ATR below VWAP in oversold condition (RSI <= 45).
    - Bearish: Price deviates > 1.0 ATR above VWAP in overbought condition (RSI >= 55).
    """
    def __init__(self, deviation_atr_mult: float = 1.0):
        super().__init__(name="VWAPMeanReversionStrategy")
        self.deviation_mult = deviation_atr_mult

    def precompute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        from src.data.indicators import calculate_vwap, calculate_rsi, calculate_atr
        df['vwap'] = calculate_vwap(df)
        df['rsi'] = calculate_rsi(df['close'], 14)
        df['atr'] = calculate_atr(df, 14)
        return df

    def evaluate(self, df: pd.DataFrame, current_bar: MarketBar) -> Optional[SignalProposal]:
        if len(df) < 15:
            return None

        from src.data.indicators import calculate_vwap, calculate_rsi, calculate_atr
        vwap = calculate_vwap(df)
        rsi = calculate_rsi(df['close'], 14)
        atr = calculate_atr(df, 14)

        close_p = current_bar.close
        vwap_val = vwap.iloc[-1]
        rsi_val = rsi.iloc[-1]
        atr_val = atr.iloc[-1]

        # Oversold below VWAP
        if (vwap_val - close_p) >= (atr_val * self.deviation_mult) and rsi_val <= 45.0:
            return SignalProposal(
                symbol=current_bar.symbol,
                direction=SignalDirection.BULLISH,
                confidence=0.80,
                reasoning=f"VWAP Mean Reversion: Price (${close_p:.2f}) oversold {self.deviation_mult}x ATR below VWAP (${vwap_val:.2f}) with RSI {rsi_val:.1f}. Target return to mean.",
                key_levels={
                    "entry": close_p,
                    "stop_loss": round(close_p - (atr_val * 1.5), 2),
                    "take_profit": round(vwap_val, 2),
                    "atr": round(atr_val, 2)
                }
            )

        # Overbought above VWAP
        if (close_p - vwap_val) >= (atr_val * self.deviation_mult) and rsi_val >= 55.0:
            return SignalProposal(
                symbol=current_bar.symbol,
                direction=SignalDirection.BEARISH,
                confidence=0.80,
                reasoning=f"VWAP Mean Reversion: Price (${close_p:.2f}) overbought {self.deviation_mult}x ATR above VWAP (${vwap_val:.2f}) with RSI {rsi_val:.1f}. Target return to mean.",
                key_levels={
                    "entry": close_p,
                    "stop_loss": round(close_p + (atr_val * 1.5), 2),
                    "take_profit": round(vwap_val, 2),
                    "atr": round(atr_val, 2)
                }
            )

        return None


class MultiTimeframeConfluenceStrategy(BaseStrategy):
    """
    Multi-timeframe trend confluence strategy.
    Aligns short-term momentum (5 EMA), intermediate trend (20 EMA),
    and macro baseline (50 EMA) with RSI confirmation for high-probability setups.
    """
    def __init__(self, short_period: int = 5, med_period: int = 20, long_period: int = 50):
        super().__init__(name="MultiTimeframeConfluenceStrategy")
        self.short_period = short_period
        self.med_period = med_period
        self.long_period = long_period

    def precompute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        from src.data.indicators import calculate_ema, calculate_rsi, calculate_atr
        df['ema_short'] = calculate_ema(df['close'], self.short_period)
        df['ema_med'] = calculate_ema(df['close'], self.med_period)
        df['ema_long'] = calculate_ema(df['close'], self.long_period)
        df['rsi'] = calculate_rsi(df['close'], 14)
        df['atr'] = calculate_atr(df, 14)
        return df

    def evaluate(self, df: pd.DataFrame, current_bar: MarketBar) -> Optional[SignalProposal]:
        if len(df) < self.long_period + 5:
            return None

        from src.data.indicators import calculate_ema, calculate_rsi, calculate_atr
        ema_short = calculate_ema(df['close'], self.short_period).iloc[-1]
        ema_med = calculate_ema(df['close'], self.med_period).iloc[-1]
        ema_long = calculate_ema(df['close'], self.long_period).iloc[-1]
        rsi = calculate_rsi(df['close'], 14).iloc[-1]
        atr = calculate_atr(df, 14).iloc[-1]

        close_p = current_bar.close

        # Full Bullish Alignment: Close > Short > Med > Long
        if (close_p > ema_short > ema_med > ema_long) and (rsi >= 45.0):
            return SignalProposal(
                symbol=current_bar.symbol,
                direction=SignalDirection.BULLISH,
                confidence=0.88,
                reasoning=(
                    f"Triple EMA Confluence: Price (${close_p:.2f}) > EMA{self.short_period} > "
                    f"EMA{self.med_period} > EMA{self.long_period} with healthy bullish RSI ({rsi:.1f})."
                ),
                key_levels={
                    "entry": close_p,
                    "stop_loss": round(close_p - (atr * 1.5), 2),
                    "take_profit": round(close_p + (atr * 3.0), 2),
                    "atr": round(atr, 2)
                }
            )

        # Full Bearish Alignment: Close < Short < Med < Long
        if (close_p < ema_short < ema_med < ema_long) and (rsi <= 55.0):
            return SignalProposal(
                symbol=current_bar.symbol,
                direction=SignalDirection.BEARISH,
                confidence=0.88,
                reasoning=(
                    f"Triple EMA Bearish Confluence: Price (${close_p:.2f}) < EMA{self.short_period} < "
                    f"EMA{self.med_period} < EMA{self.long_period} with confirming bearish RSI ({rsi:.1f})."
                ),
                key_levels={
                    "entry": close_p,
                    "stop_loss": round(close_p + (atr * 1.5), 2),
                    "take_profit": round(close_p - (atr * 3.0), 2),
                    "atr": round(atr, 2)
                }
            )

        return None
