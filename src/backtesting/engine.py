"""
Event-driven backtesting engine for quantitative strategy evaluation.
Simulates realistic execution slippage, commissions, and portfolio equity evolution.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

from src.core.models import MarketBar, SignalProposal
from src.core.enums import SignalDirection, OrderSide
from src.strategy.base import BaseStrategy


@dataclass
class BacktestTrade:
    symbol: str
    entry_time: datetime
    exit_time: Optional[datetime] = None
    side: OrderSide = OrderSide.BUY
    entry_price: float = 0.0
    exit_price: float = 0.0
    quantity: int = 0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""


@dataclass
class BacktestResult:
    initial_capital: float
    final_equity: float
    total_pnl: float
    total_return_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    max_drawdown_pct: float
    sharpe_ratio: float
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)


class BacktestEngine:
    def __init__(
        self,
        strategy: BaseStrategy,
        initial_capital: float = 30000.0,
        slippage_bps: float = 2.0,      # 2 basis points slippage
        commission_per_share: float = 0.0, # Zero-commission for US equities (Alpaca model)
        position_size_pct: float = 0.10, # 10% equity per trade
    ):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.slippage = slippage_bps / 10000.0
        self.commission = commission_per_share
        self.position_size_pct = position_size_pct

    def run(self, bars: List[MarketBar]) -> BacktestResult:
        if not bars:
            raise ValueError("No market bars provided for backtest.")

        # Build complete DataFrame upfront
        data = [
            {
                "timestamp": b.timestamp,
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume
            }
            for b in bars
        ]
        full_df = pd.DataFrame(data)
        if hasattr(self.strategy, "precompute"):
            full_df = self.strategy.precompute(full_df)

        cash = self.initial_capital
        equity = self.initial_capital
        active_trade: Optional[BacktestTrade] = None
        trades: List[BacktestTrade] = []
        equity_curve: List[Dict[str, Any]] = []
        peak_equity = self.initial_capital
        max_drawdown = 0.0

        min_window = 30 # Evaluation lookback window

        has_precomputed = "fast_ema" in full_df.columns

        for i, bar in enumerate(bars):
            # Check open position exit conditions (Stop-Loss, Take-Profit)
            if active_trade:
                stop_loss = active_trade.entry_price * 0.98  # 2% stop
                take_profit = active_trade.entry_price * 1.05 # 5% target
                
                # Check stop loss hit
                if bar.low <= stop_loss:
                    fill_p = stop_loss * (1 - self.slippage)
                    pnl = (fill_p - active_trade.entry_price) * active_trade.quantity
                    cash += (fill_p * active_trade.quantity)
                    active_trade.exit_time = bar.timestamp
                    active_trade.exit_price = fill_p
                    active_trade.pnl = pnl
                    active_trade.pnl_pct = (fill_p - active_trade.entry_price) / active_trade.entry_price
                    active_trade.exit_reason = "STOP_LOSS"
                    trades.append(active_trade)
                    active_trade = None

                # Check take profit hit
                elif bar.high >= take_profit:
                    fill_p = take_profit * (1 - self.slippage)
                    pnl = (fill_p - active_trade.entry_price) * active_trade.quantity
                    cash += (fill_p * active_trade.quantity)
                    active_trade.exit_time = bar.timestamp
                    active_trade.exit_price = fill_p
                    active_trade.pnl = pnl
                    active_trade.pnl_pct = (fill_p - active_trade.entry_price) / active_trade.entry_price
                    active_trade.exit_reason = "TAKE_PROFIT"
                    trades.append(active_trade)
                    active_trade = None

            # Strategy Signal Evaluation
            if i >= min_window:
                signal = None
                if has_precomputed:
                    prev_fast = full_df['fast_ema'].iloc[i-1]
                    curr_fast = full_df['fast_ema'].iloc[i]
                    prev_slow = full_df['slow_ema'].iloc[i-1]
                    curr_slow = full_df['slow_ema'].iloc[i]
                    curr_rsi = full_df['rsi'].iloc[i]
                    curr_atr = full_df['atr'].iloc[i]

                    bullish_cross = (prev_fast <= prev_slow) and (curr_fast > curr_slow)
                    if bullish_cross and (35.0 <= curr_rsi <= 80.0) and (bar.close > curr_fast):
                        signal = SignalProposal(
                            symbol=bar.symbol,
                            direction=SignalDirection.BULLISH,
                            confidence=0.8,
                            reasoning="Bullish EMA crossover",
                            key_levels={"entry": bar.close, "stop_loss": bar.close - curr_atr, "take_profit": bar.close + (2 * curr_atr)}
                        )
                else:
                    window_df = full_df.iloc[max(0, i - 100):i + 1]
                    signal = self.strategy.evaluate(window_df, bar)

                if signal and not active_trade:
                    if signal.direction == SignalDirection.BULLISH:
                        alloc = min(cash, equity * self.position_size_pct)
                        shares = int(alloc // bar.close)
                        if shares > 0:
                            entry_fill = bar.close * (1 + self.slippage)
                            cost = (shares * entry_fill) + (shares * self.commission)
                            if cost <= cash:
                                cash -= cost
                                active_trade = BacktestTrade(
                                    symbol=bar.symbol,
                                    entry_time=bar.timestamp,
                                    side=OrderSide.BUY,
                                    entry_price=entry_fill,
                                    quantity=shares
                                )

            # Update current equity
            current_pos_val = (active_trade.quantity * bar.close) if active_trade else 0.0
            equity = cash + current_pos_val

            # Track peak and drawdown
            if equity > peak_equity:
                peak_equity = equity
            dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
            if dd > max_drawdown:
                max_drawdown = dd

            equity_curve.append({
                "timestamp": bar.timestamp,
                "equity": round(equity, 2),
                "cash": round(cash, 2),
                "drawdown": round(dd, 4)
            })

        # Close open trade at end of backtest if still open
        if active_trade and bars:
            last_bar = bars[-1]
            fill_p = last_bar.close * (1 - self.slippage)
            pnl = (fill_p - active_trade.entry_price) * active_trade.quantity
            cash += (fill_p * active_trade.quantity)
            active_trade.exit_time = last_bar.timestamp
            active_trade.exit_price = fill_p
            active_trade.pnl = pnl
            active_trade.pnl_pct = (fill_p - active_trade.entry_price) / active_trade.entry_price
            active_trade.exit_reason = "END_OF_DATA"
            trades.append(active_trade)
            equity = cash

        # Calculate metrics
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl < 0]
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)
        win_rate = (len(winning_trades) / len(trades)) if trades else 0.0

        # Calculate Sharpe Ratio from equity returns
        eq_series = pd.Series([pt["equity"] for pt in equity_curve])
        returns = eq_series.pct_change().dropna()
        if len(returns) > 1 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 390) # Annualized for minute bars
        else:
            sharpe = 0.0

        total_pnl = equity - self.initial_capital
        total_return_pct = (total_pnl / self.initial_capital) * 100.0

        return BacktestResult(
            initial_capital=self.initial_capital,
            final_equity=round(equity, 2),
            total_pnl=round(total_pnl, 2),
            total_return_pct=round(total_return_pct, 2),
            total_trades=len(trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=round(win_rate, 4),
            profit_factor=round(profit_factor, 2),
            max_drawdown_pct=round(max_drawdown * 100.0, 2),
            sharpe_ratio=round(float(sharpe), 2),
            trades=trades,
            equity_curve=equity_curve
        )
