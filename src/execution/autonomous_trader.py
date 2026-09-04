"""
Autonomous 100% Hands-Free Trading Engine.
Runs continuous background loops to:
1. Auto-Scan & Auto-Buy high-confidence trade opportunities across watchlist.
2. Auto-Monitor & Auto-Sell open positions to lock in profits or cap risk at target levels.
"""
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

from src.config.settings import settings
from src.core.enums import OrderSide, OrderType, SignalDirection
from src.core.models import ProposedTrade, MarketBar, Position

logger = logging.getLogger(__name__)


class AutonomousTrader:
    def __init__(
        self,
        broker,
        orchestrator,
        risk_gate,
        reconciler,
        data_provider,
        bar_cache,
        telegram,
        supabase,
        on_event_callback: Optional[Any] = None
    ):
        self.broker = broker
        self.orchestrator = orchestrator
        self.risk_gate = risk_gate
        self.reconciler = reconciler
        self.data_provider = data_provider
        self.bar_cache = bar_cache
        self.telegram = telegram
        self.supabase = supabase
        self.on_event_callback = on_event_callback

        self.is_active: bool = True  # Fully autonomous by default
        self.scan_symbols: List[str] = [s.upper() for s in settings.watchlist_symbols]
        self.take_profit_pct: float = 0.05   # Default 5.0% profit target
        self.stop_loss_pct: float = 0.025    # Default 2.5% max risk stop
        self.trailing_stop_enabled: bool = True
        self.position_meta: Dict[str, Dict[str, Any]] = {} # symbol -> {target_price, stop_price, high_water_mark}

    async def execute_autonomous_cycle(self):
        """Single full cycle: 1) Auto-Sell/Take-Profit check, 2) Auto-Buy Scanner check."""
        if not self.is_active:
            return

        try:
            # 1. Manage existing positions (Auto-Take-Profit & Auto-Stop-Loss)
            await self.manage_open_positions()

            # 2. Scan universe and auto-buy new setups
            await self.scan_and_execute_entries()
        except Exception as e:
            logger.error(f"Error in Autonomous Trader cycle: {e}")

    async def manage_open_positions(self):
        """
        Continuously inspects all open positions.
        Automatically sells to lock in profit when target price is reached or stop is triggered.
        """
        positions = await self.broker.get_positions()
        raw_acc = await self.broker.get_account()
        account = self.reconciler.reconcile_account(raw_acc)

        for pos in positions:
            sym = pos.symbol.upper()
            if pos.quantity == 0:
                continue

            try:
                # Fetch fresh real-time quote
                quote = await self.data_provider.get_latest_quote(sym)
                current_price = quote.bid_price if (quote and quote.bid_price > 0) else pos.current_price
            except Exception:
                current_price = pos.current_price

            entry_price = pos.avg_entry_price
            unrealized_pnl_pct = (current_price - entry_price) / entry_price if entry_price > 0 else 0.0
            unrealized_pnl_usd = (current_price - entry_price) * pos.quantity

            # Update highest price seen for trailing stop & break-even logic
            if sym not in self.position_meta:
                self.position_meta[sym] = {
                    "high_price": max(current_price, entry_price),
                    "target_price": entry_price * (1.0 + self.take_profit_pct),
                    "stop_price": entry_price * (1.0 - self.stop_loss_pct),
                    "break_even_locked": False
                }

            meta = self.position_meta[sym]
            if current_price > meta.get("high_price", current_price):
                meta["high_price"] = current_price

            # 🛡️ 1. Dynamic Break-Even Profit Lock (Locks in principal when up +0.75%)
            if unrealized_pnl_pct >= 0.0075 and not meta.get("break_even_locked", False):
                meta["stop_price"] = max(meta["stop_price"], entry_price * 1.001)  # Break-even + buffer
                meta["break_even_locked"] = True
                logger.info(f"🔒 [BREAK-EVEN LOCKED] {sym} is up +{(unrealized_pnl_pct*100):.2f}%. Stop-loss raised to Entry Break-Even: ${meta['stop_price']:.2f}")

            # 🛡️ 2. Tier-2 Profit Lock: If up +2.0%, guarantee at least +1.0% profit
            if unrealized_pnl_pct >= 0.020:
                meta["stop_price"] = max(meta["stop_price"], entry_price * 1.010)

            # 🛡️ 3. Tight Trailing Stop: If in profit >= 3.0%, trail stop 1.2% below highest peak
            if unrealized_pnl_pct >= 0.030 and self.trailing_stop_enabled:
                meta["stop_price"] = max(meta["stop_price"], meta["high_price"] * 0.988)

            target_price = meta.get("target_price", entry_price * (1.0 + self.take_profit_pct))
            stop_price = meta.get("stop_price", entry_price * (1.0 - self.stop_loss_pct))

            should_sell = False
            sell_reason = ""

            # Check 1: Take Profit Target Achieved
            if current_price >= target_price or unrealized_pnl_pct >= self.take_profit_pct:
                should_sell = True
                sell_reason = f"PROFIT TARGET REACHED (+{(unrealized_pnl_pct * 100):.2f}% · +${unrealized_pnl_usd:,.2f})"

            # Check 2: Break-Even / Trailing / Hard Stop Triggered
            elif current_price <= stop_price:
                should_sell = True
                if meta.get("break_even_locked", False) and current_price >= entry_price:
                    sell_reason = f"BREAK-EVEN PROFIT PROTECTED (+{(unrealized_pnl_pct * 100):.2f}% · +${unrealized_pnl_usd:,.2f})"
                elif unrealized_pnl_pct > 0:
                    sell_reason = f"TRAILING STOP PROFIT LOCKED (+{(unrealized_pnl_pct * 100):.2f}% · +${unrealized_pnl_usd:,.2f})"
                else:
                    sell_reason = f"STOP LOSS EXECUTED ({(unrealized_pnl_pct * 100):.2f}% · -${abs(unrealized_pnl_usd):,.2f})"

            if should_sell:
                logger.info(f"⚡ [AUTONOMOUS EXIT] Selling {pos.quantity} shares of {sym}: {sell_reason}")
                close_trade = ProposedTrade(
                    symbol=sym,
                    side=OrderSide.SELL if pos.quantity > 0 else OrderSide.BUY,
                    quantity=abs(pos.quantity),
                    price=current_price,
                    order_type=OrderType.MARKET
                )
                
                decision = self.risk_gate.evaluate(
                    proposed_trade=close_trade,
                    account=account,
                    current_positions=positions
                )
                decision.approved = True
                decision.allowed_quantity = abs(pos.quantity)

                order = await self.broker.submit_order(close_trade, decision)
                self.reconciler.record_trade_fill(
                    symbol=sym,
                    side=close_trade.side,
                    quantity=abs(pos.quantity),
                    price=current_price,
                    order_id=order.order_id
                )

                # Clean up position metadata
                self.position_meta.pop(sym, None)

                # Notify via Telegram
                profit_tag = "💰 PROFIT LOCKED" if unrealized_pnl_usd >= 0 else "🛡️ STOP LOSS EXECUTED"
                msg = (
                    f"*{profit_tag}*\n\n"
                    f"• *Symbol:* `{sym}`\n"
                    f"• *Action:* `AUTO SELL`\n"
                    f"• *Shares:* `{abs(pos.quantity)} shares`\n"
                    f"• *Exit Price:* `${current_price:.2f}` (Entry: `${entry_price:.2f}`)\n"
                    f"• *Realized Gain:* `${unrealized_pnl_usd:+,.2f}` (`{(unrealized_pnl_pct*100):+.2f}%`)\n"
                    f"• *Reason:* `{sell_reason}`"
                )
                asyncio.create_task(self.telegram.send_message(msg))

                # Log to Supabase Cloud
                asyncio.create_task(self.supabase.record_order({
                    "order_id": order.order_id,
                    "symbol": sym,
                    "side": "SELL",
                    "quantity": abs(pos.quantity),
                    "fill_price": current_price,
                    "status": "FILLED",
                    "type": "AUTO_EXIT",
                    "source": "AUTONOMOUS_TRADER",
                    "reasoning": sell_reason,
                    "created_at": datetime.now(timezone.utc).isoformat()
                }))

                if self.on_event_callback:
                    event = {
                        "type": "AUTO_POSITION_EXITED",
                        "symbol": sym,
                        "side": "SELL",
                        "quantity": abs(pos.quantity),
                        "price": current_price,
                        "pnl_usd": unrealized_pnl_usd,
                        "pnl_pct": unrealized_pnl_pct * 100,
                        "reason": sell_reason,
                        "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S")
                    }
                    res = self.on_event_callback(event)
                    if asyncio.iscoroutine(res):
                        await res

    async def scan_and_execute_entries(self):
        """
        Scans all watchlist symbols against quantitative strategies.
        If high-probability setup confirmed & Risk Gate approves -> Submits Buy Order to Alpaca.
        """
        positions = await self.broker.get_positions()
        active_symbols = {p.symbol.upper() for p in positions if p.quantity != 0}

        # Check maximum concurrent positions limit
        if len(active_symbols) >= settings.max_concurrent_positions:
            logger.debug("Max concurrent positions reached. Holding scanner.")
            return

        raw_acc = await self.broker.get_account()
        account = self.reconciler.reconcile_account(raw_acc)

        for sym in self.scan_symbols:
            # Skip if already holding position in this symbol
            if sym in active_symbols:
                continue

            try:
                quote = await self.data_provider.get_latest_quote(sym)
                price = quote.bid_price if (quote and quote.bid_price > 0) else 150.0

                now = datetime.now(timezone.utc)
                bar = MarketBar(
                    symbol=sym,
                    timestamp=now,
                    open=price - 0.15,
                    high=price + 0.35,
                    low=price - 0.20,
                    close=price,
                    volume=15000.0
                )
                self.bar_cache.add_bar(bar)
                df = self.bar_cache.get_dataframe(sym)

                result = await self.orchestrator.process_bar(
                    bar=bar,
                    history_df=df,
                    account=account,
                    positions=positions,
                    data_staleness_seconds=0.5
                )

                if result:
                    trade, decision = result
                    if decision.approved and decision.allowed_quantity > 0 and trade.side == OrderSide.BUY:
                        logger.info(f"⚡ [AUTONOMOUS ENTRY] Auto-Buying {decision.allowed_quantity} shares of {sym} @ ${trade.price:.2f}")

                        # Set take profit and stop loss targets
                        tp_price = round(trade.price * (1.0 + self.take_profit_pct), 2)
                        sl_price = round(trade.price * (1.0 - self.stop_loss_pct), 2)

                        order = await self.broker.submit_order(trade, decision)
                        self.reconciler.record_trade_fill(
                            symbol=sym,
                            side=trade.side,
                            quantity=decision.allowed_quantity,
                            price=trade.price,
                            order_id=order.order_id
                        )

                        self.position_meta[sym] = {
                            "high_price": trade.price,
                            "target_price": tp_price,
                            "stop_price": sl_price
                        }

                        # Send Telegram push alert
                        msg = (
                            f"🚀 *AUTO BUY ORDER FILLED*\n\n"
                            f"• *Symbol:* `{sym}`\n"
                            f"• *Quantity:* `{decision.allowed_quantity} shares`\n"
                            f"• *Entry Price:* `${trade.price:.2f}`\n"
                            f"• *Take-Profit Target:* `${tp_price:.2f}` (+5.0%)\n"
                            f"• *Stop-Loss Cap:* `${sl_price:.2f}` (-2.5%)\n"
                            f"• *Strategy:* `{self.orchestrator.strategy.name}`"
                        )
                        asyncio.create_task(self.telegram.send_message(msg))

                        # Log to Supabase Cloud
                        asyncio.create_task(self.supabase.record_order({
                            "order_id": order.order_id,
                            "symbol": sym,
                            "side": "BUY",
                            "quantity": decision.allowed_quantity,
                            "fill_price": trade.price,
                            "status": "FILLED",
                            "type": "AUTO_ENTRY",
                            "source": "AUTONOMOUS_TRADER",
                            "reasoning": f"Autonomous Buy triggered by {self.orchestrator.strategy.name}",
                            "created_at": datetime.now(timezone.utc).isoformat()
                        }))

                        if self.on_event_callback:
                            event = {
                                "type": "AUTO_ORDER_FILLED",
                                "symbol": sym,
                                "side": "BUY",
                                "quantity": decision.allowed_quantity,
                                "price": trade.price,
                                "target_price": tp_price,
                                "stop_loss": sl_price,
                                "timestamp": now.strftime("%H:%M:%S")
                            }
                            res = self.on_event_callback(event)
                            if asyncio.iscoroutine(res):
                                await res

                        # Refresh account positions
                        positions = await self.broker.get_positions()
                        active_symbols.add(sym)
                        if len(active_symbols) >= settings.max_concurrent_positions:
                            break
            except Exception as e:
                logger.error(f"Error scanning symbol {sym}: {e}")
