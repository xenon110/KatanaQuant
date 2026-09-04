"""
Market Scheduler & Autonomous Execution Daemon.
Monitors US Market hours (NYSE/NASDAQ 9:30 AM - 4:00 PM EST),
triggers pre-market scans, autonomous intraday trade execution cycles,
and post-market reconciliation.
"""
import asyncio
import logging
from datetime import datetime, timezone, time
from typing import Optional, Callable, Dict, Any, List

from src.config.settings import settings
from src.scheduling import MarketPhase

logger = logging.getLogger(__name__)

# US Market Hours in Eastern Time (ET)
PRE_MARKET_START = time(9, 0)
REGULAR_MARKET_OPEN = time(9, 30)
REGULAR_MARKET_CLOSE = time(16, 0)
POST_MARKET_END = time(17, 0)


class MarketScheduler:
    def __init__(
        self,
        on_cycle_callback: Optional[Callable[[], Any]] = None,
        on_phase_change_callback: Optional[Callable[[MarketPhase], Any]] = None,
        alpaca_trading_client: Optional[Any] = None,
    ):
        self.on_cycle_callback = on_cycle_callback
        self.on_phase_change_callback = on_phase_change_callback
        self.alpaca_client = alpaca_trading_client
        self.is_running: bool = False
        self.force_active: bool = False  # Allows 24/7 simulation or testing when market is closed
        self.current_phase: MarketPhase = MarketPhase.CLOSED
        self.last_cycle_time: Optional[datetime] = None
        self.cycle_count: int = 0
        self.market_clock_info: Dict[str, Any] = {}
        self._task: Optional[asyncio.Task] = None

    def get_phase_from_alpaca(self) -> MarketPhase:
        """Fetch market status from live Alpaca clock if client is available."""
        if not self.alpaca_client:
            return self._calculate_fallback_phase()

        try:
            clock = self.alpaca_client.get_clock()
            self.market_clock_info = {
                "is_open": clock.is_open,
                "next_open": clock.next_open.isoformat() if hasattr(clock.next_open, 'isoformat') else str(clock.next_open),
                "next_close": clock.next_close.isoformat() if hasattr(clock.next_close, 'isoformat') else str(clock.next_close),
                "timestamp": clock.timestamp.isoformat() if hasattr(clock.timestamp, 'isoformat') else str(clock.timestamp)
            }
            if clock.is_open:
                return MarketPhase.REGULAR_HOURS
            else:
                return MarketPhase.CLOSED
        except Exception as e:
            logger.warning(f"Error checking Alpaca clock: {e}. Using fallback calculation.")
            return self._calculate_fallback_phase()

    def _calculate_fallback_phase(self) -> MarketPhase:
        """Calculates current market phase based on UTC time."""
        now_utc = datetime.now(timezone.utc)
        # US Eastern is UTC-4 (EDT) or UTC-5 (EST)
        # Simplified: Weekdays (0-4) between 13:30 UTC and 20:00 UTC is regular market
        weekday = now_utc.weekday()
        if weekday >= 5:  # Saturday or Sunday
            return MarketPhase.CLOSED

        current_hour_min = now_utc.time()
        # 13:00 UTC (9:00 AM EDT) to 13:30 UTC (9:30 AM EDT)
        if time(13, 0) <= current_hour_min < time(13, 30):
            return MarketPhase.PRE_MARKET
        # 13:30 UTC (9:30 AM EDT) to 20:00 UTC (4:00 PM EDT)
        elif time(13, 30) <= current_hour_min < time(20, 0):
            return MarketPhase.REGULAR_HOURS
        # 20:00 UTC (4:00 PM EDT) to 21:00 UTC (5:00 PM EDT)
        elif time(20, 0) <= current_hour_min < time(21, 0):
            return MarketPhase.POST_MARKET
        else:
            return MarketPhase.CLOSED

    def update_phase(self) -> MarketPhase:
        new_phase = self.get_phase_from_alpaca()
        if new_phase != self.current_phase:
            logger.info(f"Market phase transitioned: {self.current_phase.value} -> {new_phase.value}")
            self.current_phase = new_phase
            if self.on_phase_change_callback:
                try:
                    res = self.on_phase_change_callback(new_phase)
                    if asyncio.iscoroutine(res):
                        asyncio.create_task(res)
                except Exception as e:
                    logger.error(f"Phase change callback error: {e}")
        return self.current_phase

    async def start(self, interval_seconds: int = 15):
        """Starts the autonomous scheduling background loop."""
        if self.is_running:
            return
        self.is_running = True
        self._task = asyncio.create_task(self._run_loop(interval_seconds))
        logger.info(f"Market Scheduler started with interval {interval_seconds}s (Force Active: {self.force_active})")

    async def stop(self):
        """Stops the autonomous scheduling background loop."""
        self.is_running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Market Scheduler stopped.")

    async def _run_loop(self, interval_seconds: int):
        while self.is_running:
            try:
                phase = self.update_phase()

                # Trigger trading cycle if regular hours OR force_active is enabled
                should_trade = (phase == MarketPhase.REGULAR_HOURS) or self.force_active
                if should_trade and self.on_cycle_callback:
                    self.cycle_count += 1
                    self.last_cycle_time = datetime.now(timezone.utc)
                    res = self.on_cycle_callback()
                    if asyncio.iscoroutine(res):
                        await res
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in scheduler execution cycle: {e}")

            await asyncio.sleep(interval_seconds)

    def get_status(self) -> Dict[str, Any]:
        """Returns comprehensive status dictionary for dashboard UI and health checks."""
        return {
            "is_running": self.is_running,
            "force_active": self.force_active,
            "current_phase": self.current_phase.value,
            "cycle_count": self.cycle_count,
            "last_cycle_time": self.last_cycle_time.isoformat() if self.last_cycle_time else None,
            "market_clock": self.market_clock_info,
            "is_trading_active": self.is_running and (self.current_phase == MarketPhase.REGULAR_HOURS or self.force_active)
        }
