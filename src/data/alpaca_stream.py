"""
Alpaca Real-Time WebSocket Market Data Streaming Client.
Streams live minute bars, trades, and quotes into the system's BarCache
and dispatches market updates directly to the trading pipeline.
"""
import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Callable, Optional, Dict, Any

from alpaca.data.live import StockDataStream
from alpaca.data.enums import DataFeed

from src.config.settings import settings
from src.core.models import MarketBar

logger = logging.getLogger(__name__)


class AlpacaStreamManager:
    """
    Manages live WebSocket subscriptions to US stock market data via Alpaca.
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        feed: DataFeed = DataFeed.IEX,
        on_bar_callback: Optional[Callable[[MarketBar], Any]] = None
    ):
        self.api_key = api_key or settings.alpaca_api_key
        self.secret_key = secret_key or settings.alpaca_secret_key
        self.symbols = [s.upper() for s in (symbols or settings.watchlist_symbols)]
        self.feed = feed
        self.on_bar_callback = on_bar_callback
        self.is_connected: bool = False
        self._stream: Optional[StockDataStream] = None
        self._task: Optional[asyncio.Task] = None
        self.bars_received: int = 0
        self.last_bar_time: Optional[datetime] = None

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.secret_key and self.api_key != "placeholder_key")

    async def _handle_bar(self, bar_data: Any):
        """Internal callback invoked when a new live bar arrives over WebSocket."""
        try:
            self.bars_received += 1
            self.last_bar_time = datetime.now(timezone.utc)
            
            # Map Alpaca Bar data to internal MarketBar model
            sym = str(bar_data.symbol)
            ts = bar_data.timestamp if hasattr(bar_data, "timestamp") else datetime.now(timezone.utc)
            
            market_bar = MarketBar(
                symbol=sym,
                timestamp=ts,
                open=float(bar_data.open),
                high=float(bar_data.high),
                low=float(bar_data.low),
                close=float(bar_data.close),
                volume=float(bar_data.volume)
            )

            logger.debug(f"[WebSocket] Live Bar {sym}: Close=${market_bar.close:.2f} Vol={market_bar.volume}")

            if self.on_bar_callback:
                res = self.on_bar_callback(market_bar)
                if asyncio.iscoroutine(res):
                    await res
        except Exception as e:
            logger.error(f"Error processing live bar from WebSocket stream: {e}")

    async def start(self):
        """Initializes and runs the live streaming connection in the background."""
        if not self.is_configured:
            logger.warning("Alpaca Stream: API keys not configured. Skipping WebSocket start.")
            return

        if self.is_connected:
            return

        try:
            self._stream = StockDataStream(
                api_key=self.api_key,
                secret_key=self.secret_key,
                feed=self.feed,
                raw_data=False
            )

            # Subscribe to minute bars for all watchlist symbols
            for sym in self.symbols:
                self._stream.subscribe_bars(self._handle_bar, sym)

            self.is_connected = True
            logger.info(f"Alpaca WebSocket stream subscribing to: {', '.join(self.symbols)}")
            
            # Run stream in background asyncio task
            self._task = asyncio.create_task(self._stream._run_forever())
        except Exception as e:
            self.is_connected = False
            logger.error(f"Failed to start Alpaca WebSocket stream: {e}")

    async def stop(self):
        """Closes the WebSocket stream gracefully."""
        self.is_connected = False
        if self._stream:
            try:
                await self._stream.close()
            except Exception as e:
                logger.debug(f"Stream close notice: {e}")
        if self._task and not self._task.done():
            self._task.cancel()
        logger.info("Alpaca WebSocket stream stopped.")

    def get_status(self) -> Dict[str, Any]:
        return {
            "is_connected": self.is_connected,
            "symbols": self.symbols,
            "bars_received": self.bars_received,
            "last_bar_time": self.last_bar_time.isoformat() if self.last_bar_time else None
        }
