import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional
from datetime import datetime, time as dt_time, timedelta
import pytz
from threading import Lock
import threading
from collections import deque


@dataclass
class WebSocketListenerConfig:
    def __init__(self):
        self.logger_name = "app"
        self.market_start_time = dt_time(9, 15)
        self.market_end_time = dt_time(15, 30)
        self.timezone = pytz.timezone('Asia/Kolkata')
        self.check_interval = 60
        self.connection_check_interval = 30
        self.candle_interval_seconds = 60

        # Deadlock prevention settings
        self.max_tick_queue_size = 1000
        self.max_write_queue_timeout = 0.1  # Non-blocking queue operations
        self.candle_lock_timeout = 0.5  # Max time to wait for lock


class CandleData:
    """Thread-safe candle data container"""

    def __init__(self, timestamp: int, first_price: float, initial_volume: int = 0):
        self.timestamp = timestamp
        self.open = first_price
        self.high = first_price
        self.low = first_price
        self.close = first_price
        self.volume = 0
        self.trade_count = 1
        self.is_closed = False

        # Use thread lock for thread-safe updates from WebSocket callback
        self._lock = threading.Lock()
        self.last_total_volume = initial_volume
        self.candle_start_volume = initial_volume

    def update_tick_threadsafe(self, price: float, total_volume: int = 0, last_trade_qty: int = 0):
        """Thread-safe tick update for WebSocket callback"""
        with self._lock:
            self._update_tick_internal(price, total_volume, last_trade_qty)

    def _update_tick_internal(self, price: float, total_volume: int = 0, last_trade_qty: int = 0):
        """Internal update method - must be called with lock held"""
        self.close = price
        self.high = max(self.high, price)
        self.low = min(self.low, price)

        # Robust volume tracking
        if total_volume > self.last_total_volume:
            volume_increment = total_volume - self.last_total_volume
            self.volume += volume_increment
            self.last_total_volume = total_volume
        elif last_trade_qty > 0:
            self.volume += last_trade_qty

        self.trade_count += 1

    def close_candle_threadsafe(self):
        """Thread-safe candle closing"""
        with self._lock:
            self.is_closed = True

    def to_dict_threadsafe(self, symbol: str) -> dict:
        """Thread-safe dictionary conversion"""
        with self._lock:
            return {
                'symbol': symbol,
                'candle': {
                    'timestamp': self.timestamp,
                    'open': self.open,
                    'high': self.high,
                    'low': self.low,
                    'close': self.close,
                    'volume': self.volume,
                    'trades': self.trade_count,
                    'interval': '1m',
                    'is_closed': self.is_closed
                }
            }


class WebSocketListenerService:
    def __init__(self, symbol: str, write_queue: asyncio.Queue, monitor,
                 config: Optional[WebSocketListenerConfig] = None, fyers_connector=None):
        self.symbol = symbol
        self.write_queue = write_queue
        self.monitor = monitor
        self.config = WebSocketListenerConfig() if config is None else config
        self.logger = logging.getLogger(self.config.logger_name)
        self.connector = fyers_connector

        # Connection state
        self.websocket = None
        self.connected = False
        self.should_run = False
        self.last_message_time = 0
        self.reconnect_delay = 1
        self.max_reconnect_delay = 60
        self.ws_client = None
        self.loop = None

        # Deadlock-free tick processing using thread-safe queue
        self.tick_buffer = deque(maxlen=self.config.max_tick_queue_size)
        self.tick_buffer_lock = threading.Lock()

        # Candle management - no async locks needed
        self.current_candle: Optional[CandleData] = None
        self.current_minute_timestamp = 0
        self.tick_count = 0
        self.last_total_volume = 0

        # Task tracking for proper cleanup
        self.background_tasks = set()

        # Statistics for monitoring
        self.dropped_ticks = 0
        self.processed_ticks = 0

    def _get_minute_timestamp(self, timestamp: int = None) -> int:
        """Get the start timestamp of the current minute"""
        if timestamp is None:
            timestamp = int(time.time())
        return (timestamp // 60) * 60

    def _is_market_open(self) -> bool:
        """Check if the market is currently open"""
        now = datetime.now(self.config.timezone)
        if now.weekday() >= 5:
            return False
        current_time = now.time()
        return self.config.market_start_time <= current_time <= self.config.market_end_time

    def _time_until_market_open(self) -> int:
        """Calculate seconds until market opens"""
        if self._is_market_open():
            return 0

        now = datetime.now(self.config.timezone)

        if now.weekday() >= 5:
            days_until_monday = 7 - now.weekday()
            next_market_open = now.replace(
                hour=self.config.market_start_time.hour,
                minute=self.config.market_start_time.minute,
                second=0,
                microsecond=0
            ) + timedelta(days=days_until_monday)
        else:
            market_open_today = now.replace(
                hour=self.config.market_start_time.hour,
                minute=self.config.market_start_time.minute,
                second=0,
                microsecond=0
            )

            if now.time() > self.config.market_end_time:
                next_market_open = market_open_today + timedelta(days=1)
            else:
                next_market_open = market_open_today

        return int((next_market_open - now).total_seconds())

    async def start(self):
        """Start the websocket listener with proper task management"""
        self.logger.info(f"Starting websocket listener for {self.symbol}")
        self.should_run = True
        self.loop = asyncio.get_running_loop()

        # Start background tasks with proper tracking
        tick_processor = asyncio.create_task(self._process_tick_buffer())
        candle_checker = asyncio.create_task(self._candle_completion_checker())

        self.background_tasks.add(tick_processor)
        self.background_tasks.add(candle_checker)

        # Add cleanup callbacks
        tick_processor.add_done_callback(self.background_tasks.discard)
        candle_checker.add_done_callback(self.background_tasks.discard)

        while self.should_run:
            try:
                if not self._is_market_open():
                    wait_time = self._time_until_market_open()
                    if wait_time > 0:
                        self.logger.info(f"Market closed. Waiting {wait_time}s for {self.symbol}")
                        await asyncio.sleep(min(wait_time, self.config.check_interval))
                        continue

                await self._connect_and_listen()

            except Exception as e:
                self.logger.error(f"Error in websocket listener for {self.symbol}: {e}")

                if not self._is_market_open():
                    await asyncio.sleep(self.config.check_interval)
                    continue

                await asyncio.sleep(self.reconnect_delay)
                self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)
            else:
                self.reconnect_delay = 1

    async def _process_tick_buffer(self):
        """Process ticks from thread-safe buffer - NO LOCKS NEEDED"""
        while self.should_run:
            try:
                # Get ticks from buffer without blocking
                ticks_to_process = []

                with self.tick_buffer_lock:
                    # Process up to 50 ticks at once for efficiency
                    for _ in range(min(50, len(self.tick_buffer))):
                        if self.tick_buffer:
                            ticks_to_process.append(self.tick_buffer.popleft())

                # Process ticks without holding any locks
                for tick_data in ticks_to_process:
                    await self._process_single_tick(tick_data)

                # Small delay to prevent busy waiting
                await asyncio.sleep(0.01)  # 10ms

            except Exception as e:
                self.logger.error(f"Error processing tick buffer: {e}")
                await asyncio.sleep(0.1)

    async def _process_single_tick(self, data):
        """Process single tick without locks - deadlock-free"""
        try:
            if not isinstance(data, dict):
                return

            ltp = data.get('ltp')
            vol_traded_today = data.get('vol_traded_today', 0)
            last_traded_qty = data.get('last_traded_qty', 0)

            if ltp is None:
                return

            price = float(ltp)
            total_volume = int(vol_traded_today) if vol_traded_today else 0
            last_qty = int(last_traded_qty) if last_traded_qty else 0

            current_minute = self._get_minute_timestamp()

            # Handle candle creation/update without locks
            if (self.current_candle is None or
                    self.current_minute_timestamp != current_minute):

                # Complete previous candle if exists
                if (self.current_candle and
                        not self.current_candle.is_closed and
                        self.current_minute_timestamp != current_minute):
                    await self._complete_candle(self.current_candle)

                # Create a new candle
                self.current_candle = CandleData(current_minute, price, total_volume)
                self.current_minute_timestamp = current_minute
                self.last_total_volume = total_volume

                self.logger.debug(f"New candle for {self.symbol} at {current_minute}")
            else:
                # Update existing candle (thread-safe)
                self.current_candle.update_tick_threadsafe(price, total_volume, last_qty)

            self.processed_ticks += 1

        except Exception as e:
            self.logger.error(f"Error processing tick for {self.symbol}: {e}")

    async def _complete_candle(self, candle: CandleData):
        """Complete a candle and send to queue with timeout"""
        try:
            candle.close_candle_threadsafe()
            candle_message = candle.to_dict_threadsafe(self.symbol)

            # Non-blocking queue put with timeout to prevent deadlock
            try:
                await asyncio.wait_for(
                    self.write_queue.put(candle_message),
                    timeout=self.config.max_write_queue_timeout
                )

                self.logger.info(f"Completed candle for {self.symbol}: "
                                 f"O={candle_message['candle']['open']}, "
                                 f"H={candle_message['candle']['high']}, "
                                 f"L={candle_message['candle']['low']}, "
                                 f"C={candle_message['candle']['close']}, "
                                 f"V={candle_message['candle']['volume']}")

            except asyncio.TimeoutError:
                self.logger.warning(f"Write queue timeout for {self.symbol} - candle dropped")

        except Exception as e:
            self.logger.error(f"Error completing candle for {self.symbol}: {e}")

    async def _candle_completion_checker(self):
        """Check for candle completion every second - deadlock-free"""
        while self.should_run:
            try:
                await asyncio.sleep(1)

                if not self._is_market_open():
                    continue

                current_minute = self._get_minute_timestamp()

                # Check for candle completion without locks
                if (self.current_candle and
                        self.current_minute_timestamp != current_minute and
                        not self.current_candle.is_closed):
                    await self._complete_candle(self.current_candle)

            except Exception as e:
                self.logger.error(f"Error in candle completion checker: {e}")

    def _websocket_callback(self, data):
        """WebSocket callback - thread-safe, non-blocking"""
        try:
            self.last_message_time = time.time()
            self.monitor.report_message_received(self.symbol)

            # Add to buffer in thread-safe way
            with self.tick_buffer_lock:
                if len(self.tick_buffer) >= self.config.max_tick_queue_size:
                    self.tick_buffer.popleft()  # Drop the oldest tick
                    self.dropped_ticks += 1

                self.tick_buffer.append(data)

        except Exception as e:
            self.logger.error(f"Error in websocket callback: {e}")

    async def _connect_and_listen(self):
        """Connect to websocket with proper cleanup"""
        self.logger.info(f"Connecting to websocket for {self.symbol}")

        if self.ws_client:
            try:
                self.connector.stop_websocket_stream(self.ws_client)
            except Exception as e:
                self.logger.warning(f"Error stopping existing websocket: {e}")
            self.ws_client = None

        try:
            fyers_symbol = f"NSE:{self.symbol}-EQ" if not self.symbol.startswith("NSE:") else self.symbol

            self.ws_client = self.connector.start_symbol_updates_stream(
                symbols=[fyers_symbol],
                callback=self._websocket_callback
            )

            if not self.ws_client:
                raise Exception("Failed to create WebSocket connection")

            self.connected = True
            self.last_message_time = time.time()
            self.monitor.report_connection_status(self.symbol, True)
            self.logger.info(f"Connected to websocket for {self.symbol}")

            # Monitor connection health
            while self.should_run and self._is_market_open():
                await asyncio.sleep(self.config.connection_check_interval)

                if time.time() - self.last_message_time > 180:
                    self.logger.warning(f"Stale connection for {self.symbol}, reconnecting...")
                    break

            if not self._is_market_open():
                self.logger.info(f"Market closed, stopping websocket for {self.symbol}")

        except Exception as e:
            self.logger.error(f"WebSocket connection error for {self.symbol}: {e}")
            raise
        finally:
            await self._cleanup_connection()

    async def _cleanup_connection(self):
        """Clean up connection resources"""
        if self.ws_client:
            try:
                self.connector.stop_websocket_stream(self.ws_client)
            except Exception as e:
                self.logger.error(f"Error closing websocket: {e}")

        self.ws_client = None
        self.connected = False
        self.monitor.report_connection_status(self.symbol, False)

    async def stop(self):
        """Stop the listener with proper cleanup"""
        self.logger.info(f"Stopping websocket listener for {self.symbol}")
        self.should_run = False

        # Cancel all background tasks
        for task in self.background_tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)

        # Complete any pending candle
        if self.current_candle and not self.current_candle.is_closed:
            await self._complete_candle(self.current_candle)

        # Cleanup connection
        await self._cleanup_connection()

        self.logger.info(f"Stopped websocket listener for {self.symbol}")

    def get_status(self) -> dict:
        """Get listener status with performance metrics"""
        candle_info = {}
        if self.current_candle:
            candle_dict = self.current_candle.to_dict_threadsafe(self.symbol)
            candle_info = {
                'current_candle_timestamp': self.current_minute_timestamp,
                'current_candle_trades': candle_dict['candle']['trades'],
                'current_candle_ohlc': {
                    'open': candle_dict['candle']['open'],
                    'high': candle_dict['candle']['high'],
                    'low': candle_dict['candle']['low'],
                    'close': candle_dict['candle']['close'],
                    'volume': candle_dict['candle']['volume']
                }
            }

        with self.tick_buffer_lock:
            buffer_size = len(self.tick_buffer)

        return {
            'symbol': self.symbol,
            'connected': self.connected,
            'market_open': self._is_market_open(),
            'last_message_time': self.last_message_time,
            'should_run': self.should_run,
            'time_until_market_open': self._time_until_market_open() if not self._is_market_open() else 0,
            'processed_ticks': self.processed_ticks,
            'dropped_ticks': self.dropped_ticks,
            'buffer_size': buffer_size,
            'background_tasks': len(self.background_tasks),
            **candle_info
        }