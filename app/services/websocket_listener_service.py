import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional
from app.exchange.BinanceConnector import BinanceConnector
from app.services.monitoring_service import MonitoringSystem


@dataclass
class WebSocketListenerConfig:
    def __init__(self):
        self.logger_name = "app"
        self.interval = "1m"  # Default interval for kline data


class WebSocketListenerService:
    def __init__(self, asset: str, write_queue: asyncio.Queue, monitor: MonitoringSystem,
                 config: Optional[WebSocketListenerConfig] = None):
        self.asset = asset
        self.write_queue = write_queue
        self.monitor = monitor
        self.config = WebSocketListenerConfig() if config is None else config
        self.logger = logging.getLogger(self.config.logger_name)
        
        # State
        self.connected = False
        self.should_run = False
        self.last_message_time = 0
        self.reconnect_delay = 1
        self.max_reconnect_delay = 60
        
        # Connector components
        self.connector = BinanceConnector()
        self.ws_client = None
        self.ws_thread = None
        self.loop = None

    async def start(self):
        """Start the websocket listener."""
        self.loop = asyncio.get_running_loop()
        self.logger.info(f"Starting websocket listener for {self.asset}")
        self.should_run = True

        while self.should_run:
            try:
                await self._connect_and_listen()
            except Exception as e:
                self.logger.error(f"Error in websocket listener for {self.asset}: {e}")
                # Exponential backoff
                await asyncio.sleep(self.reconnect_delay)
                self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)
            else:
                self.reconnect_delay = 1

    async def _connect_and_listen(self):
        """Connect to websocket and listen for messages."""
        self.logger.info(f"Connecting to websocket for {self.asset}")

        # Cleanup previous connection
        self._cleanup_connection()

        # --- THREAD BRIDGE ---
        def message_callback(data):
            """
            Strictly a bridge. No logic here.
            Passes data from Thread -> Async Loop safely.
            """
            if self.should_run and self.loop:
                asyncio.run_coroutine_threadsafe(
                    self._handle_incoming_message(data),
                    self.loop
                )
        # ---------------------

        try:
            # Start synchronous websocket in a thread
            self.ws_client, self.ws_thread = self.connector.start_websocket_stream(
                symbol=self.asset,
                interval=self.config.interval,
                callback=message_callback,
                closed_only=True
            )

            self.connected = True
            self.last_message_time = time.time()
            self.monitor.report_connection_status(self.asset, True)

            # Monitor connection health
            while self.should_run:
                await asyncio.sleep(10) # Check every 10 seconds

                # 1. Stale Data Check (Heartbeat)
                if time.time() - self.last_message_time > 180: # 3 minutes
                    self.logger.warning(f"No messages for {self.asset} > 3m. Reconnecting...")
                    break
                
                # 2. Library Connection Check (Safe wrapper)
                try:
                    if self.ws_client and hasattr(self.ws_client, 'sock') and self.ws_client.sock:
                        if not self.ws_client.sock.connected:
                            self.logger.warning(f"Socket disconnected for {self.asset}")
                            break
                except Exception:
                    pass # Ignore internal library attribute errors

        except Exception as e:
            self.logger.error(f"WebSocket connection failed for {self.asset}: {e}")
            raise
        finally:
            self._cleanup_connection()

    def _cleanup_connection(self):
        """Helper to safely stop the websocket client"""
        if self.ws_client:
            try:
                self.connector.stop_websocket_stream(self.ws_client)
            except Exception as e:
                self.logger.error(f"Error stopping websocket for {self.asset}: {e}")
        
        self.ws_client = None
        self.ws_thread = None
        
        if self.connected:
            self.connected = False
            self.monitor.report_connection_status(self.asset, False)
            self.logger.info(f"Disconnected websocket for {self.asset}")

    async def _handle_incoming_message(self, data):
        """
        Process websocket message asynchronously.
        ALL logic (parsing, metrics, logging) happens here.
        """
        try:
            self.last_message_time = time.time()
            
            # Only report metrics here (Thread Safe)
            self.monitor.report_message_received(self.asset)

            if "k" in data:
                kline = data["k"]
                
                message = {
                    'symbol': self.asset,
                    'candle': {
                        'timestamp': kline.get("t"),
                        'close': float(kline.get("c")),
                        'volume': float(kline.get("v")),
                        'open': float(kline.get("o")),
                        'high': float(kline.get("h")),
                        'low': float(kline.get("l")),
                        'trades': kline.get("n"),
                        'interval': kline.get("i"),
                        'is_closed': kline.get("x", False)
                    }
                }

                await self.write_queue.put(message)

        except Exception as e:
            self.logger.error(f"Error processing message for {self.asset}: {e}", exc_info=True)

    async def stop(self):
        """Stop the websocket listener."""
        self.logger.info(f"Stopping websocket listener for {self.asset}")
        self.should_run = False
        self._cleanup_connection()