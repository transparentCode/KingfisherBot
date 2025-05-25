import asyncio
import json
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
        """
        Initialize a websocket listener.

        Args:
            asset: Asset symbol to listen for
            write_queue: Queue to send received data to
            monitor: Monitoring system for reporting metrics
        """
        self.asset = asset
        self.write_queue = write_queue
        self.monitor = monitor
        self.websocket = None
        self.connected = False
        self.should_run = False
        self.last_message_time = 0
        self.reconnect_delay = 1  # Start with 1-second delay
        self.max_reconnect_delay = 60  # Maximum delay of 60 seconds
        self.config = WebSocketListenerConfig() if config is None else config
        self.logger = logging.getLogger(self.config.logger_name)
        self.connector = BinanceConnector()
        self.ws_client = None
        self.ws_thread = None
        self.loop = asyncio.get_running_loop()

    async def start(self):
        """Start the websocket listener."""
        self.logger.info(f"Starting websocket listener for {self.asset}")
        self.should_run = True

        while self.should_run:
            try:
                await self._connect_and_listen()
            except Exception as e:
                self.logger.error(f"Error in websocket listener for {self.asset}: {e}")

                # Implement exponential backoff for reconnection
                await asyncio.sleep(self.reconnect_delay)
                self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)
            else:
                # If we got here without exception, reset the reconnected delay
                self.reconnect_delay = 1

    async def _connect_and_listen(self):
        """Connect to websocket and listen for messages."""
        self.logger.info(f"Connecting to websocket for {self.asset}")

        # Reset any existing connections
        if self.ws_client:
            try:
                self.connector.stop_websocket_stream(self.ws_client)
            except:
                self.logger.warning(f"Error stopping existing websocket for {self.asset}")
            self.ws_client = None
            self.ws_thread = None

        # Create a callback that puts messages on the asyncio queue
        def message_callback(data):
            """
            Process incoming WebSocket message and put it on the asyncio queue.
            """
            try:
                # No need to parse data again, it's already a Python dict
                # Update last message time immediately
                self.last_message_time = time.time()
                self.monitor.report_message_received(self.asset)

                # If it's a kline message, process it
                if 'k' in data:
                    # Use run_coroutine_threadsafe to safely call async code from a different thread
                    asyncio.run_coroutine_threadsafe(
                        self._handle_incoming_message(data),
                        self.loop
                    )
                    self.logger.info(f"Message received for {self.asset}: {data}")
                else:
                    self.logger.debug(f"Received non-kline message for {self.asset}: {data}")
            except Exception as e:
                import traceback
                self.logger.error(f"Error processing message: {str(e)}\n{traceback.format_exc()}")

        # Start the websocket connection using BinanceConnector
        try:
            self.ws_client, self.ws_thread = self.connector.start_websocket_stream(
                symbol=self.asset,
                interval=self.config.interval,
                callback=message_callback,
                closed_only=True
            )

            self.connected = True
            self.last_message_time = time.time()  # Initialize time
            self.monitor.report_connection_status(self.asset, True)

            # Keep checking the connection
            while self.should_run:
                await asyncio.sleep(30)  # Check connection status every 30 seconds

                # Check if websocket is still connected
                if not (self.ws_client and self.ws_client.sock and self.ws_client.sock.connected):
                    self.logger.warning(f"WebSocket for {self.asset} disconnected, reconnecting...")
                    break

                # Check for stale connection
                if time.time() - self.last_message_time > 180:
                    self.logger.warning(f"No messages received for {self.asset} in last 3 minutes, reconnecting...")
                    break
        except Exception as e:
            self.logger.error(f"Error establishing WebSocket connection for {self.asset}: {str(e)}")
            raise
        finally:
            # Clean up connection in all cases
            if self.ws_client:
                try:
                    self.connector.stop_websocket_stream(self.ws_client)
                except Exception as e:
                    self.logger.error(f"Error closing WebSocket for {self.asset}: {str(e)}")

            self.ws_client = None
            self.ws_thread = None
            self.connected = False
            self.monitor.report_connection_status(self.asset, False)
            self.logger.info(f"Disconnected from websocket for {self.asset}")

    async def _handle_incoming_message(self, data):
        """Process websocket message asynchronously"""
        try:
            # Track last message time
            self.last_message_time = time.time()
            self.monitor.report_message_received(self.asset)

            # Process kline data
            if "k" in data:
                kline = data["k"]

                # Convert to common message format
                message = {
                    'symbol': self.asset,
                    'candle' : {
                        'timestamp': kline.get("t"),
                        'close': float(kline.get("c")),  # Use close price
                        'volume': float(kline.get("v")),  # Volume
                        'open': float(kline.get("o")),
                        'high': float(kline.get("h")),
                        'low': float(kline.get("l")),
                        'trades': kline.get("n"),
                        'interval': kline.get("i"),
                        'is_closed': kline.get("x", False)
                    }
                }

                # Send a message to write queue
                await self.write_queue.put(message)

        except Exception as e:
            self.logger.error(f"Error processing message for {self.asset}: {e}")

    async def stop(self):
        """Stop the websocket listener."""
        self.logger.info(f"Stopping websocket listener for {self.asset}")
        self.should_run = False

        if self.ws_client:
            self.connector.stop_websocket_stream(self.ws_client)
            self.ws_client = None
            self.ws_thread = None

        self.connected = False