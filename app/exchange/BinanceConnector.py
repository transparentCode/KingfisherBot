import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from datetime import datetime
from binance.um_futures import UMFutures
from binance.lib.utils import config_logging
import websocket
import json
import threading
import ssl
import time

config_logging(logging, logging.DEBUG)

from app.utils.date_time_utils import DateTimeUtils


@dataclass
class BinanceConnectorConfig:
    def __init__(self):
        self.api_key = None
        self.api_secret = None
        self.logger_name = "app"


class BinanceConnector:
    """
    A class to connect to Binance API and fetch historical data.
    """

    def __init__(self, config: Optional[BinanceConnectorConfig] = None):
        """
        Initialize the BinanceConnector with API key and secret.
        """
        self.config = BinanceConnectorConfig() if config is None else config
        self.logger = logging.getLogger(self.config.logger_name)
        self.websockets = {}  # Store active connections
        if self.config.api_key and self.config.api_secret:
            self.client = UMFutures(key=self.config.api_key, secret=self.config.api_secret)
        else:
            self.client = UMFutures()

    def get_futures_klines(self, symbol: str, interval: str, start_time: int, end_time: int, limit=1000) -> pd.DataFrame:
        """
        Fetch historical data from Binance API.
        :param symbol: Trading pair symbol (e.g., 'BTCUSDT').
        :param interval: Time interval (e.g., '1m', '5m', '1h').
        :param start_time: Start time in milliseconds since epoch.
        :param end_time: End time in milliseconds since epoch.
        :param limit: Number of data points to fetch.

        :return: DataFrame containing historical OHLCV data.
        """
        klines = self.client.klines(
            symbol=symbol,
            interval=interval,
            startTime=start_time,
            endTime=end_time,
            limit=limit
        )

        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)

        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

        df['open_time'] = df['open_time'].apply(lambda x: DateTimeUtils.convert_to_ist(x))

        df.set_index('open_time', inplace=True)

        return df

    def get_historical_futures_klines(self, symbol: str, interval: str, limit=1000, start_time=None, end_time=None):
        """
        Fetch historical klines from Binance Futures API.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Time interval (e.g., '1m', '5m', '1h')
            limit: Maximum number of klines to return (max 1000)
            start_time: Start time in milliseconds
            end_time: End time in milliseconds

        Returns:
            List of klines where each kline is a list of values
        """
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }

        # Add optional parameters if provided
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time

        # Use klines method from UMFutures client
        klines = self.client.klines(**params)
        return klines

    def get_historical_data(self, symbol: str, interval: str, lookback_days=30) -> pd.DataFrame:
        """
        Fetch historical data from Binance API.
        :param symbol: Trading pair symbol (e.g., 'BTCUSDT').
        :param interval: Time interval (e.g., '1m', '5m', '1h').
        :param lookback_days: Number of data points to fetch.

        :return: DataFrame containing historical OHLCV data.
        """
        start_time = int((datetime.now().timestamp() - lookback_days * 24 * 60 * 60) * 1000)

        return self.get_futures_klines(symbol, interval, start_time, int(datetime.now().timestamp() * 1000))


    # no longer used
    async def subscribe_to_stream(self, symbol, stream_type):
        url = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@{stream_type}"

        # Check if connection exists and is open
        if symbol in self.websockets and self.websockets[symbol].open:
            self.logger.info(f"Already connected to {symbol} {stream_type} stream")
            return

        # Close existing connection if it's stale
        if symbol in self.websockets:
            try:
                await self.websockets[symbol].close()
            except Exception as e:
                self.logger.error(f"Error closing stale connection for {symbol}: {str(e)}")
                pass

        async def on_message(message):
            data = json.loads(message)

        # Create new connection
        self.websockets[symbol] = await self._create_websocket_connection(url, on_message)
        self.logger.info(f"Connected to {symbol} {stream_type} stream")

    # no longer used
    async def close_all_connections(self):
        for symbol, ws in self.websockets.items():
            try:
                await ws.close()
                self.logger.info(f"Closed connection for {symbol}")
            except Exception as e:
                self.logger.error(f"Error closing connection for {symbol}: {str(e)}")

        self.websockets = {}

    def _create_websocket_connection(self, stream_name: str, message_handler, symbol: str = None):
        """
        Create a WebSocket connection with standardized setup.
        """
        ws_url = "wss://fstream.binance.com/ws"

        # Keep a reference to the websocket to prevent early garbage collection
        active_ws = None

        def on_error(ws, error):
            self.logger.error(f"WebSocket error for {symbol}: {error}")

        def on_close(ws, close_status_code, close_msg):
            self.logger.info(f"WebSocket closed for {symbol}: {close_status_code}, {close_msg}")
            # Attempt reconnection
            if hasattr(ws, '_reconnect') and ws._reconnect:
                self.logger.info(f"Attempting to reconnect WebSocket for {symbol}...")
                time.sleep(5)

        def on_open(ws):
            try:
                symbol_info = f" for {symbol}" if symbol else ""
                self.logger.info(f"WebSocket connection opened{symbol_info}, subscribing...")
                subscription = {
                    "method": "SUBSCRIBE",
                    "params": [stream_name],
                    "id": 1
                }
                ws.send(json.dumps(subscription))
            except Exception as e:
                self.logger.error(f"Error in on_open for {symbol}: {str(e)}")

        # Create WebSocket connection with reconnect enabled
        ws = websocket.WebSocketApp(
            ws_url,
            on_open=on_open,
            on_message=message_handler,
            on_error=on_error,
            on_close=on_close
        )

        # Enable automatic reconnection
        ws._reconnect = True
        active_ws = ws

        # Run WebSocket in background thread
        ws_thread = threading.Thread(
            target=lambda: ws.run_forever(
                sslopt={"cert_reqs": ssl.CERT_NONE},
                ping_interval=30,
                ping_timeout=10,
                reconnect=5  # Enable auto-reconnect with 5 second delay
            ),
            daemon=True
        )
        ws_thread.start()

        stream_info = f" for {symbol}" if symbol else ""
        self.logger.info(f"WebSocket stream started{stream_info}")

        return ws, ws_thread

    def start_websocket_stream(self, symbol: str, interval: str, callback=None, closed_only: bool = True):
        """
        Start a kline WebSocket stream.
        """
        stream_name = f"{symbol.lower()}@kline_{interval}"

        def message_handler(ws, message):
            try:
                # Check if websocket is still connected before processing
                if not ws.sock or not ws.sock.connected:
                    self.logger.error(f"WebSocket for {symbol} disconnected, skipping message")
                    return

                self.logger.debug(f"Raw kline message received: {message[:200]}...")
                data = json.loads(message)

                if data.get("e") == "kline":
                    kline = data.get("k", {})
                    is_closed = kline.get("x", False)

                    if (not closed_only) or is_closed:
                        if callback:
                            callback(data)
                        else:
                            self.logger.debug(f"Kline: {kline.get('s')} {kline.get('i')} Close: {kline.get('c')}")
            except websocket.WebSocketConnectionClosedException:
                self.logger.warning(f"WebSocket connection closed while processing message for {symbol}")
            except Exception as e:
                self.logger.error(f"Error processing kline message: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())

        return self._create_websocket_connection(stream_name, message_handler, symbol)

    # no longer used
    def start_multipair_websocket(self, symbols: list, interval: str, callback=None, closed_only: bool = True):
        """
        Start WebSocket streams for multiple symbols with the same interval.

        :param symbols: List of trading pair symbols (e.g., ['BTCUSDT', 'ETHUSDT'])
        :param interval: Time interval (e.g., '1m', '5m', '1h')
        :param callback: Callback function to handle WebSocket messages
        :param closed_only: If True, only forward completed candles
        :return: Dictionary of WebSocket clients with symbols as keys
        """
        ws_clients = {}

        for symbol in symbols:
            stream_name = f"{symbol.lower()}@kline_{interval}"

            def create_message_handler(symbol_name):
                def message_handler(ws, message):
                    try:
                        data = json.loads(message)

                        if data.get("e") == "kline":
                            kline = data.get("k", {})
                            is_closed = kline.get("x", False)

                            if (not closed_only) or is_closed:
                                if callback:
                                    callback(data, symbol_name)
                                else:
                                    self.logger.info(f"Kline: {symbol_name} {kline.get('i')} Close: {kline.get('c')}")
                    except Exception as e:
                        self.logger.error(f"Error processing {symbol_name} kline: {str(e)}")

                return message_handler

            ws, thread = self._create_websocket_connection(
                stream_name,
                create_message_handler(symbol),
                symbol
            )
            ws_clients[symbol] = (ws, thread)

        return ws_clients

    def start_custom_stream(self, stream_type: str, symbol: str, callback=None):
        """
        Start a custom WebSocket stream.

        :param stream_type: Type of stream ('trade', 'depth', 'markprice', etc.)
        :param symbol: Trading pair symbol (e.g., 'BTCUSDT')
        :param callback: Function to receive the processed data
        :return: WebSocket client instance and thread
        """
        # Map stream_type to the appropriate stream name
        stream_params = {
            'trade': f"{symbol.lower()}@aggTrade",
            'depth': f"{symbol.lower()}@depth@100ms",
            'markprice': f"{symbol.lower()}@markPrice@1s",
            'bookticker': f"{symbol.lower()}@bookTicker",
            'kline': f"{symbol.lower()}@kline_1m",
            'miniticker': f"{symbol.lower()}@miniTicker",
        }

        if stream_type not in stream_params:
            raise ValueError(f"Unsupported stream type: {stream_type}")

        stream_name = stream_params[stream_type]

        def message_handler(ws, message):
            try:
                self.logger.debug(f"Raw {stream_type} message: {message[:200]}...")
                data = json.loads(message)
                if callback:
                    callback(data)
                else:
                    self.logger.info(f"{stream_type} data received for {symbol}")
            except Exception as e:
                self.logger.error(f"Error in {stream_type} message handler: {e}")

        return self._create_websocket_connection(stream_name, message_handler, symbol)

    def stop_websocket_stream(self, ws):
        """Safely stop a WebSocket connection"""
        if ws:
            try:
                ws.close()
                self.logger.info("WebSocket connection closed")
            except Exception as e:
                self.logger.error(f"Error closing WebSocket: {e}")


def test_improved_websocket():

    logging.basicConfig(level=logging.INFO)

    # Create connector
    config = BinanceConnectorConfig()
    connector = BinanceConnector(config)

    # Define callback
    def process_kline(data):
        kline = data.get("k", {})
        print(f"Candle data for symbol: {kline.get('s')} | Close: {kline.get('c')} | Time: {kline.get('t')}")

    # Start WebSocket
    ws, thread = connector.start_websocket_stream(
        symbol="BTCUSDT",
        interval="1m",
        callback=process_kline
    )

    # Let it run for some time
    print("WebSocket running, press Ctrl+C to stop...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up
        connector.stop_websocket_stream(ws)
        print("Test completed")


if __name__ == "__main__":
    bc = BinanceConnector()
    print(bc.get_historical_futures_klines('BTCUSDT', '1m', 10))
