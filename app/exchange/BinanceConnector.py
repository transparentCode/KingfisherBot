from app.utils.date_time_utils import DateTimeUtils
import logging
from dataclasses import dataclass
from typing import Optional, Callable

import pandas as pd
from datetime import datetime
from binance.um_futures import UMFutures
from binance.lib.utils import config_logging
import websocket
import json
import threading
import ssl
import time

if not logging.getLogger().handlers:
    config_logging(logging, logging.DEBUG)


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
        """
        try:
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

            # Use DateTimeUtils safely
            # try:
            #     df['open_time'] = df['open_time'].apply(lambda x: DateTimeUtils.convert_to_ist(x))
            # except Exception:
            #     # Fallback if DateTimeUtils fails or isn't imported correctly
            #     pass

            df.set_index('open_time', inplace=True)
            return df
        except Exception as e:
            self.logger.error(f"Error fetching klines for {symbol}: {e}")
            raise

    def get_historical_futures_klines(self, symbol: str, interval: str, limit=1000, start_time=None, end_time=None):
        """
        Fetch historical klines from Binance Futures API (Raw List).
        """
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }

        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time

        return self.client.klines(**params)

    def get_historical_data(self, symbol: str, interval: str, lookback_days=30) -> pd.DataFrame:
        """
        Fetch historical data from Binance API.
        """
        start_time = int((datetime.now().timestamp() - lookback_days * 24 * 60 * 60) * 1000)
        return self.get_futures_klines(symbol, interval, start_time, int(datetime.now().timestamp() * 1000))

    # -------------------------------------------------------------------------
    # Execution Methods
    # -------------------------------------------------------------------------

    def get_account_info(self):
        """Get account information including balances."""
        try:
            return self.client.account()
        except Exception as e:
            self.logger.error(f"Error fetching account info: {e}")
            raise

    def get_position_risk(self, symbol: str = None):
        """Get position risk for a specific symbol or all symbols."""
        try:
            return self.client.get_position_risk(symbol=symbol)
        except Exception as e:
            self.logger.error(f"Error fetching position risk for {symbol}: {e}")
            raise

    def place_order(self, symbol: str, side: str, type: str, quantity: float, price: float = None, **kwargs):
        """
        Place a new order.
        :param symbol: Trading symbol (e.g., 'BTCUSDT')
        :param side: 'BUY' or 'SELL'
        :param type: 'LIMIT', 'MARKET', 'STOP', etc.
        :param quantity: Order quantity
        :param price: Order price (required for LIMIT orders)
        :param kwargs: Additional arguments (timeInForce, stopPrice, etc.)
        """
        try:
            params = {
                'symbol': symbol,
                'side': side,
                'type': type,
                'quantity': quantity,
                **kwargs
            }
            if price:
                params['price'] = price
            
            self.logger.info(f"Placing order: {params}")
            return self.client.new_order(**params)
        except Exception as e:
            self.logger.error(f"Error placing order for {symbol}: {e}")
            raise

    def cancel_order(self, symbol: str, order_id: int):
        """Cancel an active order."""
        try:
            self.logger.info(f"Cancelling order {order_id} for {symbol}")
            return self.client.cancel_order(symbol=symbol, orderId=order_id)
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id} for {symbol}: {e}")
            raise
            
    def cancel_all_orders(self, symbol: str):
        """Cancel all open orders for a symbol."""
        try:
            self.logger.info(f"Cancelling all orders for {symbol}")
            return self.client.cancel_open_orders(symbol=symbol)
        except Exception as e:
            self.logger.error(f"Error cancelling all orders for {symbol}: {e}")
            raise

    def _create_websocket_connection(self, stream_name: str, message_handler: Callable, symbol: str = None):
        """
        Create a WebSocket connection with standardized setup.
        """
        ws_url = "wss://fstream.binance.com/ws"

        def on_error(ws, error):
            self.logger.error(f"WebSocket error for {symbol}: {error}")

        def on_close(ws, close_status_code, close_msg):
            self.logger.info(f"WebSocket closed for {symbol}: {close_status_code}, {close_msg}")

        def on_open(ws):
            try:
                self.logger.info(f"WebSocket connection opened for {symbol}, subscribing to {stream_name}...")
                subscription = {
                    "method": "SUBSCRIBE",
                    "params": [stream_name],
                    "id": 1
                }
                ws.send(json.dumps(subscription))
            except Exception as e:
                self.logger.error(f"Error in on_open for {symbol}: {str(e)}")

        # Create WebSocket connection
        ws = websocket.WebSocketApp(
            ws_url,
            on_open=on_open,
            on_message=message_handler,
            on_error=on_error,
            on_close=on_close
        )

        # Run WebSocket in background thread
        ws_thread = threading.Thread(
            target=lambda: ws.run_forever(
                sslopt={"cert_reqs": ssl.CERT_NONE},
                ping_interval=30,
                ping_timeout=10,
                reconnect=5
            ),
            daemon=True
        )
        ws_thread.start()

        self.logger.info(f"WebSocket stream started for {symbol}")
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
                    return

                data = json.loads(message)

                # Handle subscription confirmation
                if "result" in data and data["result"] is None:
                    self.logger.debug(f"Subscription confirmed for {symbol}")
                    return

                if data.get("e") == "kline":
                    kline = data.get("k", {})
                    is_closed = kline.get("x", False)

                    if (not closed_only) or is_closed:
                        if callback:
                            callback(data)
                        else:
                            self.logger.debug(f"Kline: {kline.get('s')} {kline.get('i')} Close: {kline.get('c')}")
            
            except json.JSONDecodeError:
                self.logger.error(f"Failed to decode JSON message for {symbol}")
            except Exception as e:
                self.logger.error(f"Error processing kline message for {symbol}: {str(e)}")

        return self._create_websocket_connection(stream_name, message_handler, symbol)

    def stop_websocket_stream(self, ws):
        """Safely stop a WebSocket connection"""
        if ws:
            try:
                ws.close()
                self.logger.info("WebSocket connection closed")
            except Exception as e:
                self.logger.error(f"Error closing WebSocket: {e}")

def binanceConnector_test():
    logging.basicConfig(level=logging.INFO)
    config = BinanceConnectorConfig()
    connector = BinanceConnector(config)
    df = connector.get_historical_data("BTCUSDT", "1h", lookback_days=1)
    print(df.head())


if __name__ == "__main__":
    binanceConnector_test()