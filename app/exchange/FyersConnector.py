import time

from fyers_apiv3 import fyersModel
import os
from dotenv import load_dotenv
import logging
from fyers_apiv3.FyersWebsocket import data_ws
import json

from app.utils.Data_Utils import Data_Utils

load_dotenv()


class FyersConnectorConfig:
    """
    Configuration class for FyersConnector.
    This class holds the API credentials and other configurations required for Fyers API access.
    """

    def __init__(self, client_id=None, secret_key=None, redirect_uri=None, response_type=None):
        self.client_id = os.getenv('FYERS_CLIENT_ID')
        self.secret_key = os.getenv('FYERS_SECRET_KEY')
        self.redirect_uri = os.getenv('FYERS_REDIRECT_URI')
        self.response_type = os.getenv('FYERS_RESPONSE_TYPE')
        self.logger_name = 'app'


class FyersConnector:
    _instance = None
    _is_initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FyersConnector, cls).__new__(cls)
        return cls._instance

    def __init__(self, config: FyersConnectorConfig = None):
        if not self._is_initialized:
            self._fyers_accessor = None
            self._fyers_app_session = None
            self._access_token = None
            self._is_initialized = True
            self.config = FyersConnectorConfig() if config is None else config
            self.logger = logging.getLogger(self.config.logger_name)
            self._load_persisted_token()

    @property
    def fyers_app_session(self):
        if self._fyers_app_session is None:
            if not all([self.config.client_id, self.config.redirect_uri, self.config.response_type,
                        self.config.secret_key]):
                raise ValueError("Please configure the FyersAccessorSingleton with API credentials first.")
            self._fyers_app_session = fyersModel.SessionModel(
                client_id=self.config.client_id,
                redirect_uri=self.config.redirect_uri,
                response_type=self.config.response_type,
                state="sample",
                secret_key=self.config.secret_key,
                grant_type="authorization_code"
            )
        return self._fyers_app_session

    def get_auth_url(self):
        """Public method to get the Fyers authorization URL"""
        return self._authorize_fyers_client()

    def get_access_token(self):
        """Public method to get the Fyers access token"""
        return self._access_token

    def _load_persisted_token(self):
        """Load token from storage if available"""
        token_file = "fyers_token.json"
        if os.path.exists(token_file):
            try:
                with open(token_file, 'r') as f:
                    token_data = json.load(f)

                # Check if a token is still valid
                if token_data.get('expiry', 0) > time.time():
                    self._access_token = token_data['access_token']
                    self.logger.info("Loaded valid token from storage")

                    # Initialize fyers accessor with loaded token
                    self._fyers_accessor = fyersModel.FyersModel(
                        token=self._access_token,
                        is_async=False,
                        client_id=self.config.client_id,
                        log_path=""
                    )
                    return True
            except Exception as e:
                self.logger.error(f"Error loading persisted token: {str(e)}")

        self.logger.info("No valid token available in storage")
        return False

    def _save_token(self):
        """Save current token to storage"""
        if self._access_token:
            token_file = "fyers_token.json"
            try:
                with open(token_file, 'w') as f:
                    json.dump({
                        'access_token': self._access_token,
                        'expiry': time.time() + 86400  # 24 hour validity
                    }, f)
                self.logger.info("Saved token to storage")
                return True
            except Exception as e:
                self.logger.error(f"Error saving token: {str(e)}")
        return False

    def _set_authorize_token(self, auth_code):
        if auth_code:
            self.fyers_app_session.set_token(auth_code)
            response = self.fyers_app_session.generate_token()

            try:
                self._access_token = response["access_token"]
                # Save token for reuse
                self._save_token()
            except KeyError as e:
                self.logger.error(f"Error retrieving access token: {e}")
                self.logger.info(f"Response: {response}")
                return None

        if self._access_token and self._fyers_accessor is None:
            self._fyers_accessor = fyersModel.FyersModel(
                token=self._access_token,
                is_async=False,
                client_id=self.config.client_id,
                log_path=""
            )
            return None
        return None

    def authorize(self, auth_code):
        if self._fyers_accessor is None:
            logging.getLogger('app').info("Authorizing for the first time")
            self._set_authorize_token(auth_code)
        else:
            logging.getLogger('app').info("Already authorized, returning existing instance")
        return self._fyers_accessor

    def _authorize_fyers_client(self):
        auth_url = self.fyers_app_session.generate_authcode()
        return auth_url

    def get_market_status(self):
        if self._fyers_accessor is not None:
            return self._fyers_accessor.market_status()

        self.logger.info("Fyers accessor is not initialized. Please authorize the client first.")
        return None

    # exchange values -
    # 10 	NSE (National Stock Exchange)
    # 11 	MCX (Multi Commodity Exchange)
    # 12 	BSE (Bombay Stock Exchange)

    # Segment values
    # 10 Capital Market
    # 11 Equity Derivatives
    # 12 Currency Derivatives
    # 20 Commodity Derivatives

    # status
    # CLOSE
    # OPEN
    # POSTCLOSE_START
    # POSTCLOSE_CLOSED
    # PREOPEN
    # PREOPEN_CLOSED

    # typical response
    # {
    #     "code":200,
    #     "marketStatus":[
    #         {
    #           "exchange":10,
    #           "market_type":"NORMAL",
    #           "segment":10,
    #           "status":"POSTCLOSE_CLOSED"
    #         },
    #         {
    #           "exchange":10,
    #           "market_type":"NORMAL",
    #           "segment":11,
    #           "status":"CLOSED"
    #         },
    #         {
    #           "exchange":10,
    #           "market_type":"NORMAL",
    #           "segment":12,
    #           "status":"CLOSED"
    #         }
    #     ]
    # }

    def fetch_historical_data(self, stock_name, candle_interval, date_format, from_date, to_date):

        # Check for Fyers authentication
        if not self._access_token:
            self.logger.warning(f"Cannot fetch historical data for {stock_name}: Fyers not authenticated")
            return 0

        data = {
            "symbol": "NSE:" + stock_name + "-EQ",
            "resolution": candle_interval,
            "date_format": date_format,
            "range_from": from_date,
            "range_to": to_date,
            "cont_flag": "1"
        }

        if self._fyers_accessor is not None:
            return self._fyers_accessor.history(data=data)

        self.logger.info("Fyers accessor is not initialized. Please authorize the client first.")
        return None

    def fetch_market_depth(self, stock_name):
        data = {
            "symbol": "NSE:" + stock_name + "-EQ",  # Fixed: removed duplicate "SBIN"
            "ohlcv_flag": "1"
        }

        if self._fyers_accessor is not None:
            return self._fyers_accessor.depth(data=data)

        self.logger.info("Fyers accessor is not initialized. Please authorize the client first.")
        return None

    def start_websocket_stream(self, data_type, symbols, callback=None):
        """
        Start a WebSocket connection using Fyers API v3 SDK

        Args:
            data_type: Type of data subscription ('symbolUpdate', 'depthUpdate', etc.)
            symbols: List of symbol names to subscribe to
            callback: Function to handle incoming messages

        Returns:
            FyersDataSocket object
        """
        if not self._access_token:
            self.logger.error("Access token not available. Please authorize first.")
            return None

        # Format symbols if needed - ensure they're properly formatted
        if isinstance(symbols, str):
            symbols = [symbols]

        # Ensure symbols are in correct format (NSE:SYMBOL-EQ)
        formatted_symbols = []
        for symbol in symbols:
            if not symbol.startswith("NSE:") and not symbol.startswith("BSE:"):
                symbol = f"NSE:{symbol}-EQ"
            formatted_symbols.append(symbol)

        # Create access token in correct format: client_id:access_token
        access_token = f"{self.config.client_id}:{self._access_token}"

        # Define callback functions
        def on_message(message):
            try:
                if callback:
                    callback(message)
                else:
                    self.logger.debug(f"WebSocket data received: {message}")
            except Exception as e:
                self.logger.error(f"Error processing WebSocket message: {str(e)}")

        def on_error(error):
            self.logger.error(f"WebSocket error: {str(error)}")

        def on_close():
            self.logger.info("WebSocket connection closed")

        def on_open():
            self.logger.info("WebSocket connection opened")
            # Subscribe to data after connection is established
            data = {
                "T": "SUB_DATA",
                "dataType": data_type,
                "symbols": formatted_symbols
            }
            # Send subscription message
            fs.subscribe(symbols=formatted_symbols, data_type=data_type)

        # Create WebSocket connection with corrected parameters
        fs = data_ws.FyersDataSocket(
            access_token=access_token,  # Access token in format "client_id:access_token"
            litemode=False,  # Set to True for lite response
            write_to_file=False,  # Set to True to save responses to file
            reconnect=True,  # Enable auto-reconnection
            on_connect=on_open,  # Called when connection is established
            on_close=on_close,  # Called when connection is closed
            on_error=on_error,  # Called on errors
            on_message=on_message  # Called when message is received
        )

        # Connect to WebSocket
        fs.connect()

        return fs

    def start_symbol_updates_stream(self, symbols, callback=None):
        """
        Start a WebSocket stream for symbol updates (real-time quotes)

        Args:
            symbols: List of symbol names or single symbol (e.g., ["NSE:SBIN-EQ", "NSE:RELIANCE-EQ"])
            callback: Function to handle incoming messages

        Returns:
            FyersDataSocket object
        """
        return self.start_websocket_stream("symbolUpdate", symbols, callback)

    def start_depth_updates_stream(self, symbols, callback=None):
        """
        Start a WebSocket stream for market depth updates

        Args:
            symbols: List of symbol names or single symbol
            callback: Function to handle incoming messages

        Returns:
            FyersDataSocket object
        """
        return self.start_websocket_stream("depthUpdate", symbols, callback)

    def start_orderupdate_stream(self, callback=None):
        """
        Start a WebSocket stream for order updates

        Args:
            callback: Function to handle incoming messages

        Returns:
            FyersDataSocket object
        """
        # Order updates don't require symbols
        return self.start_websocket_stream("orderUpdate", [], callback)

    def stop_websocket_stream(self, ws):
        """
        Safely stop a WebSocket connection

        Args:
            ws: FyersDataSocket object to stop
        """
        if ws:
            try:
                ws.close()
                self.logger.info("WebSocket connection closed")
            except Exception as e:
                self.logger.error(f"Error closing WebSocket: {str(e)}")


def run():
    fyers_singleton = FyersConnector()

    auth_url = fyers_singleton._authorize_fyers_client()
    print(f"Please visit this URL to authorize: {auth_url}")
    auth_code = input("Enter the auth code from the redirect URL: ")

    fyers_accessor = fyers_singleton.authorize(auth_code=auth_code)

    # This will return the same instance
    fyers_accessor_2 = fyers_singleton.authorize(auth_code="different_auth_code")

    print(f"Are fyers_accessor instances the same? {fyers_accessor is fyers_accessor_2}")

    # Fixed: removed fyers_accessor parameter (not needed)
    data = fyers_singleton.fetch_historical_data(
        "SBIN",
        "1",
        "1",
        "2025-05-08",
        "2025-06-07"
    )

    print(f"transform data: ", Data_Utils.transform_data(data).info())

    # Test websocket with proper instance method call
    test_websocket(fyers_singleton)


def test_websocket(fyers_connector):
    """Test the Fyers WebSocket connection"""

    def on_tick(data):
        print(f"Received tick data: {data}")

    symbols = ["NSE:SBIN-EQ"]  # Valid symbol format

    # Start WebSocket stream
    ws = fyers_connector.start_symbol_updates_stream(symbols, on_tick)

    if ws:
        print("WebSocket running for 30 seconds...")
        try:
            import time
            time.sleep(30)
        finally:
            fyers_connector.stop_websocket_stream(ws)
            print("WebSocket test completed")
    else:
        print("Failed to start WebSocket connection")


if __name__ == '__main__':
    run()