import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from datetime import datetime, timedelta

from app.database.db_handler import DBConfig, DBHandler
from app.exchange.FyersConnector import FyersConnector
from app.services.calc_scheduler_service import CalcSchedulerService
from app.services.db_writer_service import DBWriter
from app.services.indicator_calc_service import IndicatorCalcService
from app.services.monitoring_service import MonitoringSystem
from app.services.websocket_listener_service import WebSocketListenerService

dotenv = load_dotenv()


@dataclass
class MarketServiceConfig:
    def __init__(self):
        self.logger_name = "app"
        self.assets = os.getenv("ASSETS").split(',')
        self.previous_days_data = int(os.getenv("PREVIOUS_DAYS_DATA", 90))


async def create_db_pool(config: DBConfig) -> DBHandler:
    handler = DBHandler(config)
    await handler.initialize()
    await handler.create_candles_table()
    return handler


class MarketService:
    """
    Main system for managing market data and manages all components.
    """

    def __init__(self, config: Optional[MarketServiceConfig] = None):
        self.fyers_auth_pending = None
        self.config = MarketServiceConfig() if config is None else config
        self.logger = logging.getLogger(self.config.logger_name)
        self.assets = self.config.assets

        # shared resources
        self.write_queue = asyncio.Queue(maxsize=10000)
        self.calc_queue = asyncio.Queue(maxsize=5000)

        # components init
        self.db_pool = None
        self.websocket_listeners = {}
        self.data_processors = []
        self.ta_processors = []
        self.scheduler = None
        self.monitor = None

        # Fyers connector instance
        self.fyers_connector = FyersConnector()

        # State tracking
        self.running = False
        self.last_calculation = {asset: 0 for asset in self.assets}
        self.last_update = {asset: 0 for asset in self.assets}

        self.background_tasks = set()
        self.websocket_tasks = {}


    async def start(self):
        """Initialize and start the market service"""
        self.logger.info("Starting MarketService...")

        # Start other non-Fyers dependent services
        await self._start_core_services()

        # Check if Fyers connector is authenticated
        if not self.fyers_connector.get_access_token():
            self.logger.warning("Fyers not authenticated. Historical data fetching will be delayed.")
            self.fyers_auth_pending = True
        else:
            self.fyers_auth_pending = False
            # Proceed with regular initialization
            await self._initialize_with_historical_data()

        self.running = True
        self.logger.info("MarketService started")

    async def _initialize_with_historical_data(self):
        # First fetch historical data for all assets
        self.logger.info("Fetching historical data for all assets...")
        fetch_tasks = []
        for asset in self.assets:
            # Fetch 90 days of history by default
            fetch_task = asyncio.create_task(self.fetch_historical_data(asset, days=self.config.previous_days_data,))
            fetch_tasks.append(fetch_task)
            # Add delay between starting fetches to avoid API rate limits
            await asyncio.sleep(5)

        # Wait for all historical data to be fetched
        if fetch_tasks:
            results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"Failed to fetch historical data for {self.assets[i]}: {result}")
                else:
                    self.logger.info(f"Successfully loaded {result} historical candles for {self.assets[i]}")

        # Start websocket listeners for all assets
        for asset in self.assets:
            listener = WebSocketListenerService(
                symbol=asset,
                write_queue=self.write_queue,
                monitor=self.monitor,
                fyers_connector=self.fyers_connector,
            )
            self.websocket_listeners[asset] = listener

            task = asyncio.create_task(listener.start())
            # Properly track the task
            self.websocket_tasks[asset] = task
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)

    async def _start_core_services(self):
        # Scale workers based on asset count
        db_worker_count = min(len(self.assets), 3)  # Max 3, min 1
        calc_worker_count = min(len(self.assets) * 2, 8)  # 2 per asset, max 8

        """Start core services that don't depend on Fyers authentication"""
        self.db_pool = await create_db_pool(DBConfig())
        self.monitor = MonitoringSystem(self)
        await self.monitor.start()  # Start the monitoring system

        # Start components
        for i in range(db_worker_count):
            writer = DBWriter(i, self.db_pool, self.write_queue, self.calc_queue)
            writer_task = asyncio.create_task(writer.start())
            self.data_processors.append({"task": writer_task, "service": writer})
            self.logger.info(f"Started DB writer worker {i}")

        for i in range(calc_worker_count):
            calculator = IndicatorCalcService(
                calculator_id=i,
                calc_queue=self.calc_queue,
                db_pool=self.db_pool,
                last_calculation=self.last_calculation
            )
            calc_task = asyncio.create_task(calculator.start())
            self.ta_processors.append({"task": calc_task, "service": calculator})
            self.logger.info(f"Started calculator {i}")

        self.scheduler = CalcSchedulerService(
            assets=self.assets,
            calc_queue=self.calc_queue,
            last_calculation=self.last_calculation
        )
        scheduler_task = asyncio.create_task(self.scheduler.start())

        self.background_tasks.add(scheduler_task)

    async def initialize_after_auth(self):
        """
        Initialize the market service after Fyers authentication is complete.
        This method is called after the user has authenticated with Fyers and the access token is available.
        """
        await self._initialize_with_historical_data()

        self.fyers_auth_pending = False
        self.logger.info("Market service initialized after authentication")
        return True

    async def get_status(self):
        """Return the current status of the market service components."""
        status = {
            "service": "running",
            "db_connection": "connected" if self.db_pool and self.db_pool.read_pool else "disconnected",
            "assets": {}
        }

        # Add status for each asset being monitored
        for asset in self.assets:
            asset_status = {
                "websocket": "connected" if asset in self.websocket_listeners else "disconnected",
                "last_update": self.last_update.get(asset, 0),
                "last_calculation": self.last_calculation.get(asset, 0)
            }
            status["assets"][asset] = asset_status

        # Add worker statuses
        status["workers"] = {
            "db_writers": self.data_processors,
            "calculators": self.ta_processors
        }

        return status

    async def fetch_historical_data(self, asset: str, days: int = 60):
        """
        Fetch historical data for a specific asset using Fyers API and store in a database
        """
        self.logger.info(f"Fetching {days} days of historical data for {asset}")
        try:
            # Calculate date range for Fyers API
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            # Format dates for Fyers API (YYYY-MM-DD format)
            from_date = start_date.strftime("%Y-%m-%d")
            to_date = end_date.strftime("%Y-%m-%d")

            total_candles = 0

            # Fyers API parameters
            candle_interval = "1"  # 1 minute
            date_format = "1"  # Unix timestamp format

            self.logger.info(f"Fetching historical data for {asset} from {from_date} to {to_date}")

            # Use run_in_executor to call synchronous Fyers method in a separate thread
            loop = asyncio.get_running_loop()
            hist_data = await loop.run_in_executor(
                None,
                lambda: self.fyers_connector.fetch_historical_data(
                    stock_name=asset,
                    candle_interval=candle_interval,
                    date_format=date_format,
                    from_date=from_date,
                    to_date=to_date
                ),
            )

            if not hist_data or hist_data.get('code') != 200:
                error_msg = hist_data.get('message', 'Unknown error') if hist_data else 'No data returned'
                self.logger.error(f"Failed to fetch historical data for {asset}: {error_msg}")
                return 0

            candles_data = hist_data.get('candles', [])
            if not candles_data:
                self.logger.info(f"No historical candles data for {asset}")
                return 0

            self.logger.info(f"Fetched historical candles for {asset} with size " f"{len(candles_data)} candles")

            # Process candles data from Fyers format
            # Fyers candles format: [timestamp, open, high, low, close, volume]
            candles = []
            for candle_data in candles_data:
                if len(candle_data) >= 6:
                    # Convert timestamp to milliseconds if it's in seconds
                    timestamp = int(candle_data[0])
                    if timestamp < 10000000000:  # If timestamp is in seconds, convert to milliseconds
                        timestamp = timestamp * 1000

                    candle = {
                        'symbol': asset,
                        'timestamp': timestamp,
                        'open': float(candle_data[1]),
                        'high': float(candle_data[2]),
                        'low': float(candle_data[3]),
                        'close': float(candle_data[4]),
                        'volume': float(candle_data[5]),
                        'interval': '1'  # 1 minute interval
                    }
                    candles.append(candle)

            total_candles = len(candles)

            if candles:
                # Store candles in a database
                count = await self.db_pool.write_candles(asset, '1', candles)
                self.logger.info(f"Stored {count} historical candles for {asset}")

                # Trigger initial calculation
                await self.calc_queue.put(asset)
            else:
                self.logger.warning(f"No valid candles processed for {asset}")

            return total_candles

        except Exception as e:
            self.logger.error(f"Error fetching historical data for {asset}: {str(e)}")
            # Log additional details about the error
            import traceback
            self.logger.info(f"Full traceback: {traceback.format_exc()}")
            raise

    async def stop(self):
        self.logger.info("Stopping MarketService...")
        self.running = False

        # Stop monitoring a system first
        if self.monitor:
            await self.monitor.stop()

        # Stop all websocket listeners
        stop_tasks = []
        for asset, listener in self.websocket_listeners.items():
            stop_tasks.append(listener.stop())

        if stop_tasks:
            await asyncio.gather(*stop_tasks, return_exceptions=True)

        # Stop scheduler
        if self.scheduler:
            await self.scheduler.stop()

        if not self.write_queue.empty():
            self.logger.info("Waiting for write queue to drain...")
            try:
                # Set a timeout to avoid hanging forever
                await asyncio.wait_for(self.write_queue.join(), timeout=10.0)
            except asyncio.TimeoutError:
                self.logger.warning("Timed out waiting for write queue to drain")

        # Stop all calculator services
        for proc in self.ta_processors:
            await proc["service"].stop()
            proc["task"].cancel()

        # Stop all DB writer services
        for proc in self.data_processors:
            await proc["service"].stop()
            proc["task"].cancel()

        # Wait for all tasks to complete with exception handling
        all_tasks = [p["task"] for p in self.data_processors + self.ta_processors]
        if all_tasks:
            await asyncio.gather(*all_tasks, return_exceptions=True)

        # Close DB pool
        if self.db_pool:
            await self.db_pool.close()

        self.logger.info("MarketService stopped")

    async def add_to_write_queue(self, symbol: str, candle: dict):
        await self.write_queue.put({'symbol': symbol, 'candle': candle})