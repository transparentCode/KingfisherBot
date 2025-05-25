import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
import time
from datetime import datetime, timedelta

from app.database.db_handler import DBConfig, DBHandler
from app.exchange.BinanceConnector import BinanceConnector
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
        self.previous_days_data = int(os.getenv("PREVIOUS_DAYS_DATA", 30))

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
        self.config = MarketServiceConfig() if config is None else config
        self.logger = logging.getLogger(self.config.logger_name)
        self.assets = self.config.assets

        # shared resources
        self.write_queue = asyncio.Queue()
        self.calc_queue = asyncio.Queue()

        # components init
        self.db_pool = None
        self.websocket_listeners = {}
        self.data_processors = []
        self.ta_processors = []
        self.scheduler = None
        self.monitor = None

        # State tracking
        self.running = False
        self.last_calculation = {asset: 0 for asset in self.assets}
        self.last_update = {asset: 0 for asset in self.assets}

    async def start(self):
        """
        Start all its subcomponents.
        """
        self.logger.info("Starting MarketService...")
        self.running = True

        # Initialize components
        self.db_pool = await create_db_pool(DBConfig())
        self.monitor = MonitoringSystem(self)
        await self.monitor.start()  # Start the monitoring system

        # Start components
        for i in range(4):
            writer = DBWriter(i, self.db_pool, self.write_queue, self.calc_queue)
            writer_task = asyncio.create_task(writer.start())
            self.data_processors.append({"task": writer_task, "service": writer})
            self.logger.info(f"Started DB writer worker {i}")

        for i in range(5):
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
        asyncio.create_task(self.scheduler.start())

        # First fetch historical data for all assets
        self.logger.info("Fetching historical data for all assets...")
        fetch_tasks = []
        for asset in self.assets:
            # Fetch 30 days of history by default
            fetch_task = asyncio.create_task(self.fetch_historical_data(asset, days=self.config.previous_days_data,))
            fetch_tasks.append(fetch_task)
            # Add delay between starting fetches to avoid API rate limits
            await asyncio.sleep(1)

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
                asset=asset,
                write_queue=self.write_queue,
                monitor=self.monitor
            )
            self.websocket_listeners[asset] = listener
            asyncio.create_task(listener.start())

        self.logger.info("Market data system started successfully")

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

    async def fetch_historical_data(self, asset: str, days: int = 30):
        """
        Fetch historical data for a specific asset and store in database
        """
        self.logger.info(f"Fetching {days} days of historical data for {asset}")
        try:
            connector = BinanceConnector()

            # Calculate start time (days ago from now in milliseconds

            end_time = int(time.time() * 1000)  # current time in ms
            start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

            total_candles = 0
            current_end_time = end_time
            batch_size = 1000  # Binance API limit per request

            candle_batches = []

            # Fetch data in batches from newest to oldest
            while current_end_time > start_time:
                self.logger.debug(
                    f"Fetching batch for {asset} ending at {datetime.fromtimestamp(current_end_time / 1000)}")

                # Use run_in_executor to call synchronous method in a separate thread
                loop = asyncio.get_running_loop()
                hist_data = await loop.run_in_executor(
                    None,
                    lambda: connector.get_historical_futures_klines(
                        symbol=asset,
                        interval="1m",
                        limit=batch_size,
                        end_time=current_end_time
                    )
                )

                if not hist_data or len(hist_data) == 0:
                    self.logger.warning(f"No more historical data for {asset}")
                    break

                # Format candles for database
                candles = []
                for kline in hist_data:
                    # Store timestamp as integer (milliseconds)
                    timestamp_ms = int(kline[0])

                    candle = {
                        'timestamp': timestamp_ms,  # Keep as integer timestamp
                        'open': float(kline[1]),
                        'high': float(kline[2]),
                        'low': float(kline[3]),
                        'close': float(kline[4]),
                        'volume': float(kline[5]),
                        'trades': int(kline[8]) if len(kline) > 8 and kline[8] else 0
                    }
                    candles.append(candle)

                candle_batches.append(candles)
                total_candles += len(candles)

                # Set new end time to the oldest candle timestamp minus 1ms
                if candles:
                    oldest_timestamp = min(c['timestamp'] for c in candles)
                    current_end_time = oldest_timestamp - 1
                else:
                    break

                # Add delay to avoid hitting rate limits
                await asyncio.sleep(0.25)

            # Store all candles in database (from oldest to newest)
            for batch in reversed(candle_batches):
                if batch:
                    count = await self.db_pool.write_candles(asset, '1', batch)
                    self.logger.info(f"Stored {count} historical candles for {asset}")
                    await asyncio.sleep(0.1)  # Small delay between database writes

            # Trigger initial calculation
            await self.calc_queue.put(asset)

            return total_candles
        except Exception as e:
            self.logger.error(f"Error fetching historical data for {asset}: {str(e)}")
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