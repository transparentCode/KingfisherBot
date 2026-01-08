from app.db.redis_handler import RedisHandler
import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
import time
from datetime import datetime, timedelta

from app.db.db_handler import DBConfig, DBHandler
from app.exchange.BinanceConnector import BinanceConnector
from app.services import IndicatorRegistry
from app.services.calc_scheduler_service import CalcSchedulerService
from app.services.db_writer_service import DBWriter
from app.services.indicator_calc_service import IndicatorCalcService
from app.services.monitoring_service import MonitoringSystem
from app.services.data_cleanup_service import DataCleanupService
from app.services.websocket_listener_service import WebSocketListenerService
from config.asset_indicator_config import ConfigurationManager
from app.enums.AssetStatus import AssetStatus
from app.services.bar_close_service import BarCloseService
from app.db.mtf_data_manager import MTFDataManager

# Execution & Risk Imports
from app.execution.binance_execution import BinanceExecutionService
from app.execution.paper_execution import PaperExecutionService
from app.execution.execution_router import ExecutionRouter
from app.services.safety_service import SafetyService
from app.risk.risk_manager import RiskManager

dotenv = load_dotenv()

@dataclass
class MarketServiceConfig:
    def __init__(self):
        self.logger_name = "app"
        self.assets = os.getenv("ASSETS").split(',')
        self.previous_days_data = int(os.getenv("PREVIOUS_DAYS_DATA", 30))
        self.db_worker_count = int(os.getenv("DB_WORKER_COUNT", 4))
        self.calc_worker_count = int(os.getenv("CALC_WORKER_COUNT", 5))
        self.data_concurrency_limit = int(os.getenv("DATA_CONCURRENCY_LIMIT", 10))

async def create_db_pool(config: DBConfig) -> DBHandler:
    handler = DBHandler(config)
    await handler.initialize()
    await handler.create_candles_table()
    await handler.create_regime_table()
    return handler

class MarketService:
    """
    Main system for managing market data and manages all components.
    """

    def __init__(self, config: Optional[MarketServiceConfig] = None):
        self.config = MarketServiceConfig() if config is None else config
        self.logger = logging.getLogger(self.config.logger_name)
        self.assets = self.config.assets
        self.config_manager = ConfigurationManager()
        self.redis_handler = RedisHandler()
        self.db_worker_count = self.config.db_worker_count
        self.calc_worker_count = self.config.calc_worker_count
        self.init_semaphore = asyncio.Semaphore(self.config.data_concurrency_limit)

        # shared resources
        self.write_queue = asyncio.Queue()
        self.calc_queue = asyncio.Queue()
        self.bar_close_queue = asyncio.Queue()
        self.loop = asyncio.get_event_loop()

        # components init
        self.db_pool = None
        self.mtf_data_manager = None
        self.bar_close_service = None
        self.websocket_listeners = {}
        self.data_processors = []
        self.ta_processors = []
        self.scheduler = None
        self.monitor = None
        self.cleanup_service = None

        # State tracking
        self.running = False
        self.asset_states = {}
        self.last_calculation = {asset: 0 for asset in self.assets}
        self.last_update = {asset: 0 for asset in self.assets}

        # Initialize Execution & Risk Services
        self._initialize_execution_services()

    def _initialize_execution_services(self):
        """Initialize Execution, Safety, and Risk services."""
        try:
            # 1. Connectors
            self.binance_connector = BinanceConnector() # Uses env vars
            
            # 2. Execution Services
            self.binance_execution = BinanceExecutionService(self.binance_connector)
            self.paper_execution = PaperExecutionService()
            
            # 3. Router
            self.execution_router = ExecutionRouter(
                binance_service=self.binance_execution,
                paper_service=self.paper_execution
            )
            
            # 4. Safety Service
            self.safety_service = SafetyService(self.execution_router)
            
            # 5. Risk Manager
            self.risk_manager = RiskManager(self.config_manager)
            
            self.logger.info("Execution and Risk services initialized successfully.")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize execution services: {e}")
            raise

    def _log_configuration_summary(self):
        """Log detailed configuration summary at startup"""
        try:
            self.logger.info("=== Configuration Summary ===")
            
            summary = self.config_manager.get_configuration_summary()
            
            self.logger.info(f"Total Assets: {summary['total_assets']}")
            self.logger.info(f"Enabled Assets: {summary['enabled_assets']}")
            self.logger.info(f"Disabled Assets: {summary['disabled_assets']}")
            self.logger.info(f"Regime Adaptation Global: {'ON' if summary['regime_adaptation_global'] else 'OFF'}")
            self.logger.info(f"Regime Enabled Assets: {summary['regime_enabled_assets']}")
            self.logger.info(f"Runtime Overrides: {summary['runtime_overrides_count']} ({summary['runtime_overrides_assets']})")
            self.logger.info(f"Config Directory: {summary['config_dir']}")
            self.logger.info(f"Last Reload: {summary['last_reload']}")
            
            # Global configuration details
            global_config = summary.get('global_config', {})
            if global_config:
                regime = global_config.get('regime_adaptation', {})
                self.logger.info(f"Global Regime Settings: enabled={regime.get('enabled', False)}, interval={regime.get('update_interval', 'N/A')}")
                
                timeframes = global_config.get('default_timeframes', [])
                self.logger.info(f"Default Timeframes: {timeframes}")
                
                calc_settings = global_config.get('calculation_settings', {})
                self.logger.info(f"Calculation Settings: parallel={calc_settings.get('parallel_processing', False)}, workers={calc_settings.get('max_workers', 1)}")
            
            # Detailed asset configurations
            asset_details = summary.get('asset_details', {})
            if asset_details:
                self.logger.info("=== Asset Configurations ===")
                for asset, config in asset_details.items():
                    self.logger.info(f"Asset: {asset}")
                    self.logger.info(f"  - Enabled: {config.get('enabled', True)}")
                    self.logger.info(f"  - Regime Adaptation: {config.get('regime_adaptation_enabled', True)}")
                    
                    # MA configs
                    ma_configs = config.get('ma_configs', {})
                    if ma_configs:
                        self.logger.info(f"  - MA Configs: {ma_configs}")
                    
                    # SuperTrend configs
                    st_configs = config.get('supertrend_configs', {})
                    if st_configs:
                        self.logger.info(f"  - SuperTrend Configs: {st_configs}")
                    
                    # Oscillator configs
                    osc_configs = config.get('oscillator_configs', {})
                    if osc_configs:
                        self.logger.info(f"  - Oscillator Configs: {osc_configs}")
                    
                    # Timeframe overrides
                    tf_overrides = config.get('timeframe_overrides', {})
                    if tf_overrides:
                        self.logger.info(f"  - Timeframe Overrides: {tf_overrides}")
            
            # Runtime override details
            runtime_details = summary.get('runtime_override_details', {})
            if runtime_details:
                self.logger.info("=== Runtime Overrides ===")
                for asset, override in runtime_details.items():
                    self.logger.info(f"Runtime Override for {asset}: {override}")
            
            # Configuration validation
            issues = self.config_manager.validate_configuration()
            if issues['errors']:
                self.logger.error(f"Configuration Errors: {issues['errors']}")
            if issues['warnings']:
                self.logger.warning(f"Configuration Warnings: {issues['warnings']}")
            if issues.get('info'):
                self.logger.info(f"Configuration Info: {issues['info']}")
            
            self.logger.info("=== End Configuration Summary ===")
            
        except Exception as e:
            self.logger.error(f"Failed to log configuration summary: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")

    def _validate_asset_configuration(self):
        env_assets = set(self.assets)
        configured_assets = set(self.config_manager.get_asset_names())
        
        # Add missing assets to configuration
        missing_assets = env_assets - configured_assets
        for asset in missing_assets:
            self.logger.info(f"Adding missing asset {asset} to configuration")
            # This will create default config and save to file
            self.config_manager.get_base_asset_config(asset)
        
        # Log assets that are configured but not in environment
        extra_assets = configured_assets - env_assets
        if extra_assets:
            self.logger.info(f"Assets in config but not in environment: {extra_assets}")

    async def initialize_asset(self, asset: str):
        """
        Initialize a single asset: Fetch history -> Start WebSocket
        """
        try:
            async with self.init_semaphore:
                self.logger.info(f"Initializing asset {asset}...")
                self.asset_states[asset] = AssetStatus.FETCHING_HISTORY.value
            
                # 1. Fetch History
                await self.fetch_historical_data(asset, days=self.config.previous_days_data)
            
            # 2. Start WebSocket
            self.asset_states[asset] = AssetStatus.STARTING_STREAM.value
            
            listener = WebSocketListenerService(
                asset=asset,
                write_queue=self.write_queue,
                monitor=self.monitor,
                bar_close_queue=self.bar_close_queue
            )
            asyncio.create_task(listener.start())
            self.websocket_listeners[asset] = listener
            
            self.asset_states[asset] = AssetStatus.READY.value
            self.logger.info(f"Asset {asset} is fully initialized and running.")
            
        except Exception as e:
            self.asset_states[asset] = AssetStatus.ERROR.value
            self.logger.error(f"Failed to initialize asset {asset}: {e}")

    async def start(self):
        """
        Start all its subcomponents.
        """
        self.logger.info("Starting MarketService...")
        self.running = True

        self._validate_asset_configuration()
        await asyncio.sleep(0.1)
        self._log_configuration_summary()

        # Initialize components
        await self.redis_handler.initialize()
        self.db_pool = await create_db_pool(DBConfig())
        self.monitor = MonitoringSystem(self)
        await self.monitor.start()  # Start the monitoring system

        # Initialize MTF Data Manager and Bar Close Service
        self.mtf_data_manager = MTFDataManager(self.db_pool)
        monitored_timeframes = ['15m', '30m', '1h', '2h', '4h'] 
        self.bar_close_service = BarCloseService(
            queue=self.bar_close_queue,
            calc_queue=self.calc_queue,
            mtf_data_manager=self.mtf_data_manager,
            timeframes=monitored_timeframes
        )
        asyncio.create_task(self.bar_close_service.start())

        # Start Data Cleanup Service
        # Determine charts directory (default is ./charts in root)
        charts_dir = os.path.join(os.getcwd(), 'charts')
        
        self.cleanup_service = DataCleanupService(
            self.db_pool, 
            retention_days=self.config.previous_days_data,
            check_interval_hours=24,
            charts_dir=charts_dir
        )
        # Start as a background task
        asyncio.create_task(self.cleanup_service.start())

        # Start components
        for i in range(self.db_worker_count):
            writer = DBWriter(i, self.db_pool, self.write_queue, self.calc_queue)
            writer_task = asyncio.create_task(writer.start())
            self.data_processors.append({"task": writer_task, "service": writer})
            self.logger.info(f"Started DB writer worker {i}")

        for i in range(self.calc_worker_count):
            calculator = IndicatorCalcService(
                calculator_id=i,
                calc_queue=self.calc_queue,
                db_pool=self.db_pool,
                last_calculation=self.last_calculation,
                indicator_registry=IndicatorRegistry(),
                config_manager=self.config_manager,
                execution_router=self.execution_router,
                safety_service=self.safety_service,
                risk_manager=self.risk_manager
            )
            calc_task = asyncio.create_task(calculator.start())
            self.ta_processors.append({"task": calc_task, "service": calculator})
            self.logger.info(f"Started calculator {i}")

        self.scheduler = CalcSchedulerService(
            assets=self.assets,
            calc_queue=self.calc_queue,
            last_calculation=self.last_calculation,
            config_manager=self.config_manager
        )
        
        self.logger.info("Starting calculation scheduler...")
        asyncio.create_task(self.scheduler.start())

        # Start Asset Initialization Pipeline
        self.logger.info("Starting asset initialization pipeline...")
        for asset in self.assets:
            self.asset_states[asset] = AssetStatus.PENDING.value
            asyncio.create_task(self.initialize_asset(asset))
            await asyncio.sleep(0.5) # Stagger start slightly


        self.logger.info("Market data system started successfully")

    async def get_status(self):
        """Return the current status of the market service components."""
        config_summary = self.config_manager.get_configuration_summary()
        
        status = {
            "service": "running",
            "db_connection": "connected" if self.db_pool and self.db_pool.read_pool else "disconnected",
            "assets": {},
            "configuration": {
                "total_assets": config_summary['total_assets'],
                "enabled_assets": config_summary['enabled_assets'],
                "regime_adaptation": config_summary['regime_adaptation_global'],
                "runtime_overrides": config_summary['runtime_overrides_count']
            },
            "asset_list": self.assets,
        }

        # Add status for each asset being monitored
        for asset in self.assets:
            asset_config = self.config_manager.get_effective_asset_config(asset)
            asset_status = {
                "enabled": asset_config.enabled,
                "regime_adaptation": asset_config.regime_adaptation_enabled,
                "websocket": "connected" if asset in self.websocket_listeners else "disconnected",
                "last_update": self.last_update.get(asset, 0),
                "last_calculation": self.last_calculation.get(asset, 0)
            }
            status["assets"][asset] = asset_status

        # Add worker statuses
        status["workers"] = {
            "db_writers": [
                {
                    "id": p["service"].worker_id,
                    "alive": not p["task"].done()
                } for p in self.data_processors
            ],
            "calculators": [
                {
                    "id": p["service"].calculator_id,
                    "alive": not p["task"].done()
                } for p in self.ta_processors
            ]
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

        # Stop cleanup service
        if self.cleanup_service:
            await self.cleanup_service.stop()

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

        if self.redis_handler:
            await self.redis_handler.close()

        self.logger.info("MarketService stopped")

    async def add_to_write_queue(self, symbol: str, candle: dict):
        await self.write_queue.put({'symbol': symbol, 'candle': candle})