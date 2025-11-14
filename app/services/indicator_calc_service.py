import asyncio
import logging
import os
import time
from typing import Dict, List, Optional, Set

from dotenv import load_dotenv

from app.db.mtf_data_manager import MTFDataManager
from app.models.indicator_context import IndicatorExecutionContext
from app.orchestration.indicator.rsi_orchestrator import OscillatorOrchestrator
from app.orchestration.indicator.sma_orchestrator import MovingAverageOrchestrator
from app.orchestration.indicator.trendline_orchestrator import TrendAnalysisOrchestrator
from app.orchestration.indicator_pipeline import IndicatorPipeline
from app.orchestration.processor.mtf_confluence_processor import (
    SignalAggregationProcessor,
)
from app.orchestration.processor.telegram_notification_processor import (
    TelegramNotificationProcessor,
)
from app.telegram import TelegramClient, telegram_client
from config.asset_indicator_config import ConfigurationManager

load_dotenv()


class IndicatorCalcServiceConfig:
    """Configuration class for IndicatorCalcService"""

    def __init__(self):
        self.logger_name = "app"
        self.timeframes_to_monitor = ["15m", "30m"]
        self.telegram_enabled = (
            os.getenv("TELEGRAM_NOTIFICATIONS_ENABLED", "true").lower() == "true"
        )
        self.save_charts = os.getenv("SAVE_CHARTS", "true").lower() == "true"
        self.charts_dir = os.path.join(os.path.dirname(__file__), "../charts")
        self.lookback_hours = 96
        self.minimum_data_points = 20
        self.queue_timeout = 1.0


class IndicatorCalcService:
    """Service for calculating indicators using orchestration pipeline"""

    def __init__(
        self,
        calculator_id: int,
        calc_queue: asyncio.Queue,
        db_pool,
        indicator_registry,
        last_calculation: Dict[str, float],
        config: Optional[IndicatorCalcServiceConfig] = None,
        config_manager: Optional[ConfigurationManager] = None,
    ):
        self.calculator_id = calculator_id
        self.calc_queue = calc_queue
        self.db_pool = db_pool
        self.indicator_registry = indicator_registry
        self.last_calculation = last_calculation
        self.should_run = False
        self.config = config or IndicatorCalcServiceConfig()
        self.logger = logging.getLogger(self.config.logger_name)
        self.config_manager = config_manager

        # Initialize components
        self.mtf_data_manager = MTFDataManager(
            self.db_pool, lookback_hours=self.config.lookback_hours
        )
        self.indicator_pipeline = IndicatorPipeline()

        self._setup_indicator_pipeline()
        self._setup_result_processors()
        self._ensure_charts_directory()
 
    def _setup_indicator_pipeline(self) -> None:
        """Setup indicator pipeline with orchestrators"""
        self.logger.info("Setting up indicator orchestration pipeline")

        orchestrators = [
            TrendAnalysisOrchestrator(
                self.indicator_registry, config_manager=self.config_manager
            ),
            MovingAverageOrchestrator(
                self.indicator_registry, config_manager=self.config_manager
            ),
            OscillatorOrchestrator(
                self.indicator_registry, config_manager=self.config_manager
            ),
        ]

        for orchestrator in orchestrators:
            self.indicator_pipeline.add_orchestrator(orchestrator)

        self.logger.info(
            f"Added {len(self.indicator_pipeline.orchestrators)} orchestrators to pipeline"
        )

    def _setup_result_processors(self) -> None:
        """Setup result processors for handling calculation results"""
        self.logger.info("Setting up result processors")

        aggregation_processor = SignalAggregationProcessor(
            config_manager=self.config_manager
        )

        self.indicator_pipeline.add_result_processor(aggregation_processor)

    def _ensure_charts_directory(self) -> None:
        """Ensure charts directory exists if chart saving is enabled"""
        if self.config.save_charts and not os.path.exists(self.config.charts_dir):
            try:
                os.makedirs(self.config.charts_dir)
                self.logger.info(f"Created charts directory: {self.config.charts_dir}")
            except OSError as e:
                self.logger.error(f"Failed to create charts directory: {e}")

    async def start(self) -> None:
        """Start the indicator calculation service"""
        self.should_run = True
        self.logger.info(f"Starting indicator calculator {self.calculator_id}")

        while self.should_run:
            asset = None
            try:
                asset = await asyncio.wait_for(
                    self.calc_queue.get(), timeout=self.config.queue_timeout
                )
                await self._calculate_indicators(asset)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error in calculator {self.calculator_id}: {e}")
            finally:
                if asset is not None:
                    self.calc_queue.task_done()

        self.logger.info(f"Indicator calculator {self.calculator_id} stopped")

    async def _calculate_indicators(self, asset: str) -> None:
        """Calculate indicators for given asset using orchestration pipeline"""
        try:
            start_time = time.time()
            self.logger.debug(f"Starting indicator calculation for {asset}")

            required_timeframes = self._get_required_timeframes(asset=asset)
            mtf_data = await self.mtf_data_manager.get_mtf_data(
                asset, required_timeframes
            )

            valid_timeframes = self._validate_data(mtf_data)
            if not valid_timeframes:
                self.logger.warning(f"No valid data available for {asset}")
                return

            context = IndicatorExecutionContext(
                asset=asset,
                primary_timeframe=self.config.timeframes_to_monitor[0],
                data_cache=valid_timeframes,
            )

            results = await self.indicator_pipeline.execute_pipeline(context)

            calculation_time = time.time() - start_time
            self.logger.info(
                f"Calculated {len(results)} indicator results for {asset} in {calculation_time:.2f}s"
            )

            self.last_calculation[asset] = time.time()

        except Exception as e:
            self.logger.error(f"Error calculating indicators for {asset}: {e}")
            self.logger.debug("Exception details:", exc_info=True)

    def _validate_data(self, mtf_data: Dict) -> Dict:
        """Validate MTF data meets minimum requirements"""
        return {
            tf: data
            for tf, data in mtf_data.items()
            if not data.empty and len(data) > self.config.minimum_data_points
        }

    def _get_required_timeframes(self, asset: Optional[str] = None) -> List[str]:
        """Get required timeframes from configuration"""
        timeframes: Set[str] = set(self.config.timeframes_to_monitor)

        if asset and self.config_manager:
            try:
                asset_config = self.config_manager.get_effective_asset_config(asset)

                # Add timeframes from timeframe_overrides
                if asset_config.timeframe_overrides:
                    configured_timeframes = asset_config.timeframe_overrides.keys()
                    timeframes.update(configured_timeframes)
                    self.logger.info(
                        f"Added configured timeframes for {asset}: {list(configured_timeframes)}"
                    )

            except Exception as e:
                self.logger.error(
                    f"Failed to get configured timeframes for {asset}: {e}"
                )

        timeframes.add("1m")

        valid_timeframes = [
            tf
            for tf in timeframes
            if tf in ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]
        ]
        return valid_timeframes

    async def stop(self) -> None:
        """Stop the indicator calculation service"""
        self.logger.info(f"Stopping indicator calculator {self.calculator_id}")
        self.should_run = False

    async def cleanup(self) -> None:
        """Cleanup resources"""
        if self.telegram_client:
            # Add telegram client cleanup if needed
            pass
