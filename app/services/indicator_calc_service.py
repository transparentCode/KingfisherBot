from app.orchestration.indicator.ta_orchestrator import TechnicalAnalysisOrchestrator
from app.orchestration.indicator.trendline_orchestrator import TrendlineOrchestrator
from app.orchestration.strategy.strategy_orchestrator import StrategyOrchestrator
from app.indicators.regime_metrices import RegimeMetrics
from app.indicators.hilbert_cycle import HilbertCycle
from app.db.redis_handler import RedisHandler
import asyncio
import logging
import os
import time
from typing import Dict, List, Optional, Set

from dotenv import load_dotenv

from app.db.mtf_data_manager import MTFDataManager
from app.models.indicator_context import IndicatorExecutionContext
from app.orchestration.indicator_pipeline import IndicatorPipeline
from app.orchestration.processor.telegram_notification_processor import TelegramNotificationProcessor
from app.telegram.telegram_client import TelegramClient
from app.orchestration.processor.bot_brain_processor import BotBrainProcessor
from app.orchestration.processor.execution_processor import ExecutionProcessor
from app.execution.execution_router import ExecutionRouter
from app.services.safety_service import SafetyService
from app.risk.risk_manager import RiskManager
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
        redis_handler: Optional[RedisHandler] = None,
        execution_router: Optional[ExecutionRouter] = None,
        safety_service: Optional[SafetyService] = None,
        risk_manager: Optional[RiskManager] = None,
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
        self.redis_handler = redis_handler
        
        self.execution_router = execution_router
        self.safety_service = safety_service
        self.risk_manager = risk_manager

        # Initialize components
        self.mtf_data_manager = MTFDataManager(
            self.db_pool,
            lookback_hours=self.config.lookback_hours,
        )
        self.indicator_pipeline = IndicatorPipeline()

        self._setup_indicator_pipeline()
        self._setup_result_processors()
        self._ensure_charts_directory()

    def _setup_indicator_pipeline(self) -> None:
        """Setup indicator pipeline with orchestrators"""
        self.logger.info("Setting up indicator orchestration pipeline")

        orchestrators = [
            TechnicalAnalysisOrchestrator(
                self.indicator_registry, config_manager=self.config_manager
            ),
            TrendlineOrchestrator(
                self.indicator_registry, config_manager=self.config_manager
            ),
            StrategyOrchestrator(
                self.indicator_registry, config_manager=self.config_manager
            )
        ]

        for orchestrator in orchestrators:
            self.indicator_pipeline.add_orchestrator(orchestrator)

        self.logger.info(
            f"Added {len(self.indicator_pipeline.orchestrators)} orchestrators to pipeline"
        )

    def _setup_result_processors(self) -> None:
        """Setup result processors for handling calculation results"""
        self.logger.info("Setting up result processors")

        try:
            # Add BotBrain Processor (The Decision Maker)
            try:
                bot_brain_processor = BotBrainProcessor(
                    redis_handler=self.redis_handler,
                    db_pool=self.db_pool
                )
                self.indicator_pipeline.add_result_processor(bot_brain_processor)
                self.logger.info("BotBrainProcessor added to pipeline")
            except Exception as e:
                self.logger.error(f"Failed to setup BotBrainProcessor: {e}", exc_info=True)

            # Add Telegram Notification Processor
            try:
                # Get global config for telegram
                telegram_config = self.config_manager.global_config.get('telegram_notifications', {})
                telegram_client = TelegramClient()
                
                if telegram_client:
                    telegram_processor = TelegramNotificationProcessor(
                        config=telegram_config,
                        telegram_client=telegram_client,
                        telegram_enabled=telegram_config.get('enabled', True)
                    )
                    # Use async initialize if needed (though constructor handles sync setup)
                    # If redis init is needed, it will happen in process_results or via pipeline initialize hook if added
                    # For now, let's just add it.
                    self.indicator_pipeline.add_result_processor(telegram_processor)
                    self.logger.info("TelegramNotificationProcessor added to pipeline")
            except Exception as e:
                self.logger.error(f"Failed to setup TelegramNotificationProcessor: {e}", exc_info=True)
            
            # Add Execution Processor if services are available
            if self.execution_router and self.safety_service and self.risk_manager:
                execution_processor = ExecutionProcessor(
                    execution_router=self.execution_router,
                    safety_service=self.safety_service,
                    risk_manager=self.risk_manager
                )
                self.indicator_pipeline.add_result_processor(execution_processor)
                self.logger.info("ExecutionProcessor added to pipeline")
            else:
                self.logger.warning("Execution services not provided. ExecutionProcessor NOT added.")

        except Exception as e:
            self.logger.error(f"Error setting up result processors: {e}")
            self.logger.info("Successfully added SignalAggregationProcessor to pipeline")
        except Exception as e:
            self.logger.error(f"Failed to setup SignalAggregationProcessor: {e}", exc_info=True)

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
                task = await asyncio.wait_for(
                    self.calc_queue.get(), timeout=self.config.queue_timeout
                )

                if isinstance(task, dict):
                    task_type = task.get("type")
                    asset = task.get("asset")
                    
                    self.logger.info(f"Calculator {self.calculator_id} received task: {task_type} for {asset}")

                    if task_type == "INDICATOR_UPDATE":
                        await self._calculate_indicators(asset)
                    elif task_type == "REGIME_UPDATE":
                        await self._handle_regime_update(asset)
                    elif task_type == "BAR_CLOSE_CALC":
                        await self._calculate_on_close_metrics(asset, task.get("timeframe"))
                    else:
                        self.logger.warning(f"Unknown task type: {task_type}")
                elif isinstance(task, str):
                    await self._calculate_indicators(task)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error in calculator {self.calculator_id}: {e}", exc_info=True)
            finally:
                # Only mark task as done if we actually got one
                # The 'task' variable might be undefined if TimeoutError occurred before assignment
                # or if wait_for raised an exception
                try:
                    if 'task' in locals() and task is not None:
                        self.calc_queue.task_done()
                except ValueError:
                    # task_done() called too many times
                    pass

        self.logger.info(f"Indicator calculator {self.calculator_id} stopped")

    async def _handle_regime_update(self, asset: str) -> None:
        """Handle market regime update task"""
        try:
            self.logger.info(f"Processing regime update for {asset}")
            # TODO: Implement actual regime detection logic here
            # For now, we can just log it or trigger a specific pipeline
            pass
        except Exception as e:
            self.logger.error(f"Error updating regime for {asset}: {e}")

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

            # Fetch latest regime metrics
            # We try to fetch for the primary timeframe and a higher timeframe (e.g. 1h or 4h)
            regime_tf = "1h" # Default strategic timeframe
            if context.primary_timeframe == "1h": regime_tf = "4h"
            
            regime_data = await self.db_pool.get_latest_regime_metrics(asset, regime_tf)
            if not regime_data:
                 # Fallback to primary timeframe if strategic not available
                 regime_data = await self.db_pool.get_latest_regime_metrics(asset, context.primary_timeframe)
            
            if regime_data:
                context.metadata['regime'] = regime_data

            results = await self.indicator_pipeline.execute_pipeline(context)

            calculation_time = time.time() - start_time
            self.logger.info(
                f"Calculated {len(results)} indicator results for {asset} in {calculation_time:.2f}s"
            )

            self.last_calculation[asset] = time.time()

        except Exception as e:
            self.logger.error(f"Error calculating indicators for {asset}: {e}")
            self.logger.debug("Exception details:", exc_info=True)

    async def _calculate_on_close_metrics(self, asset: str, timeframe: str) -> None:
        """
        Calculate specific metrics that need to run exactly on bar close.
        This is triggered by BarCloseService via BAR_CLOSE_CALC task.
        """
        try:
            self.logger.info(f"Running On-Close Metrics for {asset} {timeframe}")
            
            # 1. Fetch Data for this specific timeframe
            # We might need a bit more history for things like Hurst
            mtf_data = await self.mtf_data_manager.get_mtf_data(asset, [timeframe])
            df = mtf_data.get(timeframe)
            
            if df is None or df.empty:
                self.logger.warning(f"No data available for {asset} {timeframe} on close")
                return

            # 2. Run Calculations - Regime
            regime_metrics = RegimeMetrics()
            regime_features = regime_metrics.get_features(df)
            
            # 2.1 Run Calculations - Hilbert Cycle (Adaptive Period)
            # Efficiently computes the dominant market cycle for adaptive strategies
            hilbert_cycle = HilbertCycle()
            hilbert_features = hilbert_cycle.get_features(df)
            
            # Access internal state for metadata
            metrics_df = regime_metrics.metrics_df
            
            if metrics_df is None or metrics_df.empty:
                self.logger.warning(f"Regime metrics calculation failed for {asset} {timeframe}")
                return

            latest_meta = metrics_df.iloc[-1]
            
            # Prepare data for DB and Redis
            metrics_data = {
                'timestamp': latest_meta.name, 
                # Regime Metrics
                'hurst': regime_features.get('regime_hurst', 0.5),
                'volatility': regime_features.get('regime_vol_stress', 0.0),
                'regime': str(latest_meta.get('regime', 'UNCERTAIN')),
                'trend_strength': regime_features.get('regime_trend_score', 0.0),
                'skew': regime_features.get('regime_skew', 0.0),
                'kurtosis': regime_features.get('regime_tail_risk', 0.0),
                
                # Hilbert Metrics (Included for Redis/Runtime usage)
                # Note: These won't be saved to SQL unless the table schema is updated,
                # but they will be available in Redis for strategies.
                'cycle_period': hilbert_features.get('hilbert_period', 20.0),
                'cycle_phase': hilbert_features.get('hilbert_phase', 0.0),
                'cycle_state': hilbert_features.get('hilbert_state', 0.0) # -1 (valley) to 1 (peak)
            }

            # 3. Save Results
            await self.db_pool.write_regime_metrics(asset, timeframe, metrics_data)
            
            # Publish to Redis for real-time access if needed
            if self.redis_handler:
                await self.redis_handler.set_key(f"regime:{asset}:{timeframe}", metrics_data)
            
            self.logger.info(f"Completed On-Close Metrics for {asset} {timeframe}")

        except Exception as e:
            self.logger.error(f"Error in _calculate_on_close_metrics for {asset} {timeframe}: {e}", exc_info=True)

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
