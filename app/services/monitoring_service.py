import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Set, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.market_service import MarketService


@dataclass
class MonitoringSystemConfig:
    logger_name: str = "app"
    rate_calculation_interval: int = 60  # seconds
    status_log_interval: int = 300  # seconds
    write_queue_threshold: int = 1000
    calc_queue_threshold: int = 500
    check_interval: int = 1  # seconds


class MonitoringSystem:
    """Monitors system health and reports metrics."""

    def __init__(self, system: Any, config: Optional[MonitoringSystemConfig] = None):
        self.system = system
        self.should_run = False
        self.connected_assets: Set[str] = set()
        self.message_counts: Dict[str, int] = {}
        self.message_rate: Dict[str, float] = {}
        self.last_counts: Dict[str, int] = {}
        
        # Timers
        self.last_rate_calculation = time.time()
        self.last_status_log = time.time()
        
        self.config = MonitoringSystemConfig() if config is None else config
        self.logger = logging.getLogger(self.config.logger_name)
        self._monitoring_task = None
        self._initialize_metrics()

    def _initialize_metrics(self):
        """Initialize metric dictionaries for all assets."""
        # Only add new assets, don't reset existing ones
        for asset in self.system.assets:
            if asset not in self.message_counts:
                self.message_counts[asset] = 0
                self.message_rate[asset] = 0.0
                self.last_counts[asset] = 0

    async def start(self):
        """Start the monitoring system."""
        self.logger.info("Starting monitoring system")
        self.should_run = True
        self._monitoring_task = asyncio.create_task(self._periodic_status_check())

    async def _periodic_status_check(self):
        """Periodically check system status and report metrics."""
        while self.should_run:
            try:
                # 1. Ensure metrics exist (lightweight check)
                if len(self.message_counts) != len(self.system.assets):
                    self._initialize_metrics()

                current_time = time.time()

                # 2. Calculate message rates
                if current_time - self.last_rate_calculation >= self.config.rate_calculation_interval:
                    for asset in self.system.assets:
                        # Handle case where asset might not be in counts yet
                        current_count = self.message_counts.get(asset, 0)
                        last_count = self.last_counts.get(asset, 0)
                        
                        count_diff = current_count - last_count
                        self.message_rate[asset] = count_diff / self.config.rate_calculation_interval
                        self.last_counts[asset] = current_count

                    self.last_rate_calculation = current_time

                # 3. Check queue sizes
                write_queue_size = self.system.write_queue.qsize()
                calc_queue_size = self.system.calc_queue.qsize()

                if write_queue_size > self.config.write_queue_threshold:
                    self.logger.warning(f"HIGH LOAD: Write queue size: {write_queue_size}")

                if calc_queue_size > self.config.calc_queue_threshold:
                    self.logger.warning(f"HIGH LOAD: Calculation queue size: {calc_queue_size}")

                # 4. Check Redis Connection
                if hasattr(self.system, 'redis_handler') and self.system.redis_handler:
                    if not self.system.redis_handler.connected:
                        self.logger.error("ALERT: Redis is disconnected!")

                # 5. Log disconnected assets
                for asset in self.system.assets:
                    listener = self.system.websocket_listeners.get(asset)
                    if listener and not listener.connected:
                        if asset in self.connected_assets:
                            self.logger.warning(f"Asset {asset} disconnected")
                            self.connected_assets.remove(asset)

                # 6. Log system status periodically (Fixed Logic)
                if current_time - self.last_status_log >= self.config.status_log_interval:
                    status = await self.system.get_status()
                    
                    # Add Redis stats to status log
                    if hasattr(self.system, 'redis_handler'):
                        redis_stats = await self.system.redis_handler.get_cache_stats()
                        status['redis'] = redis_stats

                    self.logger.info(f"System Status Report: {status}")
                    self.last_status_log = current_time

            except Exception as e:
                self.logger.error(f"Error in monitoring system: {e}", exc_info=True)

            await asyncio.sleep(self.config.check_interval)

    def report_connection_status(self, asset: str, connected: bool):
        """Report connection status for an asset."""
        if connected:
            if asset not in self.connected_assets:
                self.connected_assets.add(asset)
                self.logger.info(f"Asset {asset} connected")
        else:
            if asset in self.connected_assets:
                self.connected_assets.remove(asset)
                self.logger.warning(f"Asset {asset} disconnected")

        # Initialize metrics for new assets immediately
        if asset not in self.message_counts:
            self.message_counts[asset] = 0
            self.message_rate[asset] = 0.0
            self.last_counts[asset] = 0

    def report_message_received(self, asset: str):
        """Report a message received for an asset."""
        if asset not in self.message_counts:
            self.message_counts[asset] = 0
            self.message_rate[asset] = 0.0
            self.last_counts[asset] = 0

        self.message_counts[asset] += 1

    async def stop(self):
        """Stop the monitoring system."""
        self.logger.info("Stopping monitoring system")
        self.should_run = False

        if self._monitoring_task:
            try:
                # Cancel immediately if it's just sleeping
                self._monitoring_task.cancel()
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                self.logger.error(f"Error stopping monitoring task: {e}")
