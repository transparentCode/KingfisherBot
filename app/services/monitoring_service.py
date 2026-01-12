import asyncio
import logging
import time
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Set, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.market_service import MarketService


@dataclass
class MonitoringSystemConfig:
    logger_name: str = "app"
    rate_calculation_interval: int = 60  # seconds
    status_log_interval: int = 300  # seconds
    metric_storage_interval: int = 5 # Store metrics frequently for charts
    write_queue_threshold: int = 1000
    calc_queue_threshold: int = 500
    check_interval: int = 1  # seconds


class MonitoringSystem:
    """Monitors system health and reports metrics."""

    def __init__(self, system: Any, config: Optional[MonitoringSystemConfig] = None):
        self.system = system
        self.should_run = False
        self.connected_assets: Set[str] = set()
        self.interval_counts: Dict[str, int] = {}  # Counts since last calculation
        self.message_rate: Dict[str, float] = {}
        
        # Timers
        self.last_rate_calculation = time.time()
        self.last_status_log = time.time()
        self.last_metric_storage = time.time()
        
        self.config = MonitoringSystemConfig() if config is None else config
        self.logger = logging.getLogger(self.config.logger_name)
        self._monitoring_task = None
        self._initialize_metrics()

    def _initialize_metrics(self):
        """Initialize metric dictionaries for all assets."""
        # Only add new assets, don't reset existing ones
        for asset in self.system.assets:
            if asset not in self.interval_counts:
                self.interval_counts[asset] = 0
                self.message_rate[asset] = 0.0

    async def start(self):
        """Start the monitoring system."""
        self.logger.info("Starting monitoring system")
        self.should_run = True
        self._monitoring_task = asyncio.create_task(self._periodic_status_check())

    async def _store_metrics(self):
        """Store system metrics in Redis for historical graphing."""
        if not hasattr(self.system, 'redis_handler') or not self.system.redis_handler:
            return

        try:
            timestamp = int(time.time() * 1000)
            
            # 1. CPU Load (1 min avg)
            # Note: os.getloadavg() returns (1min, 5min, 15min)
            load_avg = os.getloadavg()[0]
            await self.system.redis_handler.add_metric(
                "metrics:system:cpu_load", 
                {"ts": timestamp, "val": load_avg}
            )

            # 2. Queue Depths
            write_q = self.system.write_queue.qsize()
            calc_q = self.system.calc_queue.qsize()
            
            await self.system.redis_handler.add_metric(
                "metrics:system:queue_write", 
                {"ts": timestamp, "val": write_q}
            )
            await self.system.redis_handler.add_metric(
                "metrics:system:queue_calc", 
                {"ts": timestamp, "val": calc_q}
            )

            # 3. Message Rate (Total)
            total_rate = sum(self.message_rate.values())
            await self.system.redis_handler.add_metric(
                "metrics:system:msg_rate", 
                {"ts": timestamp, "val": total_rate}
            )

        except Exception as e:
            self.logger.error(f"Error storing metrics: {e}")

    async def _periodic_status_check(self):
        """Periodically check system status and report metrics."""
        while self.should_run:
            try:
                # 1. Ensure metrics exist (lightweight check)
                if len(self.interval_counts) != len(self.system.assets):
                    self._initialize_metrics()

                current_time = time.time()

                # 2. Calculate message rates
                if current_time - self.last_rate_calculation >= self.config.rate_calculation_interval:
                    for asset in self.system.assets:
                        # Handle case where asset might not be in counts yet
                        count = self.interval_counts.get(asset, 0)
                        
                        # Rate = count / interval
                        self.message_rate[asset] = count / self.config.rate_calculation_interval
                        
                        # Reset counter for next interval
                        self.interval_counts[asset] = 0

                    self.last_rate_calculation = current_time

                # 2.5 Store metrics frequency check
                if current_time - self.last_metric_storage >= self.config.metric_storage_interval:
                    await self._store_metrics()
                    self.last_metric_storage = current_time

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
        if asset not in self.interval_counts:
            self.interval_counts[asset] = 0
            self.message_rate[asset] = 0.0

    def report_message_received(self, asset: str):
        """Report a message received for an asset."""
        if asset not in self.interval_counts:
            self.interval_counts[asset] = 0
            self.message_rate[asset] = 0.0

        self.interval_counts[asset] += 1

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
