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
        """
        Initialize the monitoring system.

        Args:
            system: Reference to the main system for status queries
            config: Optional configuration for the monitoring system
        """
        self.system = system
        self.should_run = False
        self.connected_assets: Set[str] = set()
        self.message_counts: Dict[str, int] = {}
        self.message_rate: Dict[str, float] = {}
        self.last_counts: Dict[str, int] = {}
        self.last_rate_calculation = time.time()
        self.config = MonitoringSystemConfig() if config is None else config
        self.logger = logging.getLogger(self.config.logger_name)
        self._monitoring_task = None
        self._initialize_metrics()

    def _initialize_metrics(self):
        """Initialize metric dictionaries for all assets."""
        for asset in self.system.assets:
            if asset not in self.message_counts:
                self.message_counts[asset] = 0
                self.message_rate[asset] = 0.0
                self.last_counts[asset] = 0

    async def start(self):
        """Start the monitoring system."""
        self.logger.info("Starting monitoring system")
        self.should_run = True

        # Start periodic status check
        self._monitoring_task = asyncio.create_task(self._periodic_status_check())

    async def _periodic_status_check(self):
        """Periodically check system status and report metrics."""
        while self.should_run:
            try:
                # Ensure we have metrics for all assets (in case new ones were added)
                self._initialize_metrics()

                # Calculate message rates
                current_time = time.time()
                time_diff = current_time - self.last_rate_calculation

                if time_diff >= self.config.rate_calculation_interval:
                    for asset in self.system.assets:
                        count_diff = self.message_counts[asset] - self.last_counts[asset]
                        self.message_rate[asset] = count_diff / time_diff
                        self.last_counts[asset] = self.message_counts[asset]

                    self.last_rate_calculation = current_time

                # Check queue sizes
                write_queue_size = self.system.write_queue.qsize()
                calc_queue_size = self.system.calc_queue.qsize()

                if write_queue_size > self.config.write_queue_threshold:
                    self.logger.warning(f"Write queue size is high: {write_queue_size}")

                if calc_queue_size > self.config.calc_queue_threshold:
                    self.logger.warning(f"Calculation queue size is high: {calc_queue_size}")

                # Log disconnected assets
                for asset in self.system.assets:
                    listener = self.system.websocket_listeners.get(asset)
                    if listener and not listener.connected:
                        if asset in self.connected_assets:
                            self.logger.warning(f"Asset {asset} disconnected")
                            self.connected_assets.remove(asset)

                # Log system status periodically
                if int(current_time) % self.config.status_log_interval < 1:
                    status = await self.system.get_status()
                    self.logger.info(f"System status: {status}")

            except Exception as e:
                self.logger.error(f"Error in monitoring system: {e}", exc_info=True)

            # Check at a configured interval
            await asyncio.sleep(self.config.check_interval)

    def report_connection_status(self, asset: str, connected: bool):
        """Report connection status for an asset."""
        if connected and asset not in self.connected_assets:
            self.connected_assets.add(asset)
            self.logger.info(f"Asset {asset} connected")
        elif not connected and asset in self.connected_assets:
            self.connected_assets.remove(asset)
            self.logger.warning(f"Asset {asset} disconnected")

        # Initialize metrics for new assets
        if asset not in self.message_counts:
            self.message_counts[asset] = 0
            self.message_rate[asset] = 0.0
            self.last_counts[asset] = 0

    def report_message_received(self, asset: str):
        """Report a message received for an asset."""
        if asset not in self.message_counts:
            # Initialize metrics for new assets
            self.message_counts[asset] = 0
            self.message_rate[asset] = 0.0
            self.last_counts[asset] = 0

        self.message_counts[asset] += 1

    async def stop(self):
        """Stop the monitoring system."""
        self.logger.info("Stopping monitoring system")
        self.should_run = False

        # Wait for monitoring task to finish
        if self._monitoring_task:
            try:
                await asyncio.wait_for(self._monitoring_task, timeout=5.0)
            except asyncio.TimeoutError:
                self.logger.warning("Monitoring task did not stop gracefully, cancelling")
                self._monitoring_task.cancel()