import asyncio
import logging
import time
from datetime import datetime
from typing import List, Dict, Optional

from config.asset_indicator_config import ConfigurationManager


class CalcSchedulerService:
    """
    Enhanced scheduler for asset calculation tasks and regime updates.
    """

    def __init__(self, assets: List[str], calc_queue: asyncio.Queue, last_calculation: Dict[str, float],
                 config_manager: Optional[ConfigurationManager] = None):
        """
        Initialize the calculation scheduler.

        Args:
            assets: List of assets to schedule calculations for
            calc_queue: Queue to add calculation tasks to
            last_calculation: Dict tracking last calculation time for each asset
            config_manager: Configuration manager for asset settings
        """
        self.assets = assets
        self.calc_queue = calc_queue
        self.last_calculation = last_calculation
        self.config_manager = config_manager
        self.should_run = False
        self.logger = logging.getLogger("app")

        # Indicator calculation settings
        self.min_calculation_interval = 60  # 60 seconds between indicator calculations
        self.max_queue_size = 200  # Max queue size before skipping scheduling

        # Regime update tracking - per asset
        self.last_regime_update: Dict[str, float] = {asset: 0 for asset in assets}
        self.regime_update_intervals: Dict[str, int] = {}

        # Load configuration
        self._load_regime_configuration()

    def _load_regime_configuration(self):
        """Load regime update configuration from config manager"""
        try:
            global_regime_config = self.config_manager.get_regime_config()

            # Default regime update interval from global config
            default_interval_str = global_regime_config.get('update_interval', '30m')
            default_interval_seconds = self._parse_interval(default_interval_str)

            # Set per-asset regime update intervals
            for asset in self.assets:
                asset_config = self.config_manager.get_base_asset_config(asset)

                # Check if asset has custom regime update interval
                if hasattr(asset_config, 'regime_update_interval'):
                    interval_str = asset_config.regime_update_interval
                    self.regime_update_intervals[asset] = self._parse_interval(interval_str)
                else:
                    # Use global default
                    self.regime_update_intervals[asset] = default_interval_seconds

                self.logger.debug(f"Regime update interval for {asset}: {self.regime_update_intervals[asset]}s")

        except Exception as e:
            self.logger.error(f"Failed to load regime configuration: {e}")
            # Fallback: 15 minutes for all assets
            for asset in self.assets:
                self.regime_update_intervals[asset] = 900  # 15 minutes

    def _parse_interval(self, interval_str: str) -> int:
        """
        Parse interval string to seconds.
        Supports: 30s, 5m, 1h, 2d
        """
        try:
            if interval_str.endswith('s'):
                return int(interval_str[:-1])
            elif interval_str.endswith('m'):
                return int(interval_str[:-1]) * 60
            elif interval_str.endswith('h'):
                return int(interval_str[:-1]) * 3600
            elif interval_str.endswith('d'):
                return int(interval_str[:-1]) * 86400
            else:
                # Assume seconds if no unit
                return int(interval_str)
        except (ValueError, TypeError):
            self.logger.warning(f"Invalid interval format: {interval_str}, using 15m default")
            return 900  # 15 minutes default

    async def start(self):
        """Start the enhanced calculation scheduler."""
        self.logger.info("Starting enhanced calculation scheduler")
        self.should_run = True

        while self.should_run:
            current_time = time.time()

            for asset in self.assets:
                if not self.config_manager.is_asset_enabled(asset):
                    continue

                # Schedule indicator calculations
                await self._schedule_indicator_calculation(asset, current_time)

                # Schedule regime updates (if enabled for this asset)
                await self._schedule_regime_update(asset, current_time)

            # Check every 10 seconds
            await asyncio.sleep(10)

    async def _schedule_indicator_calculation(self, asset: str, current_time: float):
        """Schedule indicator calculation for an asset if needed"""
        last_calc_time = self.last_calculation.get(asset, 0)

        if current_time - last_calc_time >= self.min_calculation_interval:
            # Prevent queue flooding if consumer is backed up
            if self.calc_queue.qsize() > self.max_queue_size:
                self.logger.warning(f"Calc queue full ({self.calc_queue.qsize()}), skipping schedule for {asset}")
                return

            task = {
                "type": "INDICATOR_UPDATE",
                "asset": asset,
                "timestamp": current_time
            }
            await self.calc_queue.put(task)

            self.last_calculation[asset] = current_time
            self.logger.debug(f"Scheduled indicator calculation for {asset}")

    async def _schedule_regime_update(self, asset: str, current_time: float):
        """Schedule regime update for an asset if needed"""
        # Check if regime adaptation is enabled globally
        if not self.config_manager.is_regime_adaptation_enabled():
            return

        # Check if regime adaptation is enabled for this specific asset
        if not self.config_manager.is_asset_regime_enabled(asset):
            return

        # Check timing
        last_regime_time = self.last_regime_update.get(asset, 0)
        regime_interval = self.regime_update_intervals.get(asset, 900)  # Default 15m

        if current_time - last_regime_time >= regime_interval:
            task = {
                "type": "REGIME_UPDATE",
                "asset": asset,
                "timestamp": current_time
            }
            await self.calc_queue.put(task)

            self.last_regime_update[asset] = current_time
            self.logger.info(f"Scheduled regime update for {asset}")

    def get_scheduler_status(self) -> Dict[str, any]:
        """Get overall scheduler status"""
        current_time = time.time()

        # Count assets by status
        total_assets = len(self.assets)
        enabled_assets = len([a for a in self.assets if self.config_manager.is_asset_enabled(a)])
        regime_enabled_assets = len([a for a in self.assets if self.config_manager.is_asset_regime_enabled(a)])

        # Next scheduled times
        next_indicator_calc = None
        next_regime_update = None

        for asset in self.assets:
            if self.config_manager.is_asset_enabled(asset):
                # Next indicator calculation
                last_calc = self.last_calculation.get(asset, 0)
                next_calc = last_calc + self.min_calculation_interval
                if next_indicator_calc is None or next_calc < next_indicator_calc:
                    next_indicator_calc = next_calc

                # Next regime update
                if self.config_manager.is_asset_regime_enabled(asset):
                    last_regime = self.last_regime_update.get(asset, 0)
                    regime_interval = self.regime_update_intervals.get(asset, 900)
                    next_regime = last_regime + regime_interval
                    if next_regime_update is None or next_regime < next_regime_update:
                        next_regime_update = next_regime

        return {
            'running': self.should_run,
            'total_assets': total_assets,
            'enabled_assets': enabled_assets,
            'regime_enabled_assets': regime_enabled_assets,
            'regime_adaptation_global': self.config_manager.is_regime_adaptation_enabled(),
            'next_indicator_calculation': {
                'timestamp': next_indicator_calc,
                'time_str': datetime.fromtimestamp(next_indicator_calc).isoformat() if next_indicator_calc else 'None',
                'seconds_until': max(0, next_indicator_calc - current_time) if next_indicator_calc else None
            },
            'next_regime_update': {
                'timestamp': next_regime_update,
                'time_str': datetime.fromtimestamp(next_regime_update).isoformat() if next_regime_update else 'None',
                'seconds_until': max(0, next_regime_update - current_time) if next_regime_update else None
            },
            'regime_intervals': self.regime_update_intervals
        }

    async def stop(self):
        """Stop the calculation scheduler."""
        self.logger.info("Stopping enhanced calculation scheduler")
        self.should_run = False
