import asyncio
import logging
import time
from typing import List, Dict


class CalcSchedulerService:
    """
    Scheduler for asset calculation tasks.
    """

    def __init__(self, assets: List[str], calc_queue: asyncio.Queue, last_calculation: Dict[str, float]):
        """
        Initialize the calculation scheduler.

        Args:
            assets: List of assets to schedule calculations for
            calc_queue: Queue to add calculation tasks to
            last_calculation: Dict tracking last calculation time for each asset
        """
        self.assets = assets
        self.calc_queue = calc_queue
        self.last_calculation = last_calculation
        self.should_run = False
        self.min_interval = 60  # Minimum 60 seconds between calculations
        self.logger = logging.getLogger("app")

    async def start(self):
        """Start the calculation scheduler."""
        self.logger.info("Starting calculation scheduler")
        self.should_run = True

        while self.should_run:
            current_time = time.time()

            for asset in self.assets:
                # If asset hasn't been calculated recently, schedule it
                last_time = self.last_calculation.get(asset, 0)
                if current_time - last_time >= self.min_interval:
                    await self.calc_queue.put(asset)
                    self.last_calculation[asset] = current_time
                    self.logger.debug(f"Scheduled calculation for {asset}")

            # Check every 10 seconds
            await asyncio.sleep(10)

    async def stop(self):
        self.logger.info("Stopping calculation scheduler")
        self.should_run = False