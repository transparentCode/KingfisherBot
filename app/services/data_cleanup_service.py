import asyncio
import logging
import os
import time
import glob
from typing import Optional

from app.db.db_handler import DBHandler

class DataCleanupService:
    """
    Service to periodically clean up old data from the database and charts directory.
    """
    def __init__(self, db_pool: DBHandler, retention_days: int = 30, check_interval_hours: int = 24, charts_dir: Optional[str] = None):
        self.db_pool = db_pool
        self.retention_days = retention_days
        self.check_interval_seconds = check_interval_hours * 3600
        self.charts_dir = charts_dir
        self.chart_cleanup_interval = 60  # 1 minute
        self.logger = logging.getLogger("app")
        self.running = False
        self.db_task = None
        self.chart_task = None

    async def start(self):
        """Start the cleanup service loops."""
        self.logger.info(f"Starting DataCleanupService (Retention: {self.retention_days} days)")
        self.running = True
        
        # Start DB cleanup loop
        self.db_task = asyncio.create_task(self._db_cleanup_loop())
        
        # Start Chart cleanup loop if directory is provided
        if self.charts_dir:
            self.chart_task = asyncio.create_task(self._chart_cleanup_loop())

    async def _db_cleanup_loop(self):
        """Loop for database cleanup"""
        while self.running:
            try:
                await self.db_pool.cleanup_old_data(self.retention_days)
            except Exception as e:
                self.logger.error(f"Error in DB cleanup: {e}")
            
            try:
                await asyncio.sleep(self.check_interval_seconds)
            except asyncio.CancelledError:
                break

    async def _chart_cleanup_loop(self):
        """Loop for charts directory cleanup"""
        self.logger.info(f"Starting chart cleanup loop for {self.charts_dir}")
        while self.running:
            try:
                self._cleanup_charts()
            except Exception as e:
                self.logger.error(f"Error in chart cleanup: {e}")
            
            try:
                await asyncio.sleep(self.chart_cleanup_interval)
            except asyncio.CancelledError:
                break

    def _cleanup_charts(self):
        """Delete files in charts directory older than 1 minute"""
        if not os.path.exists(self.charts_dir):
            return

        files = glob.glob(os.path.join(self.charts_dir, "*"))
        deleted_count = 0
        current_time = time.time()
        
        for f in files:
            try:
                if os.path.isfile(f):
                    # Delete if older than 60 seconds
                    if (current_time - os.path.getctime(f)) > 60:
                        os.remove(f)
                        deleted_count += 1
            except Exception as e:
                self.logger.error(f"Error deleting chart {f}: {e}")
        
        if deleted_count > 0:
            self.logger.info(f"Cleaned up {deleted_count} old charts")

    async def stop(self):
        """Stop the cleanup service."""
        self.logger.info("Stopping DataCleanupService")
        self.running = False
        
        if self.db_task:
            self.db_task.cancel()
            try:
                await self.db_task
            except asyncio.CancelledError:
                pass
                
        if self.chart_task:
            self.chart_task.cancel()
            try:
                await self.chart_task
            except asyncio.CancelledError:
                pass
