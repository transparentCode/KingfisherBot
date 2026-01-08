import asyncio
import logging
import pandas as pd
from typing import List
from app.db.mtf_data_manager import MTFDataManager

class BarCloseService:
    """
    Service to handle 'Bar Close' events.
    It listens to 1m candle close events and determines if higher timeframe candles have closed.
    It then triggers specific processing for those timeframes.
    """
    def __init__(self, queue: asyncio.Queue, calc_queue: asyncio.Queue, mtf_data_manager: MTFDataManager, timeframes: List[str]):
        self.queue = queue
        self.calc_queue = calc_queue
        self.mtf_data_manager = mtf_data_manager
        self.timeframes = timeframes
        self.logger = logging.getLogger("app")
        self.running = False
        
        # Map timeframes to minutes for calculation
        self.tf_minutes = {
            '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '6h': 360,
            '12h': 720, '1d': 1440
        }

    async def start(self):
        self.running = True
        self.logger.info(f"BarCloseService started. Monitoring timeframes: {self.timeframes}")
        while self.running:
            try:
                event = await self.queue.get()
                await self.process_event(event)
                self.queue.task_done()
            except Exception as e:
                self.logger.error(f"Error in BarCloseService: {e}", exc_info=True)

    async def process_event(self, event):
        """
        Process a 1m candle close event.
        """
        candle = event['candle']
        timestamp_ms = candle['timestamp']
        asset = event['asset']
        
        # Calculate which timeframes just closed based on the 1m candle timestamp
        # Logic: If (1m_Open_Time + 1min) is divisible by TF_Minutes, then TF just closed.
        open_time_min = timestamp_ms // 60000
        close_time_min = open_time_min + 1
        
        closed_timeframes = []
        for tf in self.timeframes:
            minutes = self.tf_minutes.get(tf)
            if minutes and close_time_min % minutes == 0:
                closed_timeframes.append(tf)
        
        if closed_timeframes:
            self.logger.info(f"Bar Close detected for {asset}: {closed_timeframes}")
            for tf in closed_timeframes:
                await self.handle_timeframe_close(asset, tf, candle)

    async def handle_timeframe_close(self, asset: str, timeframe: str, latest_1m_candle: dict):
        """
        Handle the logic when a specific timeframe closes.
        """
        try:
            self.logger.info(f"Processing {timeframe} close for {asset}")
            
            # Push task to calculation queue for heavy processing
            task = {
                "type": "BAR_CLOSE_CALC",
                "asset": asset,
                "timeframe": timeframe,
                "timestamp": latest_1m_candle['timestamp']
            }
            await self.calc_queue.put(task)
            
        except Exception as e:
            self.logger.error(f"Error handling {timeframe} close for {asset}: {e}")

    async def stop(self):
        self.running = False
        self.logger.info("BarCloseService stopped")
