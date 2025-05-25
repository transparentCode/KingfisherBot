import asyncio
import logging
from dataclasses import dataclass
from typing import Optional


@dataclass
class DBConfig:
    def __init__(self):
        self.logger_name = "app"

class DBWriter:
    def __init__(self, worker_id : int, db_pool, writer_queue: asyncio.Queue, indicator_calc_queue: asyncio.Queue,
                 dbconfig: Optional[DBConfig] = None):

        self.worker_id = worker_id
        self.write_queue = writer_queue
        self.db_pool = db_pool
        self.indicator_calc_queue = indicator_calc_queue
        self.dbconfig = DBConfig() if dbconfig is None else dbconfig
        self.logger = logging.getLogger(self.dbconfig.logger_name)
        self.running = False

    async def start(self):
        self.logger.info(f"DB writer worker {self.worker_id} started")
        self.running = True
        batch = []
        batch_size = 50
        last_flush_time = asyncio.get_event_loop().time()
        flush_interval = 5.0  # seconds


        while self.running:
            try:
                try:
                    data = await asyncio.wait_for(self.write_queue.get(), timeout=1.0)
                    batch.append(data)
                    self.write_queue.task_done()

                    if len(batch) >= batch_size:
                        await self._flush_batch(batch, self.worker_id)
                        batch = []
                        last_flush_time = asyncio.get_event_loop().time()

                except asyncio.TimeoutError:
                    current_time = asyncio.get_event_loop().time()
                    if batch and (current_time - last_flush_time >= flush_interval):
                        await self._flush_batch(batch, self.worker_id)
                        batch = []
                        last_flush_time = current_time
                    continue

            except Exception as e:
                self.logger.error(f"Error in DB writer worker {self.worker_id}: {str(e)}")
                await asyncio.sleep(1)

    async def _flush_batch(self, batch, worker_id):
        if not batch:
            return

        by_symbol = {}
        for item in batch:
            symbol = item.get('symbol')
            if symbol not in by_symbol:
                by_symbol[symbol] = []
            by_symbol[symbol].append(item.get('candle', {}))

        for symbol, candles in by_symbol.items():
            if candles:
                try:
                    count = await self.db_pool.write_candles(symbol, '1', candles)
                    self.logger.debug(f"Worker {worker_id} wrote {count} candles for {symbol}")
                    await self.indicator_calc_queue.put(symbol)
                except Exception as e:
                    self.logger.error(f"Worker {worker_id} failed to write candles for {symbol}: {str(e)}")

    async def stop(self):
        self.logger.info(f"Stopping DB writer {self.worker_id}")
        self.running = False