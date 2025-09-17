import asyncio
import logging
import os
from dataclasses import dataclass
import datetime
from typing import Optional, List, Dict, Any

import asyncpg
from dotenv import load_dotenv

dotenv = load_dotenv()


@dataclass
class DBConfig:
    def __init__(self):
        self.db_name = os.getenv("DB_NAME")
        self.db_user = os.getenv("DB_USER")
        self.db_password = os.getenv("DB_PASSWORD")
        self.db_host = os.getenv("DB_HOST")
        self.db_port = os.getenv("DB_PORT")
        self.logger_name = "app"


class DBHandler:
    def __init__(self, config: Optional[DBConfig] = None):
        self.config = DBConfig() if config is None else config
        self.logger = logging.getLogger(self.config.logger_name)
        self.db_name = self.config.db_name
        self.db_user = self.config.db_user
        self.db_password = self.config.db_password
        self.db_host = self.config.db_host
        self.db_port = self.config.db_port
        self.write_pool = None
        self.read_pool = None
        self._initialize_lock = asyncio.Lock()

        # State tracking
        self.connected = False

    async def initialize(self):
        async with self._initialize_lock:
            if self.write_pool is None or self.read_pool is None:
                self.logger.info("Initializing database connection pools...")
                try:
                    dsn = f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

                    self.write_pool = await asyncpg.create_pool(
                        dsn=dsn,
                        min_size=1,
                        max_size=10,
                        timeout=60,
                    )

                    self.read_pool = await asyncpg.create_pool(
                        dsn=dsn,
                        min_size=2,
                        max_size=10,
                        timeout=60,
                    )

                    self.logger.info("Database connection pool created.")
                except Exception as e:
                    self.logger.error(f"Error creating database connection pool: {str(e)}")
                    raise

    async def close(self):
        if self.write_pool:
            await self.write_pool.close()
            self.write_pool = None
            self.logger.info("Write connection pool closed.")

        if self.read_pool:
            await self.read_pool.close()
            self.read_pool = None
            self.logger.info("Read connection pool closed.")

    async def write_candles(self, symbol: str, interval: str, candles: List[Dict[str, Any]]):
        if not self.write_pool:
            await self.initialize()

        ist_tz = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
        try:
            async with self.write_pool.acquire() as conn:
                insert_count = 0
                update_count = 0

                for candle in candles:
                    timestamp = datetime.datetime.fromtimestamp(candle['timestamp'] / 1000).astimezone(ist_tz)

                    # Use ON CONFLICT DO UPDATE to handle duplicates
                    query = """
                    INSERT INTO candles (symbol, interval, timestamp, open, high, low, close, volume)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (symbol, interval, timestamp) 
                    DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume
                    """

                    status = await conn.execute(query,
                                                symbol,
                                                interval,
                                                timestamp,
                                                candle['open'],
                                                candle['high'],
                                                candle['low'],
                                                candle['close'],
                                                candle['volume']
                                                )

                    if "UPDATE" in status:
                        update_count += 1
                    else:
                        insert_count += 1

                self.logger.debug(f"Candles for {symbol}: {insert_count} inserted, {update_count} updated")
                return insert_count + update_count

        except Exception as e:
            self.logger.error(f"Error writing candles for symbol {symbol}: {str(e)}")
            raise

    async def read_candles(self, symbol: str, interval: str, start_time: Optional[str] = None,
                           end_time: Optional[str] = None, limit: int = 1000):
        if not self.read_pool:
            await self.initialize()

        try:
            async with self.read_pool.acquire() as conn:
                # Map common interval formats
                interval_mapping = {
                    '1m': '1',
                    '3m': '3',
                    '5m': '5',
                    '15m': '15',
                    '30m': '30',
                    '1h': '60',
                    '4h': '240',
                    '1d': '1440'
                }

                # Use mapped interval or original
                db_interval = interval_mapping.get(interval, interval)

                # Use the workng query pattern
                if start_time or end_time:
                    # Time-based query
                    query = """
                    SELECT
                        timestamp as bucket,
                        open,
                        high,
                        low,
                        close,
                        volume
                    FROM candles
                    WHERE symbol = $1 AND interval = $2
                    """
                    params = [symbol, db_interval]
                    param_count = 2

                    if start_time:
                        param_count += 1
                        query += f" AND timestamp >= ${param_count}"
                        params.append(start_time)

                    if end_time:
                        param_count += 1
                        query += f" AND timestamp <= ${param_count}"
                        params.append(end_time)

                    query += " ORDER BY timestamp DESC"

                    if limit:
                        param_count += 1
                        query += f" LIMIT ${param_count}"
                        params.append(limit)

                else:
                    # Simple limit-based query
                    query = """
                    SELECT
                        timestamp as bucket,
                        open,
                        high,
                        low,
                        close,
                        volume
                    FROM candles
                    WHERE symbol = $1 AND interval = $2
                    ORDER BY timestamp DESC
                    LIMIT $3
                    """
                    params = [symbol, db_interval, limit]

                self.logger.debug(f"🔍 Query: {query}")
                self.logger.debug(f"📊 Params: {params}")

                rows = await conn.fetch(query, *params)

                self.logger.debug(f"✅ Found {len(rows)} candles for {symbol} {interval} (db_interval: {db_interval})")

                return rows

        except Exception as e:
            self.logger.error(f"❌ Error reading candles for {symbol}: {str(e)}", exc_info=True)
            raise

    async def create_candles_table(self):
        if not self.write_pool:
            await self.initialize()

        try:
            async with self.write_pool.acquire() as conn:
                # Drop existing table to rebuild with correct schema
                await conn.execute("DROP TABLE IF EXISTS candles;")

                # Create table with the explicitly formatted timestamp type
                await conn.execute("""
                CREATE TABLE IF NOT EXISTS candles (
                    symbol TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                    open NUMERIC NOT NULL,
                    high NUMERIC NOT NULL,
                    low NUMERIC NOT NULL,
                    close NUMERIC NOT NULL,
                    volume NUMERIC NOT NULL,
                    PRIMARY KEY (symbol, interval, timestamp)
                );
                """)
                self.logger.info("Candles table created or already exists.")

                # Check if TimescaleDB is installed
                is_timescale = await conn.fetchval("SELECT COUNT(*) FROM pg_extension WHERE extname = 'timescaledb'")

                if is_timescale:
                    # Create hypertable with explicit parameters
                    await conn.execute("""
                    SELECT create_hypertable('candles', 'timestamp', 
                                            if_not_exists => TRUE);
                    """)

                    # Create index
                    await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_candles_symbol_interval_time
                    ON candles (symbol, interval, timestamp DESC);
                    """)

                self.logger.info("Candles table initialized successfully")

        except Exception as e:
            self.logger.error(f"Error creating candles table: {str(e)}")
            raise

    async def execute_query(self, query: str, *args, is_write: bool = False):

        if not self.write_pool or not self.read_pool:
            await self.initialize()

        pool = self.write_pool if is_write else self.read_pool

        async with pool.acquire() as conn:
            if is_write:
                result = await conn.execute(query, *args)
                return int(result.split()[-1]) if result else 0
            else:
                return await conn.fetch(query, *args)
