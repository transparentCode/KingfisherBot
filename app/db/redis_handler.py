import asyncio
import logging
import threading
from typing import Optional, Any, Dict, List
import json

import redis.asyncio as redis
from dotenv import load_dotenv

from app.models.RedisConfig import RedisConfig

load_dotenv()


class RedisHandler:
    """Redis handler for managing Redis connections and operations"""
    
    _instance: Optional['RedisHandler'] = None
    _lock = threading.Lock()
    
    def __new__(cls, config: Optional[RedisConfig] = None) -> 'RedisHandler':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config: Optional[RedisConfig] = None):
        # Prevent re-initialization
        if hasattr(self, '_initialized'):
            return
            
        self.config = RedisConfig() if config is None else config
        self.logger = logging.getLogger(self.config.logger_name)
        self.redis_client: Optional[redis.Redis] = None
        self._initialize_lock = asyncio.Lock()
        
        # State tracking
        self.connected = False
        self._initialized = True

    async def initialize(self):
        """Initialize Redis connection pool"""
        async with self._initialize_lock:
            if self.redis_client is None:
                self.logger.info("Initializing Redis connection pool...")
                try:
                    # Create Redis client with connection pool
                    self.redis_client = await redis.from_url(
                        self.config.redis_url,
                        max_connections=self.config.max_connections,
                        socket_timeout=self.config.socket_timeout,
                        encoding="utf-8",
                        decode_responses=True,
                    )

                    # Test connection
                    await self.redis_client.ping()

                    self.connected = True
                    self.logger.info(
                        "Redis connection pool created and tested successfully."
                    )

                except Exception as e:
                    self.logger.error(f"Error creating Redis connection pool: {str(e)}")
                    self.connected = False
                    raise

    async def close(self):
        """Close Redis connection pool"""
        if self.redis_client:
            await self.redis_client.aclose()
            self.redis_client = None
            self.connected = False
            self.logger.info("Redis connection pool closed.")

    async def _get_client(self) -> redis.Redis:
        """Get Redis client from pool"""
        if not self.redis_client:
            await self.initialize()
        return self.redis_client

    # ================================
    # Market Data Caching Methods
    # ================================

    async def cache_market_data(self, symbol: str, timeframe: str, data: Any, ttl: Optional[int] = None) -> bool:
        """Cache market data (OHLCV) with automatic serialization"""
        try:
            client = await self._get_client()
            key = f"market_data:{symbol}:{timeframe}"

            # Serialize data (handle pandas DataFrame, dict, list)
            if hasattr(data, "to_json"):  # pandas DataFrame
                serialized_data = data.to_json(orient="records", date_format="iso")
            else:
                serialized_data = json.dumps(data, default=str)

            ttl = ttl or self.config.market_data_ttl
            await client.setex(key, ttl, serialized_data)

            self.logger.debug(
                f"Cached market data for {symbol}:{timeframe} (TTL: {ttl}s)"
            )
            return True

        except Exception as e:
            self.logger.error(
                f"Error caching market data for {symbol}:{timeframe}: {e}"
            )
            return False

    async def get_market_data(self, symbol: str, timeframe: str) -> Optional[Any]:
        """Retrieve cached market data"""
        try:
            client = await self._get_client()
            key = f"market_data:{symbol}:{timeframe}"

            data = await client.get(key)
            if data:
                # Try to deserialize as JSON
                try:
                    return json.loads(data)
                except json.JSONDecodeError:
                    return data
            return None

        except Exception as e:
            self.logger.error(
                f"Error retrieving market data for {symbol}:{timeframe}: {e}"
            )
            return None

    # ================================
    # Indicator Results Caching
    # ================================

    async def cache_indicator_results(
        self,
        asset: str,
        indicator_type: str,
        timeframe: str,
        results: Dict,
        ttl: Optional[int] = None,
    ) -> bool:
        """Cache indicator calculation results"""
        try:
            client = await self._get_client()
            key = f"indicators:{asset}:{indicator_type}:{timeframe}"

            # Add metadata
            cache_data = {
                "results": results,
                "cached_at": str(asyncio.get_event_loop().time()),
                "asset": asset,
                "indicator_type": indicator_type,
                "timeframe": timeframe,
            }

            serialized_data = json.dumps(cache_data, default=str)
            ttl = ttl or self.config.indicator_ttl

            await client.setex(key, ttl, serialized_data)
            self.logger.debug(
                f"Cached indicator results: {asset}:{indicator_type}:{timeframe}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Error caching indicator results: {e}")
            return False

    async def get_indicator_results(
        self, asset: str, indicator_type: str, timeframe: str
    ) -> Optional[Dict]:
        """Retrieve cached indicator results"""
        try:
            client = await self._get_client()
            key = f"indicators:{asset}:{indicator_type}:{timeframe}"

            data = await client.get(key)
            if data:
                cache_data = json.loads(data)
                return cache_data.get("results")
            return None

        except Exception as e:
            self.logger.error(f"Error retrieving indicator results: {e}")
            return None

    # ================================
    # Signal Aggregation Caching
    # ================================

    async def cache_aggregated_signals(
        self, asset: str, signals: List[Dict], ttl: Optional[int] = None
    ) -> bool:
        """Cache aggregated signals from MTF analysis"""
        try:
            client = await self._get_client()
            key = f"signals:aggregated:{asset}"

            cache_data = {
                "signals": signals,
                "cached_at": str(asyncio.get_event_loop().time()),
                "asset": asset,
                "signal_count": len(signals),
            }

            serialized_data = json.dumps(cache_data, default=str)
            ttl = ttl or self.config.indicator_ttl

            await client.setex(key, ttl, serialized_data)
            self.logger.debug(f"Cached {len(signals)} aggregated signals for {asset}")
            return True

        except Exception as e:
            self.logger.error(f"Error caching aggregated signals for {asset}: {e}")
            return False

    async def get_aggregated_signals(self, asset: str) -> Optional[List[Dict]]:
        """Retrieve cached aggregated signals"""
        try:
            client = await self._get_client()
            key = f"signals:aggregated:{asset}"

            data = await client.get(key)
            if data:
                cache_data = json.loads(data)
                return cache_data.get("signals", [])
            return None

        except Exception as e:
            self.logger.error(f"Error retrieving aggregated signals for {asset}: {e}")
            return None

    # ================================
    # Configuration Caching
    # ================================

    async def cache_asset_config(
        self, asset: str, config: Dict, ttl: Optional[int] = None
    ) -> bool:
        """Cache asset configuration"""
        try:
            client = await self._get_client()
            key = f"config:asset:{asset}"

            serialized_data = json.dumps(config, default=str)
            ttl = ttl or self.config.default_ttl

            await client.setex(key, ttl, serialized_data)
            self.logger.debug(f"Cached configuration for {asset}")
            return True

        except Exception as e:
            self.logger.error(f"Error caching asset config for {asset}: {e}")
            return False

    async def get_asset_config(self, asset: str) -> Optional[Dict]:
        """Retrieve cached asset configuration"""
        try:
            client = await self._get_client()
            key = f"config:asset:{asset}"

            data = await client.get(key)
            if data:
                return json.loads(data)
            return None

        except Exception as e:
            self.logger.error(f"Error retrieving asset config for {asset}: {e}")
            return None

    # ================================
    # Generic Cache Operations
    # ================================

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Generic set operation"""
        try:
            client = await self._get_client()

            if isinstance(value, (dict, list)):
                value = json.dumps(value, default=str)

            if ttl:
                await client.setex(key, ttl, value)
            else:
                await client.set(key, value)

            return True

        except Exception as e:
            self.logger.error(f"Error setting key {key}: {e}")
            return False

    async def set_key(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Alias for set() to match indicator_calc_service usage"""
        return await self.set(key, value, ttl)

    async def get(self, key: str) -> Optional[Any]:
        """Generic get operation"""
        try:
            client = await self._get_client()
            return await client.get(key)

        except Exception as e:
            self.logger.error(f"Error getting key {key}: {e}")
            return None

    async def delete(self, key: str) -> bool:
        """Delete a key"""
        try:
            client = await self._get_client()
            result = await client.delete(key)
            return bool(result)

        except Exception as e:
            self.logger.error(f"Error deleting key {key}: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        try:
            client = await self._get_client()
            return bool(await client.exists(key))

        except Exception as e:
            self.logger.error(f"Error checking existence of key {key}: {e}")
            return False

    async def expire(self, key: str, ttl: int) -> bool:
        """Set TTL for existing key"""
        try:
            client = await self._get_client()
            return bool(await client.expire(key, ttl))

        except Exception as e:
            self.logger.error(f"Error setting TTL for key {key}: {e}")
            return False

    async def clear_pattern(self, pattern: str) -> int:
        """Clear keys matching pattern"""
        try:
            client = await self._get_client()
            keys = await client.keys(pattern)
            if keys:
                deleted = await client.delete(*keys)
                self.logger.info(f"Cleared {deleted} keys matching pattern: {pattern}")
                return deleted
            return 0

        except Exception as e:
            self.logger.error(f"Error clearing pattern {pattern}: {e}")
            return 0

    async def incr(self, key: str, ttl: Optional[int] = None) -> int:
        """Increment value of a key"""
        try:
            client = await self._get_client()
            val = await client.incr(key)
            if ttl:
                await client.expire(key, ttl)
            return val
        except Exception as e:
            self.logger.error(f"Error incrementing key {key}: {e}")
            return 0

    async def expire(self, key: str, time: int) -> bool:
        """Set expiration for a key"""
        try:
            client = await self._get_client()
            return await client.expire(key, time)
        except Exception as e:
            self.logger.error(f"Error setting expiration for {key}: {e}")
            return False

    # ================================
    # Health & Status Methods
    # ================================

    async def ping(self) -> bool:
        """Test Redis connection"""
        try:
            client = await self._get_client()
            response = await client.ping()
            return bool(response)

        except Exception as e:
            self.logger.error(f"Redis ping failed: {e}")
            return False

    async def get_info(self) -> Optional[Dict]:
        """Get Redis server info"""
        try:
            client = await self._get_client()
            return await client.info()

        except Exception as e:
            self.logger.error(f"Error getting Redis info: {e}")
            return None

    # ================================
    # System Metrics (Time Series)
    # ================================

    async def add_metric(self, key: str, value: Dict, max_len: int = 1000) -> bool:
        """
        Add a metric data point to a time-series list.
        Pushes to the right and trims from the left to maintain max_len.
        """
        try:
            client = await self._get_client()
            serialized_value = json.dumps(value, default=str)
            
            # Use a pipeline for atomicity
            async with client.pipeline() as pipe:
                pipe.rpush(key, serialized_value)
                pipe.ltrim(key, -max_len, -1) # Keep only the last max_len elements
                await pipe.execute()
                
            return True
        except Exception as e:
            self.logger.error(f"Error adding metric to {key}: {e}")
            return False

    async def get_metrics(self, key: str, limit: int = 100) -> List[Dict]:
        """
        Retrieve the last N metrics from the list.
        """
        try:
            client = await self._get_client()
            # Get last 'limit' elements
            # lrange start end (inclusive)
            # -limit to -1 gets the last 'limit' items
            raw_data = await client.lrange(key, -limit, -1)
            
            return [json.loads(item) for item in raw_data]
        except Exception as e:
            self.logger.error(f"Error retrieving metrics from {key}: {e}")
            return []

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache usage statistics"""
        try:
            client = await self._get_client()
            info = await client.info()

            # Count keys by pattern
            market_data_keys = len(await client.keys("market_data:*"))
            indicator_keys = len(await client.keys("indicators:*"))
            signal_keys = len(await client.keys("signals:*"))
            config_keys = len(await client.keys("config:*"))

            return {
                "connected": self.connected,
                "total_keys": info.get("db0", {}).get("keys", 0) if info else 0,
                "used_memory": info.get("used_memory_human", "Unknown")
                if info
                else "Unknown",
                "hit_rate": f"{info.get('keyspace_hits', 0) / max(info.get('keyspace_hits', 0) + info.get('keyspace_misses', 0), 1) * 100:.2f}%"
                if info
                else "Unknown",
                "key_counts": {
                    "market_data": market_data_keys,
                    "indicators": indicator_keys,
                    "signals": signal_keys,
                    "config": config_keys,
                },
            }

        except Exception as e:
            self.logger.error(f"Error getting cache stats: {e}")
            return {"connected": False, "error": str(e)}