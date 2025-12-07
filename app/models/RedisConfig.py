from dataclasses import dataclass
import os

from dotenv import load_dotenv


load_dotenv()


@dataclass
class RedisConfig:
    """Configuration settings for Redis connection and caching."""

    def __init__(self):
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.redis_host = os.getenv("REDIS_HOST", "localhost")
        self.redis_port = int(os.getenv("REDIS_PORT", "6379"))
        self.redis_db = int(os.getenv("REDIS_DB", "0"))
        self.redis_password = os.getenv("REDIS_PASSWORD")
        self.logger_name = "app"

        # Connection pool settings
        self.max_connections = int(os.getenv("REDIS_MAX_CONNECTIONS", "20"))
        self.socket_timeout = int(os.getenv("REDIS_SOCKET_TIMEOUT", "5"))

        # Default TTL settings
        self.default_ttl = int(os.getenv("REDIS_DEFAULT_TTL", "3600"))  # 1 hour
        self.market_data_ttl = int(os.getenv("REDIS_MARKET_DATA_TTL", "300"))  # 5 minutes
        self.indicator_ttl = int(os.getenv("REDIS_INDICATOR_TTL", "900"))  # 15 minutes
