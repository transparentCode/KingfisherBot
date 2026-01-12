import os
import shutil
import logging
from collections import deque
from typing import Dict, Any, List, Optional
import asyncio
from concurrent.futures import TimeoutError

logger = logging.getLogger(__name__)

def get_disk_usage(path: str = "/") -> Dict[str, Any]:
    """Get disk usage statistics."""
    try:
        total, used, free = shutil.disk_usage(path)
        return {
            'total_gb': round(total / (2**30), 2),
            'used_gb': round(used / (2**30), 2),
            'free_gb': round(free / (2**30), 2),
            'percent': round((used / total) * 100, 1)
        }
    except Exception as e:
        logger.error(f"Error checking disk usage: {e}")
        return {'error': str(e)}

def get_system_load() -> List[float]:
    """Get system load average (1, 5, 15 min)."""
    try:
        return list(os.getloadavg())
    except Exception as e:
        logger.error(f"Error checking system load: {e}")
        return []

def read_log_tail(file_path: str, lines: int = 100) -> List[str]:
    """
    Read the last N lines of a file efficiently.
    Returns list of strings.
    """
    if not os.path.exists(file_path):
        return []
        
    try:
        with open(file_path, 'r') as f:
            # efficient for large files, deque(file, maxlen=n) reads entire file but keeps only n
            # For extremely large files, one should seek backwards, but deque is better than readlines()
            # method which loads everything into list memory.
            return list(deque(f, maxlen=lines))
    except Exception as e:
        logger.error(f"Error reading log file {file_path}: {e}")
        return [f"Error reading logs: {str(e)}"]

def get_market_service_status(market_service, timeout: int = 2) -> Dict[str, Any]:
    """
    Safely fetch status from the running MarketService instance.
    """
    if not market_service or not market_service.loop or not market_service.loop.is_running():
        return {'status': 'offline', 'error': 'MarketService not running or loop closed'}

    try:
        future = asyncio.run_coroutine_threadsafe(
            market_service.get_status(),
            market_service.loop
        )
        return future.result(timeout=timeout)
    except TimeoutError:
        logger.error("Timeout fetching MarketService status")
        return {'error': 'timeout'}
    except Exception as e:
        logger.error(f"Error fetching MarketService status: {e}")
        return {'error': str(e)}
