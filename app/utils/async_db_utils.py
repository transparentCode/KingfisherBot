import asyncio
from typing import Callable, Any, Coroutine
from app.db.db_handler import DBHandler
import logging

logger = logging.getLogger(__name__)

async def precise_db_task(task_func: Callable[[DBHandler], Coroutine[Any, Any, Any]]) -> Any:
    """
    Executes a task with a fresh, ephemeral DBHandler instance.
    Handles initialization and cleanup automatically.
    
    Args:
        task_func: An async function that accepts a DBHandler instance and returns a result.
    
    Returns:
        The result of task_func.
    """
    db_handler = DBHandler()
    try:
        await db_handler.initialize()
        return await task_func(db_handler)
    except Exception as e:
        logger.error(f"DB Task Error: {e}", exc_info=True)
        raise
    finally:
        await db_handler.close()

def run_in_asyncio(coro: Coroutine) -> Any:
    """
    Helper to run a coroutine in a new event loop (synchronous wrapper).
    """
    return asyncio.run(coro)
