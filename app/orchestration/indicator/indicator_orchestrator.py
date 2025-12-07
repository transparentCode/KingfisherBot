import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, TYPE_CHECKING

from app.db.redis_handler import RedisHandler
from app.models.indicator_context import IndicatorExecutionContext
from app.services.indicator_registers import IndicatorRegistry, IndicatorMetadata

if TYPE_CHECKING:
    from config.asset_indicator_config import ConfigurationManager


class BaseIndicatorOrchestrator(ABC):
    """
    Abstract base class for indicator orchestrators.
    """

    def __init__(
        self, 
        indicator_registry: IndicatorRegistry, 
        logger_name: str = "app",
        config_manager: Optional['ConfigurationManager'] = None
    ):
        self.indicator_registry = indicator_registry
        self.logger = logging.getLogger(logger_name)
        self.config_manager = config_manager
        self.redis_handler = RedisHandler()

    @abstractmethod
    async def execute(self, context: IndicatorExecutionContext) -> Dict[str, Any]:
        pass

    def _validate_context(self, context: IndicatorExecutionContext) -> bool:
        """Validate that the execution context has required data."""
        if not context:
            self.logger.error("Execution context is None")
            return False

        if not context.asset:
            self.logger.error("Asset not specified in context")
            return False

        if not context.data_cache:
            self.logger.error(f"No data available in context for asset {context.asset}")
            return False

        if not context.primary_timeframe:
            self.logger.error(f"Primary timeframe not specified for asset {context.asset}")
            return False

        if context.primary_timeframe not in context.data_cache:
            self.logger.error(
                f"Primary timeframe '{context.primary_timeframe}' not found in data cache "
                f"for asset {context.asset}. Available: {list(context.data_cache.keys())}"
            )
            return False

        return True

    def _get_indicators_by_category(self, category: str) -> Dict[str, IndicatorMetadata]:
        try:
            indicators = self.indicator_registry.get_indicators_by_category(category)
            self.logger.debug(f"Retrieved {len(indicators)} indicators for category '{category}'")
            return indicators
        except Exception as e:
            self.logger.error(f"Error retrieving indicators for category '{category}': {e}")
            return {}

    def _create_indicator_instance(self, indicator_id: str, **params) -> Optional[Any]:
        try:
            return self.indicator_registry.create_indicator_instance(indicator_id, **params)
        except Exception as e:
            self.logger.error(f"Failed to create indicator instance '{indicator_id}': {e}")
            return None

    async def _cache_results(
        self, 
        asset: str, 
        indicator_type: str, 
        timeframe: str, 
        results: Dict[str, Any]
    ) -> bool:
        """Cache indicator results to Redis."""
        try:
            return await self.redis_handler.cache_indicator_results(
                asset=asset,
                indicator_type=indicator_type,
                timeframe=timeframe,
                results=results
            )
        except Exception as e:
            self.logger.error(f"Failed to cache {indicator_type} for {asset}:{timeframe}: {e}")
            return False

    async def _get_cached_results(
        self, 
        asset: str, 
        indicator_type: str, 
        timeframe: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve cached indicator results from Redis."""
        try:
            return await self.redis_handler.get_indicator_results(
                asset=asset,
                indicator_type=indicator_type,
                timeframe=timeframe
            )
        except Exception as e:
            self.logger.error(f"Failed to get cached {indicator_type} for {asset}:{timeframe}: {e}")
            return None