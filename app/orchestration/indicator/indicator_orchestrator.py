import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from app.models.indicator_context import IndicatorExecutionContext
from app.services.indicator_registers import IndicatorRegistry, IndicatorMetadata


class BaseIndicatorOrchestrator(ABC):
    """
    Abstract base class for indicator orchestrators.
    """

    def __init__(self, indicator_registry: IndicatorRegistry, logger_name: str = "app"):
        self.indicator_registry = indicator_registry
        self.logger = logging.getLogger(logger_name)

    @abstractmethod
    async def execute(self, context: IndicatorExecutionContext) -> Dict[str, Any]:
        pass

    def _validate_context(self, context: IndicatorExecutionContext) -> bool:
        """
        Validate that the execution context has required data.

        Args:
            context: The execution context to validate

        Returns:
            True if context is valid, False otherwise
        """
        if not context:
            self.logger.error("Execution context is None")
            return False

        if not context.asset:
            self.logger.error("Asset not specified in context")
            return False

        if not context.data_cache:
            self.logger.error("No data available in context")
            return False

        if not context.primary_timeframe:
            self.logger.error("Primary timeframe not specified")
            return False

        return True

    def _get_indicators_by_category(self, category: str) -> Dict[str, IndicatorMetadata]:
        try:
            return self.indicator_registry.get_indicators_by_category(category)
        except Exception as e:
            self.logger.error(f"Error retrieving indicators for category '{category}': {e}")
            return {}

    def _create_indicator_instance(self, indicator_id: str, **params) -> Optional[Any]:
        try:
            return self.indicator_registry.create_indicator_instance(indicator_id, **params)
        except Exception as e:
            self.logger.error(f"Failed to create indicator instance '{indicator_id}': {e}")
            return None
