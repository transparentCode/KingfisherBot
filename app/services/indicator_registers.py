import logging
import threading
from typing import Dict, Any, Optional, Type, List
from dataclasses import dataclass

from app.indicators.custom_supertrend import StudentTSuperTrend
from app.indicators.exponential_moving_average import ExponentialMovingAverage
from app.indicators.rsi import RSI
from app.indicators.simple_moving_average import SimpleMovingAverage
from app.indicators.trendline_with_breaks import TrendLineWithBreaks


@dataclass
class IndicatorMetadata:
    """Metadata for a registered indicator"""
    display_name: str
    class_ref: Type
    description: str
    category: str
    parameter_schema: Dict[str, Any]
    input_columns: List[str]


class IndicatorRegistry:
    """singleton registry for managing indicators"""

    _instance: Optional['IndicatorRegistry'] = None
    _lock = threading.Lock()

    def __new__(cls) -> 'IndicatorRegistry':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return

        self.logger = logging.getLogger("app")
        self.registered_indicators: Dict[str, IndicatorMetadata] = {}
        self._initialized = True

    def register_indicator(
            self,
            indicator_id: str,
            display_name: str,
            class_ref: Type,
            description: Optional[str] = None,
            category: str = "Technical"
    ) -> None:
        """
        Register a new indicator with enhanced metadata.

        Args:
            indicator_id: Unique identifier for the indicator
            display_name: Human-readable name
            class_ref: The indicator class
            description: Optional description
            category: Indicator category

        Raises:
            ValueError: If indicator_id is invalid or class_ref is not callable
        """
        if not indicator_id or not isinstance(indicator_id, str):
            raise ValueError("indicator_id must be a non-empty string")

        if not callable(class_ref):
            raise ValueError("class_ref must be a callable class")

        if indicator_id in self.registered_indicators:
            self.logger.warning(f"Indicator '{indicator_id}' is already registered, skipping")
            return

        try:
            # Safely create temporary instance to extract metadata
            temp_instance = class_ref()
            parameter_schema = getattr(temp_instance, 'get_parameter_schema', lambda: {})()
            input_columns = getattr(temp_instance, 'input_columns', ['close'])

            # Clean up temporary instance if it has cleanup method
            if hasattr(temp_instance, 'cleanup'):
                temp_instance.cleanup()

        except Exception as e:
            self.logger.error(f"Failed to create temporary instance of {class_ref.__name__}: {e}")
            raise ValueError(f"Invalid indicator class: {class_ref.__name__}")

        metadata = IndicatorMetadata(
            display_name=display_name,
            class_ref=class_ref,
            description=description or '',
            category=category,
            parameter_schema=parameter_schema,
            input_columns=input_columns
        )

        self.registered_indicators[indicator_id] = metadata
        self.logger.info(f"Successfully registered indicator: {indicator_id}")

    def get_indicator_for_ui(self, indicator_id: str) -> Optional[Dict[str, Any]]:
        """
        Get indicator metadata formatted for UI consumption.

        Args:
            indicator_id: The indicator identifier

        Returns:
            Formatted indicator data or None if not found
        """
        metadata = self.registered_indicators.get(indicator_id)
        if not metadata:
            self.logger.warning(f"Indicator '{indicator_id}' not found")
            return None

        return {
            'id': indicator_id,
            'display_name': metadata.display_name,
            'description': metadata.description,
            'category': metadata.category,
            'parameters': metadata.parameter_schema,
            'input_columns': metadata.input_columns
        }

    def get_all_indicators_for_ui(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all indicators formatted for UI with categories.

        Returns:
            Dictionary with categories as keys and indicator lists as values
        """
        categories: Dict[str, List[Dict[str, Any]]] = {}

        for indicator_id in self.registered_indicators:
            indicator_ui_data = self.get_indicator_for_ui(indicator_id)
            if not indicator_ui_data:
                continue

            category = indicator_ui_data['category']
            if category not in categories:
                categories[category] = []

            categories[category].append(indicator_ui_data)

        return categories

    def create_indicator_instance(self, indicator_id: str, **params) -> Any:
        """
        Create indicator instance with custom parameters.

        Args:
            indicator_id: The indicator identifier
            **params: Parameters to pass to the indicator constructor

        Returns:
            Configured indicator instance

        Raises:
            ValueError: If indicator not found or creation fails
        """
        metadata = self.registered_indicators.get(indicator_id)
        if not metadata:
            raise ValueError(f"Indicator '{indicator_id}' not found")

        try:
            return metadata.class_ref(**params)
        except Exception as e:
            self.logger.error(f"Failed to create indicator '{indicator_id}': {e}")
            raise ValueError(f"Failed to create indicator '{indicator_id}': {e}")

    def get_indicators_by_category(self, category: str) -> Dict[str, IndicatorMetadata]:
        """
        Get all indicators from a specific category.

        Args:
            category: The category to filter by

        Returns:
            Dictionary of indicators in the specified category
        """
        return {
            indicator_id: metadata
            for indicator_id, metadata in self.registered_indicators.items()
            if metadata.category == category
        }

    def is_registered(self, indicator_id: str) -> bool:
        """Check if an indicator is registered"""
        return indicator_id in self.registered_indicators

    def get_all_categories(self) -> List[str]:
        """Get all available indicator categories"""
        return list(set(metadata.category for metadata in self.registered_indicators.values()))

    @staticmethod
    def register_default_indicators() -> 'IndicatorRegistry':
        """Register default indicators and return registry instance"""
        registry = IndicatorRegistry()

        if registry.registered_indicators:
            return registry

        default_indicators = [
            {
                'indicator_id': 'SMA',
                'display_name': 'Simple Moving Average',
                'class_ref': SimpleMovingAverage,
                'description': 'Basic moving average indicator',
                'category': 'Moving Averages'
            },
            {
                'indicator_id': 'EMA',
                'display_name': 'Exponential Moving Average',
                'class_ref': ExponentialMovingAverage,
                'description': 'Exponential moving average indicator',
                'category': 'Moving Averages'
            },
            {
                'indicator_id': 'RSI',
                'display_name': 'RSI',
                'class_ref': RSI,
                'description': 'Advanced RSI with Gaussian weighting',
                'category': 'Oscillators'
            },
            {
                'indicator_id': 'supertrend',
                'display_name': 'Supertrend Indicator',
                'class_ref': StudentTSuperTrend,
                'description': 'Trend-following indicator based on ATR',
                'category': 'Trend Analysis'
            }
        ]

        for indicator_config in default_indicators:
            try:
                registry.register_indicator(**indicator_config)
            except Exception as e:
                registry.logger.error(f"Failed to register {indicator_config['indicator_id']}: {e}")

        return registry
