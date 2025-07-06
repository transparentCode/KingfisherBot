from app.indicators.custom_supertrend import StudentTSuperTrend
from app.indicators.rsi import RSI
from app.indicators.simple_moving_average import SimpleMovingAverage
from app.indicators.trendline_with_breaks import TrendLineWithBreaks

import logging


class IndicatorRegisters:
    _instance = None
    logger = None

    def __init__(self):
        self.logger = logging.getLogger("app")

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(IndicatorRegisters, cls).__new__(cls)
            cls._instance.registered_indicators = {}
        return cls._instance

    def register_indicator(self, indicator_id, display_name, class_ref, description=None, category="Technical"):
        """Register a new indicator with enhanced metadata."""
        if indicator_id in self.registered_indicators:
            self.logger.info(f"Indicator '{indicator_id}' is already registered.")
            return

        # Get parameter schema from indicator instance
        temp_instance = class_ref()
        parameter_schema = getattr(temp_instance, 'get_parameter_schema', lambda: {})()

        self.registered_indicators[indicator_id] = {
            'display_name': display_name,
            'class': class_ref,
            'description': description or '',
            'category': category,
            'parameter_schema': parameter_schema,
            'input_columns': getattr(temp_instance, 'input_columns', ['close'])
        }

    def get_indicator_for_ui(self, indicator_id):
        """Get indicator metadata formatted for UI consumption."""
        indicator = self.registered_indicators.get(indicator_id)
        if not indicator:
            return None

        return {
            'id': indicator_id,
            'display_name': indicator['display_name'],
            'description': indicator['description'],
            'category': indicator['category'],
            'parameters': indicator['parameter_schema'],
            'input_columns': indicator['input_columns']
        }

    def get_all_indicators_for_ui(self):
        """Get all indicators formatted for UI with categories."""
        categories = {}
        for indicator_id, indicator_data in self.registered_indicators.items():
            category = indicator_data['category']

            if category not in categories:
                categories[category] = []

            categories[category].append(self.get_indicator_for_ui(indicator_id))

        return categories

    def create_indicator_instance(self, indicator_id, **params):
        """Create indicator instance with custom parameters."""
        indicator_data = self.registered_indicators.get(indicator_id)
        if not indicator_data:
            raise ValueError(f"Indicator '{indicator_id}' not found")

        return indicator_data['class'](**params)

    @staticmethod
    def register_indicators():
        indicator_register = IndicatorRegisters()

        if indicator_register.registered_indicators:
            return

        # Enhanced registration with categories
        indicator_register.register_indicator(
            indicator_id="trendline_with_breaks",
            display_name="Trend Line with Breaks",
            class_ref=TrendLineWithBreaks,
            description='Detects trend lines and breakout points',
            category="Trend Analysis"
        )

        indicator_register.register_indicator(
            indicator_id="SMA",
            display_name="Simple Moving Average",
            class_ref=SimpleMovingAverage,
            description='Basic moving average indicator',
            category="Moving Averages"
        )

        indicator_register.register_indicator(
            indicator_id="RSI",
            display_name="RSI",
            class_ref=RSI,
            description='Advanced RSI with Gaussian weighting',
            category="Oscillators"
        )

        indicator_register.register_indicator(
            indicator_id="supertrend",
            display_name="Supertrend Indicator",
            class_ref=StudentTSuperTrend,
            description='Trend-following indicator based on ATR',
            category="Trend Analysis"
        )