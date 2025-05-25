from app.indicators.rsi import RSI
from app.indicators.simple_moving_average import SimpleMovingAverage
from app.indicators.trendline_with_breaks import TrendLineWithBreaks


class IndicatorRegisters:

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(IndicatorRegisters, cls).__new__(cls)
            cls._instance.registered_indicators = {}
        return cls._instance

    def register_indicator(self, indicator_id, display_name, class_ref, description=None):
        """
        Register a new indicator with its ID, display name, and class reference.
        """
        if indicator_id in self.registered_indicators:
            raise ValueError(f"Indicator '{indicator_id}' is already registered.")

        self.registered_indicators[indicator_id] = {
            'display_name': display_name,
            'class': class_ref,
            'description': description or ''
        }

    def get_indicator(self, indicator_id):
        """
        Get an indicator by ID.
        """
        return self.registered_indicators.get(indicator_id)

    def get_all_indicators(self):
        """
        Get all registered indicators.
        """
        return self.registered_indicators

    @staticmethod
    def register_indicators():
        indicator_register = IndicatorRegisters()
        """
        Register an indicator class to the global registry.
        """
        indicator_register.register_indicator(indicator_id="trendline_with_breaks",
                                              display_name="Trend Line with Breaks",
                                              class_ref=TrendLineWithBreaks, description='')
        indicator_register.register_indicator(indicator_id="SMA",
                                              display_name="Simple Moving Average",
                                              class_ref=SimpleMovingAverage, description='')
        indicator_register.register_indicator(indicator_id="RSI",
                                              display_name="Relative Strength Index",
                                              class_ref=RSI, description='')