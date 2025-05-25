from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd


class BaseStrategyInterface(ABC):
    """
    Base interface for all trading strategies.
    """

    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name
        self.data = None
        self.parameters = {}
        self.signals = None

    @abstractmethod
    def initialize(self, **kwargs):
        """
        Initialize the strategy with parameters.

        :param kwargs: Strategy parameters
        """
        pass

    @abstractmethod
    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Execute the strategy on the provided data.

        :param data: Price data as pandas DataFrame
        :return: DataFrame with strategy signals
        """
        pass

    @abstractmethod
    def plot(self, data: Optional[pd.DataFrame] = None, **kwargs):
        """
        Plot the strategy signals and indicators.

        :param data: Optional data to plot (uses self.data if None)
        :param kwargs: Additional plotting parameters
        :return: Plotly figure object
        """
        pass

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy/sell signals based on strategy logic.

        :param data: DataFrame with technical indicators
        :return: DataFrame with added signal columns
        """
        return data

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get the strategy parameters.

        :return: Dictionary of strategy parameters
        """
        return self.parameters

    def set_parameters(self, parameters: Dict[str, Any]):
        """
        Update strategy parameters.

        :param parameters: Dictionary of parameters to update
        """
        self.parameters.update(parameters)