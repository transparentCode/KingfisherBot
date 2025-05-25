from typing import Dict, Union, List, Optional, Any
import pandas as pd

from app.strategy.base_strategy_interface import BaseStrategyInterface
from app.strategy.rsi_strategy import RSIStrategy
from app.strategy.sma_confluence import SmaConfluence


class StrategyFactory:
    """
    Factory class for creating and managing trading strategies.
    Supports generating signals using different strategy implementations.
    """

    def __init__(self):
        self.strategies = {}
        self.register_default_strategies()

    def register_default_strategies(self):
        """Register built-in strategy implementations."""
        self.register_strategy("sma_confluence", SmaConfluence)
        self.register_strategy("rsi_strategy", RSIStrategy)

    def register_strategy(self, strategy_id: str, strategy_class):
        """
        Register a new strategy implementation.

        :param strategy_id: Unique identifier for the strategy
        :param strategy_class: Strategy class reference
        """
        self.strategies[strategy_id] = strategy_class

    def create_strategy(self, strategy_id: str, **kwargs) -> BaseStrategyInterface:
        """
        Create a strategy instance with the given parameters.

        :param strategy_id: ID of the strategy to create
        :param kwargs: Parameters for strategy initialization
        :return: Initialized strategy instance
        """
        if strategy_id not in self.strategies:
            raise ValueError(f"Strategy '{strategy_id}' not registered")

        strategy_class = self.strategies[strategy_id]
        strategy = strategy_class()
        strategy.initialize(**kwargs)
        return strategy

    def generate_signals(self, strategy_id: str, data: pd.DataFrame,
                         parameters: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Generate trading signals using the specified strategy.

        :param strategy_id: ID of the strategy to use
        :param data: Market data as DataFrame
        :param parameters: Optional parameters for the strategy
        :return: DataFrame with generated signals
        """
        if parameters is None:
            parameters = {}

        strategy = self.create_strategy(strategy_id, **parameters)
        results = strategy.execute(data)

        return results

    def list_available_strategies(self) -> List[str]:
        """
        Get list of available strategy IDs.

        :return: List of strategy identifiers
        """
        return list(self.strategies.keys())

    def get_strategy_defaults(self, strategy_id: str) -> Dict[str, Any]:
        """
        Get default parameters for a strategy.

        :param strategy_id: ID of the strategy
        :return: Dictionary of default parameters
        """
        if strategy_id not in self.strategies:
            raise ValueError(f"Strategy '{strategy_id}' not registered")

        strategy = self.strategies[strategy_id]()
        strategy.initialize()
        return strategy.get_parameters()
