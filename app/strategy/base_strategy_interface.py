from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
from app.strategy.models import StrategySignal


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
        Should append signal columns to the dataframe.

        :param data: Price data as pandas DataFrame
        :return: DataFrame with strategy signals
        """
        pass
        
    def calculate_signal(self, data: pd.DataFrame) -> StrategySignal:
        """
        Calculate the signal for the latest candle.
        Default implementation runs execute() and takes the last row.
        Override for optimization if needed.
        """
        df_result = self.execute(data)
        if df_result is None or df_result.empty:
            return StrategySignal(self.strategy_name, 0, 0.0)
            
        last_row = df_result.iloc[-1]
        
        # Assume 'signal' column exists (1, -1, 0)
        # Assume 'confidence' column exists (optional, default 1.0)
        
        signal = 0
        if 'signal' in last_row:
            signal = int(last_row['signal'])
            
        confidence = 1.0
        if 'confidence' in last_row:
            confidence = float(last_row['confidence'])
            
        return StrategySignal(
            strategy_name=self.strategy_name,
            signal=signal,
            confidence=confidence,
            metadata=last_row.to_dict()
        )

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