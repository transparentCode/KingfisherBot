from abc import ABC
from typing import Dict, Any, List
import pandas as pd


class BaseIndicatorInterface(ABC):
    """
    Base interface for all indicators.
    """

    def __init__(
            self,
            name: str,
            input_columns: List[str] = None,
            **kwargs
    ):
        """
        Initialize the indicator with a name and optional parameters.

        :param name: The name of the indicator.
        :param input_columns: List of input columns for the indicator.
        :param kwargs: Optional parameters for the indicator.
        """
        self.name = name
        self.input_columns = input_columns if input_columns is not None else []

    def calculate(self, data: pd.DataFrame, **kwargs) -> float:
        """
        Calculate the indicator value based on the provided data.

        :param data: The data to calculate the indicator on.
        :return: The calculated indicator value.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def plot(self, data: pd.DataFrame, **kwargs) -> None:
        """
        Plot the indicator on the provided data.

        :param data: The data to plot the indicator on.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def _get_plot_trace(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Get the plot trace for the indicator.

        :param data: The data to generate the plot trace from.
        :return: A dictionary representing the plot trace.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def _get_default_params(self):
        """Return default parameters for this indicator"""
        return {}

    def get_parameter_schema(self):
        """Return parameter schema for UI generation"""
        return {
            'name': self.name,
            'parameters': self._get_default_params(),
            'category': getattr(self, 'category', 'Technical')
        }

    def _clean_data_for_json(self, data, columns):
        """Helper method to clean data for JSON serialization"""
        import pandas as pd

        # Remove NaN values and ensure proper data types
        clean_data = data[columns].dropna()

        result = {}
        for col in columns:
            if col in clean_data.columns:
                # Convert timestamps to ISO format strings
                if isinstance(clean_data.index, pd.DatetimeIndex):
                    result['timestamps'] = [ts.isoformat() for ts in clean_data.index]
                else:
                    result['timestamps'] = clean_data.index.tolist()

                # Convert values to list, replacing any remaining NaN with null
                values = clean_data[col].tolist()
                result[col] = [None if pd.isna(val) else val for val in values]

        return result