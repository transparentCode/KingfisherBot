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