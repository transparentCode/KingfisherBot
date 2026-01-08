from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any

class BaseSignalFactor(ABC):
    """
    Base class for all signal factors (Layer A).
    """
    def __init__(self, name: str):
        self.name = name
        self.params = {}

    def initialize(self, **kwargs):
        self.params = kwargs

    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> float:
        """
        Calculate the signal score for the latest candle.
        Returns: float between -1.0 (Bearish) and 1.0 (Bullish).
        """
        pass
