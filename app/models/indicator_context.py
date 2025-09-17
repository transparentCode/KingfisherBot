from dataclasses import dataclass
from typing import Dict

import pandas as pd

@dataclass
class IndicatorExecutionContext:
    def __init__(self, asset: str, primary_timeframe: str, data_cache: Dict[str, pd.DataFrame]):
        self.asset = asset
        self.primary_timeframe = primary_timeframe
        self.data_cache = data_cache  # MTF data cache
        self.results = {}
        self.metadata = {}  # Store additional context data