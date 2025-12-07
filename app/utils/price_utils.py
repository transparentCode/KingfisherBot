import numpy as np
import pandas as pd
import re
from typing import Dict, Optional, Union, List

def get_price_source_data(data, source):
    """
    Calculate price source data based on the specified source type.

    :param data: DataFrame containing OHLC price data
    :param source: Source type ('close', 'hlc3', 'hl2', 'ohlc4', 'hlcc4', or column name)
    :return: Series with calculated source data
    """
    if source == 'hlcc4':
        return (data['high'] + data['low'] + data['close'] + data['close']) / 4
    elif source == 'hlc3':
        return (data['high'] + data['low'] + data['close']) / 3
    elif source == 'hl2':
        return (data['high'] + data['low']) / 2
    elif source == 'ohlc4':
        return (data['open'] + data['high'] + data['low'] + data['close']) / 4
    elif source in data.columns:
        return data[source]
    else:
        raise ValueError(f"Invalid source '{source}'. Must be a column name or a valid combination.")


def get_available_price_sources():
    """Return list of available price source options."""
    return ['close', 'open', 'high', 'low', 'hlc3', 'hl2', 'ohlc4', 'hlcc4']


class PriceUtils:
    """Utility class for price and indicator calculations."""

    @staticmethod
    def calculate_slope(series: pd.Series, lookback: int = 5, method: str = 'linear_regression') -> float:
        """
        Calculate slope using configurable method.
        Linear Regression is deterministic (OLS) and more robust to noise than simple ROC.
        """
        try:
            # 1. Slice the data to the specific lookback window
            series_clean = series.dropna().tail(lookback)
            
            if len(series_clean) < 3:
                return 0.0
            
            if method == 'linear_regression':
                # Deterministic Linear Algebra (Least Squares)
                y = series_clean.values
                x = np.arange(len(y))
                
                # np.polyfit is faster than sklearn.LinearRegression for 1D data
                # It returns [slope, intercept]
                slope, _ = np.polyfit(x, y, 1)
                return float(slope)
                
            elif method == 'simple':
                # Simple Rate of Change (End - Start)
                recent_values = series_clean.values
                return float((recent_values[-1] - recent_values[0]) / len(recent_values))
                
            else:
                # Default fallback
                recent_values = series_clean.values
                return float((recent_values[-1] - recent_values[0]) / len(recent_values))
                
        except Exception as e:
            # In a utility class, we might want to log, but for now let's return 0.0
            return 0.0

    @staticmethod
    def calculate_confluence_line(ma_values: Dict[str, float], method: str = 'mean') -> Dict:
        """Calculate explicit confluence line from EMA values with validation"""
        ema_data = {}
        for indicator_id, value in ma_values.items():
            # Improved check: Ensure value is valid number
            if pd.notna(value) and value > 0:
                # Try to get period from ID if possible, or just use the ID as key
                period = PriceUtils.extract_ema_period(indicator_id)
                key = period if period else indicator_id
                ema_data[key] = float(value)
        
        if len(ema_data) < 2:
            return {'line': None, 'method': method, 'ema_count': len(ema_data)}
        
        values = np.array(list(ema_data.values()))
        
        if method == 'median':
            confluence_line = np.median(values)
        else:  # 'mean' default
            confluence_line = np.mean(values)
        
        return {
            'line': confluence_line,
            'method': method,
            'ema_count': len(ema_data),
            'ema_data': ema_data
        }

    @staticmethod
    def extract_ema_period(indicator_id: str) -> Optional[int]:
        """Extract EMA period from indicator ID string"""
        try:
            numbers = re.findall(r'\d+', indicator_id)
            if numbers:
                return int(numbers[0])
            if 'fast' in indicator_id.lower(): return 9
            if 'medium' in indicator_id.lower(): return 21
            if 'slow' in indicator_id.lower(): return 50
            return None
        except:
            return None

