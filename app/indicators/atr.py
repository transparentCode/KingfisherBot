import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from app.indicators.BaseIndicatorInterface import BaseIndicatorInterface


class AverageTrueRange(BaseIndicatorInterface):
    """Average True Range (ATR) indicator with configurable EMA smoothing."""

    def __init__(self, name: str = "AverageTrueRange", atr_period: int = 14, ema_period: int = 14, smoothing_type: str = "ema", **kwargs):
        super().__init__(name)
        self.name = name
        self.atr_period = atr_period  # Period for ATR calculation
        self.ema_period = ema_period  # Period for EMA smoothing of ATR
        self.smoothing_type = smoothing_type  # 'ema', 'sma', or 'rma'
        self.input_columns = ['high', 'low', 'close', 'open']
        self.category = 'Volatility'

    def calculate(self, data, **kwargs):
        """
        Calculate Average True Range with configurable smoothing.

        :param data: DataFrame containing high, low, close price data
        :param kwargs: Optional arguments to override class settings
        :return: DataFrame with ATR values added
        """
        atr_period = kwargs.get('atr_period', self.atr_period)
        ema_period = kwargs.get('ema_period', self.ema_period)
        smoothing_type = kwargs.get('smoothing_type', self.smoothing_type)

        # Ensure we have required columns
        required_cols = ['high', 'low', 'close']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")

        # Calculate True Range components
        tr1 = data['high'] - data['low']  # High - Low
        tr2 = abs(data['high'] - data['close'].shift(1))  # High - Previous Close
        tr3 = abs(data['low'] - data['close'].shift(1))   # Low - Previous Close

        # True Range is the maximum of the three
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Store raw True Range
        data[f'tr_{atr_period}'] = true_range

        # Calculate ATR based on smoothing type with separate periods
        column_name = f'atr_{atr_period}_{smoothing_type}_{ema_period}'
        
        if smoothing_type.lower() == 'ema':
            # Exponential Moving Average of True Range
            data[column_name] = true_range.ewm(span=ema_period, adjust=False).mean()
        elif smoothing_type.lower() == 'sma':
            # Simple Moving Average of True Range
            data[column_name] = true_range.rolling(window=ema_period).mean()
        elif smoothing_type.lower() == 'rma':
            # Wilder's Moving Average (RMA) - traditional ATR calculation
            alpha = 1.0 / ema_period
            data[column_name] = true_range.ewm(alpha=alpha, adjust=False).mean()
        else:
            raise ValueError(f"Unsupported smoothing type: {smoothing_type}. Use 'ema', 'sma', or 'rma'")

        # Also create a simplified column name for backward compatibility
        data[f'atr_{atr_period}'] = data[column_name]

        return data

    def calculate_atr_bands(self, data, price_source='close', multiplier=2.0, **kwargs):
        """
        Calculate ATR-based bands (useful for SuperTrend, Keltner Channels, etc.)
        
        :param data: DataFrame with calculated ATR
        :param price_source: Price to use as center line ('close', 'hlc3', etc.)
        :param multiplier: ATR multiplier for bands
        :return: DataFrame with upper and lower ATR bands
        """
        atr_period = kwargs.get('atr_period', self.atr_period)
        ema_period = kwargs.get('ema_period', self.ema_period)
        smoothing_type = kwargs.get('smoothing_type', self.smoothing_type)
        atr_column = f'atr_{atr_period}_{smoothing_type}_{ema_period}'
        
        if atr_column not in data.columns:
            data = self.calculate(data, **kwargs)
        
        # Get price source data
        if price_source == 'close':
            center_line = data['close']
        elif price_source == 'hlc3':
            center_line = (data['high'] + data['low'] + data['close']) / 3
        elif price_source == 'hl2':
            center_line = (data['high'] + data['low']) / 2
        elif price_source == 'ohlc4':
            center_line = (data['open'] + data['high'] + data['low'] + data['close']) / 4
        else:
            center_line = data[price_source]
        
        # Calculate bands
        data[f'atr_upper_{atr_period}_{ema_period}_{multiplier}'] = center_line + (multiplier * data[atr_column])
        data[f'atr_lower_{atr_period}_{ema_period}_{multiplier}'] = center_line - (multiplier * data[atr_column])
        
        return data

    def plot(self, data, **kwargs):
        """
        Plot the ATR indicator.

        :param data: DataFrame containing price data and calculated ATR
        :param kwargs: Optional parameters for plotting
        """
        atr_period = kwargs.get('atr_period', self.atr_period)
        ema_period = kwargs.get('ema_period', self.ema_period)
        smoothing_type = kwargs.get('smoothing_type', self.smoothing_type)
        column_name = f'atr_{atr_period}_{smoothing_type}_{ema_period}'
        
        if column_name not in data.columns:
            data = self.calculate(data, **kwargs)

        if column_name not in data.columns:
            raise ValueError(f"ATR column {column_name} not found after calculation")

        # Create subplot with price on top, ATR on bottom
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.05,
            subplot_titles=("Price", f"ATR({atr_period}) - {smoothing_type.upper()}({ema_period})"),
            row_heights=[0.7, 0.3]
        )

        # Price chart
        fig.add_trace(
            go.Candlestick(
                x=data['open_time'] if 'open_time' in data.columns else data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='Price'
            ),
            row=1, col=1
        )

        # ATR line
        fig.add_trace(
            go.Scatter(
                x=data['open_time'] if 'open_time' in data.columns else data.index,
                y=data[column_name],
                mode='lines',
                line=dict(color='orange', width=2),
                name=f"ATR({atr_period}) {smoothing_type.upper()}({ema_period})"
            ),
            row=2, col=1
        )

        # Optional: Add raw True Range for comparison
        tr_column = f'tr_{atr_period}'
        if tr_column in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data['open_time'] if 'open_time' in data.columns else data.index,
                    y=data[tr_column],
                    mode='lines',
                    line=dict(color='rgba(128,128,128,0.5)', width=1),
                    name=f"True Range({atr_period})"
                ),
                row=2, col=1
            )

        fig.update_layout(
            title=f"{self.name} - ATR Period: {atr_period}, {smoothing_type.upper()} Period: {ema_period}",
            height=800,
            xaxis_rangeslider_visible=False,
            template='plotly_dark'
        )

        # Update y-axis labels
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="ATR Value", row=2, col=1)

        return fig

    def get_parameter_schema(self):
        """
        Returns the schema for the parameters of this indicator.
        This is used for UI generation and validation.
        """
        return {
            'name': self.name,
            'parameters': self._get_default_params(),
            'category': getattr(self, 'category')
        }

    def _get_default_params(self):
        return {
            'atr_period': {
                'type': 'int', 
                'default': 14, 
                'min': 1, 
                'max': 100, 
                'description': 'ATR Calculation Period'
            },
            'ema_period': {
                'type': 'int', 
                'default': 14, 
                'min': 1, 
                'max': 100, 
                'description': 'EMA Smoothing Period'
            },
            'smoothing_type': {
                'type': 'select', 
                'default': 'ema', 
                'options': ['ema', 'sma', 'rma'],
                'description': 'Smoothing Method'
            }
        }

    def _get_plot_trace(self, data, **kwargs):
        """Return plotly trace for ATR (for overlay plotting)"""
        atr_period = kwargs.get('atr_period', self.atr_period)
        ema_period = kwargs.get('ema_period', self.ema_period)
        smoothing_type = kwargs.get('smoothing_type', self.smoothing_type)
        column_name = f'atr_{atr_period}_{smoothing_type}_{ema_period}'

        # Ensure the column exists
        if column_name not in data.columns:
            data = self.calculate(data, **kwargs)

        if column_name not in data.columns:
            raise ValueError(f"Column {column_name} not found after calculation")

        # Clean data
        clean_data = data[[column_name]].dropna()
        
        if len(clean_data) > len(data):
            clean_data = clean_data.tail(len(data))

        timestamps = clean_data.index.tolist()
        values = clean_data[column_name].tolist()

        return {
            'data': [{
                'x': timestamps,
                'y': values,
                'type': 'scatter',
                'mode': 'lines',
                'name': f'ATR({atr_period}) {smoothing_type.upper()}({ema_period})',
                'line': {
                    'color': '#ffa500',
                    'width': 2
                },
                'yaxis': 'y2'  # Use secondary y-axis for ATR
            }],
            'layout_update': {
                'yaxis2': {
                    'title': 'ATR Value',
                    'overlaying': 'y',
                    'side': 'right'
                }
            }
        }
    
    @staticmethod
    def get_volatility_percentile(data, lookback_period=252, **kwargs):
        """
        Calculate ATR percentile rank over a lookback period.
        Useful for identifying high/low volatility regimes.
        
        :param data: DataFrame with calculated ATR
        :param lookback_period: Period for percentile calculation
        :return: DataFrame with ATR percentile rank added
        """
        atr_period = kwargs.get('atr_period', 14)
        ema_period = kwargs.get('ema_period', 14)
        smoothing_type = kwargs.get('smoothing_type', 'ema')
        atr_column = f'atr_{atr_period}_{smoothing_type}_{ema_period}'
        
        if atr_column not in data.columns:
            data = AverageTrueRange.calculate(data, **kwargs)
        
        # Calculate rolling percentile rank
        data[f'atr_percentile_{atr_period}_{ema_period}'] = (
            data[atr_column].rolling(window=lookback_period)
            .rank(pct=True) * 100
        )
        
        return data