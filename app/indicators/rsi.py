import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from app.indicators.BaseIndicatorInterface import BaseIndicatorInterface
from app.utils.price_utils import get_price_source_data


class RSI(BaseIndicatorInterface):
    """
    Relative Strength Index (RSI) indicator.
    """

    def __init__(self, name: str = "RSI", **kwargs):
        super().__init__(name, **kwargs)
        self.length = kwargs.get('length', 14)
        self.source = kwargs.get('source', 'close')
        self.overbought = kwargs.get('overbought', 70)
        self.oversold = kwargs.get('oversold', 30)
        self.input_columns = ['open', 'high', 'low', 'close']
        self.category = 'Oscillator'

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Calculate traditional RSI."""
        df = data.copy()
        source = kwargs.get('source', self.source)
        length = kwargs.get('length', self.length)

        # Get source data using utility function
        source_data = get_price_source_data(df, source)

        # Calculate price changes
        delta = source_data.diff()

        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)

        # Calculate average gains and losses using RMA (Wilder's smoothing)
        avg_gains = gains.ewm(alpha=1 / length, adjust=False).mean()
        avg_losses = losses.ewm(alpha=1 / length, adjust=False).mean()

        # Calculate RS and RSI
        rs = np.where(avg_losses == 0, np.nan, avg_gains / avg_losses)
        rsi = 100 - (100 / (1 + rs))

        # Store result
        column_name = f'rsi_{length}_{source}'
        df[column_name] = rsi

        return df

    def _find_pivots(self, data, lookback_left=5, lookback_right=5):
        """Find pivot highs and lows in the data."""
        data_array = np.array(data)
        n = len(data_array)

        pivot_highs = np.full(n, np.nan)
        pivot_lows = np.full(n, np.nan)

        for i in range(lookback_left, n - lookback_right):
            # Check for pivot high
            left_slice = data_array[i - lookback_left:i]
            right_slice = data_array[i + 1:i + lookback_right + 1]

            if (data_array[i] > np.max(left_slice)) and (data_array[i] > np.max(right_slice)):
                pivot_highs[i] = data_array[i]

            # Check for pivot low
            if (data_array[i] < np.min(left_slice)) and (data_array[i] < np.min(right_slice)):
                pivot_lows[i] = data_array[i]

        return pivot_highs, pivot_lows

    def _in_range(self, bars_since, range_lower=5, range_upper=60):
        """Check if bars since condition is within specified range."""
        return range_lower <= bars_since <= range_upper

    def detect_divergence(self, data, **kwargs):
        """
        Detect RSI divergences based on pivot points.
        """
        df = data.copy()
        length = kwargs.get('length', self.length)
        source = kwargs.get('source', self.source)
        lookback_left = kwargs.get('lookback_left', 5)
        lookback_right = kwargs.get('lookback_right', 5)
        range_upper = kwargs.get('range_upper', 60)
        range_lower = kwargs.get('range_lower', 5)

        rsi_column = f'rsi_{length}_{source}'

        # Ensure RSI is calculated
        if rsi_column not in df.columns:
            df = self.calculate(df, **kwargs)

        # Initialize divergence columns
        df['bullish_divergence'] = False
        df['bearish_divergence'] = False
        df['divergence_signal'] = 0

        # Get data arrays
        rsi_values = df[rsi_column].fillna(method='bfill').fillna(method='ffill').values
        high_values = df['high'].values
        low_values = df['low'].values

        # Find pivot points for RSI and price
        rsi_pivot_highs, rsi_pivot_lows = self._find_pivots(rsi_values, lookback_left, lookback_right)
        price_pivot_highs, _ = self._find_pivots(high_values, lookback_left, lookback_right)
        _, price_pivot_lows = self._find_pivots(low_values, lookback_left, lookback_right)

        # Get valid pivot indices
        rsi_high_indices = np.where(~np.isnan(rsi_pivot_highs))[0]
        rsi_low_indices = np.where(~np.isnan(rsi_pivot_lows))[0]
        price_high_indices = np.where(~np.isnan(price_pivot_highs))[0]
        price_low_indices = np.where(~np.isnan(price_pivot_lows))[0]

        # Detect bullish divergence (RSI higher low + Price lower low)
        for i in range(1, len(rsi_low_indices)):
            current_idx = rsi_low_indices[i]

            # Find corresponding previous pivot within range
            for j in range(i - 1, -1, -1):
                previous_idx = rsi_low_indices[j]
                bars_since = current_idx - previous_idx

                if self._in_range(bars_since, range_lower, range_upper):
                    # Check if we have corresponding price pivots
                    current_price_idx = None
                    previous_price_idx = None

                    # Find closest price pivot lows
                    for price_idx in price_low_indices:
                        if abs(price_idx - current_idx) <= lookback_right:
                            current_price_idx = price_idx
                            break

                    for price_idx in price_low_indices:
                        if abs(price_idx - previous_idx) <= lookback_right:
                            previous_price_idx = price_idx
                            break

                    if current_price_idx is not None and previous_price_idx is not None:
                        # RSI higher low AND price lower low
                        rsi_higher_low = rsi_values[current_idx] > rsi_values[previous_idx]
                        price_lower_low = low_values[current_price_idx] < low_values[previous_price_idx]

                        if rsi_higher_low and price_lower_low:
                            df.iloc[current_idx, df.columns.get_loc('bullish_divergence')] = True
                            df.iloc[current_idx, df.columns.get_loc('divergence_signal')] = 1
                            break

        # Detect bearish divergence (RSI lower high + Price higher high)
        for i in range(1, len(rsi_high_indices)):
            current_idx = rsi_high_indices[i]

            # Find corresponding previous pivot within range
            for j in range(i - 1, -1, -1):
                previous_idx = rsi_high_indices[j]
                bars_since = current_idx - previous_idx

                if self._in_range(bars_since, range_lower, range_upper):
                    # Check if we have corresponding price pivots
                    current_price_idx = None
                    previous_price_idx = None

                    # Find closest price pivot highs
                    for price_idx in price_high_indices:
                        if abs(price_idx - current_idx) <= lookback_right:
                            current_price_idx = price_idx
                            break

                    for price_idx in price_high_indices:
                        if abs(price_idx - previous_idx) <= lookback_right:
                            previous_price_idx = price_idx
                            break

                    if current_price_idx is not None and previous_price_idx is not None:
                        # RSI lower high AND price higher high
                        rsi_lower_high = rsi_values[current_idx] < rsi_values[previous_idx]
                        price_higher_high = high_values[current_price_idx] > high_values[previous_price_idx]

                        if rsi_lower_high and price_higher_high:
                            df.iloc[current_idx, df.columns.get_loc('bearish_divergence')] = True
                            df.iloc[current_idx, df.columns.get_loc('divergence_signal')] = -1
                            break

        return df

    def plot_with_divergence(self, data: pd.DataFrame, **kwargs) -> go.Figure:
        """Plot RSI with divergence signals."""
        # First detect divergences
        df = self.detect_divergence(data, **kwargs)

        length = kwargs.get('length', self.length)
        source = kwargs.get('source', self.source)
        column_name = f'rsi_{length}_{source}'

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                            row_heights=[0.7, 0.3])

        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=df['open_time'] if 'open_time' in df.columns else df.index,
            open=df['open'], high=df['high'], low=df['low'], close=df['close'],
            name='Price'
        ), row=1, col=1)

        # Add RSI line
        fig.add_trace(go.Scatter(
            x=df['open_time'] if 'open_time' in df.columns else df.index,
            y=df[column_name],
            name=f'RSI({length})',
            line=dict(color='#D40CC2', width=2)
        ), row=2, col=1)

        # Add divergence signals
        bullish_signals = df[df['bullish_divergence'] == True]
        bearish_signals = df[df['bearish_divergence'] == True]

        if not bullish_signals.empty:
            fig.add_trace(go.Scatter(
                x=bullish_signals['open_time'] if 'open_time' in df.columns else bullish_signals.index,
                y=bullish_signals[column_name],
                mode='markers',
                marker=dict(color='green', size=10, symbol='triangle-up'),
                name='Bullish Divergence'
            ), row=2, col=1)

        if not bearish_signals.empty:
            fig.add_trace(go.Scatter(
                x=bearish_signals['open_time'] if 'open_time' in df.columns else bearish_signals.index,
                y=bearish_signals[column_name],
                mode='markers',
                marker=dict(color='red', size=10, symbol='triangle-down'),
                name='Bearish Divergence'
            ), row=2, col=1)

        # Add level lines
        fig.add_hline(y=self.overbought, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=self.oversold, line_dash="dash", line_color="green", row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="grey", row=2, col=1)

        fig.update_layout(
            title_text=f"RSI({length}) with Divergence Analysis",
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            height=800
        )
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)

        return fig

    def plot(self, data: pd.DataFrame, **kwargs) -> go.Figure:
        """Plot RSI indicator."""
        length = kwargs.get('length', self.length)
        source = kwargs.get('source', self.source)
        column_name = f'rsi_{length}_{source}'

        if column_name not in data.columns:
            raise ValueError(f"RSI column {column_name} not found. Please run calculate() first.")

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                            row_heights=[0.7, 0.3])

        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=data['open_time'] if 'open_time' in data.columns else data.index,
            open=data['open'], high=data['high'], low=data['low'], close=data['close'],
            name='Price'
        ), row=1, col=1)

        # Add RSI line
        fig.add_trace(go.Scatter(
            x=data['open_time'] if 'open_time' in data.columns else data.index,
            y=data[column_name],
            name=f'RSI({length})',
            line=dict(color='#D40CC2', width=2)
        ), row=2, col=1)

        # Add level lines
        fig.add_hline(y=self.overbought, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=self.oversold, line_dash="dash", line_color="green", row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="grey", row=2, col=1)

        fig.update_layout(
            title_text=f"RSI({length}) Analysis",
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            height=800
        )
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)

        return fig

    def get_parameter_schema(self):
        """Returns parameter schema for UI generation."""
        return {
            'name': self.name,
            'parameters': self._get_default_params(),
            'category': self.category
        }

    def _get_default_params(self):
        return {
            'length': {'type': 'int', 'default': 14, 'min': 1, 'max': 100, 'description': 'RSI Period'},
            'source': {'type': 'select', 'default': 'close', 'options': ['close', 'hlc3', 'hl2', 'ohlc4'],
                       'description': 'Source'},
            'overbought': {'type': 'float', 'default': 70, 'min': 50, 'max': 100, 'description': 'Overbought Level'},
            'oversold': {'type': 'float', 'default': 30, 'min': 0, 'max': 50, 'description': 'Oversold Level'},
            'lookback_left': {'type': 'int', 'default': 5, 'min': 1, 'max': 20, 'description': 'Pivot Lookback Left'},
            'lookback_right': {'type': 'int', 'default': 5, 'min': 1, 'max': 20, 'description': 'Pivot Lookback Right'},
            'range_upper': {'type': 'int', 'default': 60, 'min': 10, 'max': 200,
                            'description': 'Max Bars Between Pivots'},
            'range_lower': {'type': 'int', 'default': 5, 'min': 1, 'max': 50, 'description': 'Min Bars Between Pivots'}
        }

    def _get_plot_trace(self, data, **kwargs):
        """Generate plot trace for RSI indicator."""
        length = kwargs.get('length', self.length)
        source = kwargs.get('source', self.source)
        column_name = f'rsi_{length}_{source}'

        if column_name not in data.columns:
            data = self.calculate(data, **kwargs)

        clean_data = data[[column_name]].dropna()

        return [{
            'x': clean_data.index.tolist(),
            'y': clean_data[column_name].tolist(),
            'type': 'scatter',
            'mode': 'lines',
            'name': f'RSI({length})',
            'line': {
                'color': '#D40CC2',
                'width': 2
            }
        }]
