import pandas as pd
import numpy as np
import plotly.graph_objects as go
from numba import njit
from plotly.subplots import make_subplots
from app.indicators.BaseIndicatorInterface import BaseIndicatorInterface
import datetime


@njit
def _calculate_qrsi_numba(src: np.ndarray, length: int) -> np.ndarray:
    """
    Calculates the Quantum RSI using a Numba-accelerated loop.
    """

    n = len(src)
    qrsi_out = np.full(n, np.nan)
    weights = np.array([np.exp(-np.power((i + 1) / length, 2)) for i in range(length)])

    for i in range(length + 1, n):
        g = 0.0
        l = 0.0
        for j in range(length):
            # Corresponds to Pine Script's `src[j] - src[j+1]` in its loop
            diff = src[i - (j + 1)] - src[i - (j + 2)]
            weight = weights[j]
            if diff > 0:
                g += diff * weight
            else:
                l += -diff * weight

        net_momentum = g - l
        total_energy = g + l
        wave_ratio = net_momentum / total_energy if total_energy != 0 else 0.0
        qrsi_out[i] = 50.0 + 50.0 * wave_ratio

    return qrsi_out


class RSI(BaseIndicatorInterface):
    """
    Gaussian Weighted Relative Strength Index (RSI) indicator.

    This implementation is based on the "Quantum RSI" which uses a Gaussian
    decay function to weight price momentum. The calculation is performed
    on a DEMA of the selected source.
    """

    def __init__(self, name: str = "RSI", **kwargs):
        super().__init__(name, **kwargs)
        self.length = kwargs.get('length', 14)
        self.ma_length = kwargs.get('ma_length', 5)
        self.ma_type = kwargs.get('ma_type', "EMA")
        self.dema_length = kwargs.get('dema_length', 14)
        self.source = kwargs.get('source', 'close')
        self.overbought = kwargs.get('overbought', 70)
        self.oversold = kwargs.get('oversold', 30)
        self.input_columns = ['open', 'high', 'low', 'close']
        self.category = 'Oscillator'

    def _get_source_data(self, data: pd.DataFrame, source: str) -> pd.Series:
        """Selects the source data series from the dataframe."""
        if source == 'hlc3':
            return (data['high'] + data['low'] + data['close']) / 3
        elif source == 'hl2':
            return (data['high'] + data['low']) / 2
        elif source == 'ohlc4':
            return (data['open'] + data['high'] + data['low'] + data['close']) / 4
        elif source in data.columns:
            return data[source]
        else:
            raise ValueError(f"Invalid source '{source}'.")

    def _calculate_ma(self, source: pd.Series, length: int, ma_type: str) -> pd.Series:
        """Calculates a moving average of a given type."""
        if ma_type == "SMA":
            return source.rolling(window=length, min_periods=length).mean()
        elif ma_type == "EMA":
            return source.ewm(span=length, adjust=False).mean()
        elif ma_type == "WMA":
            weights = np.arange(1, length + 1)
            return source.rolling(window=length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
        elif ma_type == "SMMA (RMA)":
            return source.ewm(alpha=1 / length, adjust=False).mean()
        elif ma_type == "VWMA":
            raise NotImplementedError("VWMA requires a 'volume' column, which is not provided.")
        else:
            raise ValueError(f"Unknown MA type: {ma_type}")

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculates the Quantum RSI and its moving average.
        """
        df = data.copy()
        source_str = kwargs.get('source', self.source)
        dema_len = kwargs.get('dema_length', self.dema_length)
        rsi_len = kwargs.get('length', self.length)
        ma_len = kwargs.get('ma_length', self.ma_length)
        ma_t = kwargs.get('ma_type', self.ma_type)

        # 1. Get base source and calculate DEMA
        source_data = self._get_source_data(df, source_str)
        ema1 = source_data.ewm(span=dema_len, adjust=False).mean()
        ema2 = ema1.ewm(span=dema_len, adjust=False).mean()
        dema_src = 2 * ema1 - ema2

        # 2. Calculate Quantum RSI
        qrsi_raw = _calculate_qrsi_numba(dema_src.to_numpy(), rsi_len)
        df['qrsi'] = pd.Series(qrsi_raw, index=df.index)

        # 3. Smooth RSI and calculate its MA
        df['qrsi_smoothed'] = df['qrsi'].ewm(span=2, adjust=False).mean()
        df['qrsi_ma'] = self._calculate_ma(df['qrsi_smoothed'], ma_len, ma_t)

        return df

    def plot(self, data: pd.DataFrame, **kwargs) -> go.Figure:
        """
        Plots the Quantum RSI indicator with Plotly.
        """
        if 'qrsi_smoothed' not in data.columns or 'qrsi_ma' not in data.columns:
            raise ValueError("Quantum RSI columns not found. Please run calculate() first.")

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])

        # Add Candlestick chart
        fig.add_trace(go.Candlestick(
            x=data.index, open=data['open'], high=data['high'], low=data['low'], close=data['close'], name='Price'
        ), row=1, col=1)

        # Add RSI lines
        fig.add_trace(go.Scatter(
            x=data.index, y=data['qrsi_smoothed'], name='Quantum RSI', line=dict(color='#D40CC2', width=1.5)
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=data.index, y=data['qrsi_ma'], name=f'RSI MA({self.ma_length})', line=dict(color='#3AFFa3', width=1.5)
        ), row=2, col=1)

        # Add level lines
        fig.add_hline(y=self.overbought, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=self.oversold, line_dash="dash", line_color="green", row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="grey", row=2, col=1)

        # Add fills for overbought and oversold zones
        y_over = np.where(data['qrsi_smoothed'] >= self.overbought, data['qrsi_smoothed'], self.overbought)
        y_under = np.where(data['qrsi_smoothed'] <= self.oversold, data['qrsi_smoothed'], self.oversold)

        fig.add_trace(go.Scatter(
            x=data.index, y=y_over,
            fill='tonexty', fillcolor='rgba(255, 0, 0, 0.2)',
            line=dict(width=0), showlegend=False
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=data.index, y=y_under,
            fill='tonexty', fillcolor='rgba(0, 255, 0, 0.2)',
            line=dict(width=0), showlegend=False
        ), row=2, col=1)

        fig.update_layout(
            title_text=f"{self.name} ({self.length}) Analysis",
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            height=800
        )
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)

        return fig

    def get_parameter_schema(self):
        """
        Returns the schema for the parameters of this indicator.
        This is used for UI generation and validation.
        """
        return {
            'name': self.name,
            'parameters': self._get_default_params(),
            'category': getattr(self, 'category', 'Oscillator')
        }

    def _get_default_params(self):
        return {
            'length': {'type': 'int', 'default': 14, 'min': 1, 'max': 100, 'description': 'RSI Period'},
            'ma_length': {'type': 'int', 'default': 5, 'min': 1, 'max': 100, 'description': 'MA Length'},
            'ma_type': {'type': 'select', 'default': 'EMA', 'options': ['SMA', 'EMA', 'WMA', 'SMMA (RMA)'],
                        'description': 'MA Type'},
            'dema_length': {'type': 'int', 'default': 14, 'min': 1, 'max': 100, 'description': 'DEMA Length'},
            'source': {'type': 'select', 'default': 'close', 'options': ['close', 'hlc3', 'hl2', 'ohlc4'],
                       'description': 'Source'},
            'overbought': {'type': 'float', 'default': 70, 'min': 50, 'max': 100, 'description': 'Overbought Level'},
            'oversold': {'type': 'float', 'default': 30, 'min': 0, 'max': 50, 'description': 'Oversold Level'}
        }

    def _get_plot_trace(self, data, **kwargs):
        """Generate plot trace for RSI indicator"""
        if data.empty:
            return []

        # Check which RSI columns are available
        available_columns = []
        if 'qrsi_smoothed' in data.columns:
            available_columns.append('qrsi_smoothed')
        if 'qrsi_ma' in data.columns:
            available_columns.append('qrsi_ma')
        if 'qrsi' in data.columns and 'qrsi_smoothed' not in data.columns:
            available_columns.append('qrsi')

        if not available_columns:
            return []

        # Use the base class method to clean data for JSON serialization
        clean_data = self._clean_data_for_json(data, available_columns)

        if not clean_data or 'timestamps' not in clean_data:
            return []

        timestamps = clean_data['timestamps']
        traces = []

        # Main RSI line (qrsi_smoothed or qrsi)
        main_column = 'qrsi_smoothed' if 'qrsi_smoothed' in clean_data else 'qrsi'
        if main_column in clean_data:
            traces.append({
                'x': timestamps,
                'y': clean_data[main_column],
                'type': 'scatter',
                'mode': 'lines',
                'name': f'RSI({self.length})',
                'line': {
                    'color': '#D40CC2',
                    'width': 2
                },
                'showlegend': True
                # 'yaxis' property removed
            })

        # Add RSI MA line if available
        if 'qrsi_ma' in clean_data:
            traces.append({
                'x': timestamps,
                'y': clean_data['qrsi_ma'],
                'type': 'scatter',
                'mode': 'lines',
                'name': f'RSI MA({self.ma_length})',
                'line': {
                    'color': '#3AFFa3',
                    'width': 1.5
                },
                'showlegend': True
                # 'yaxis' property removed
            })

        # The reference lines for overbought/oversold are better handled
        # by the frontend's getSubplotAxisConfig function using layout.shapes,
        # but if you must send them as traces, remove the yaxis property here too.
        # It's recommended to remove these from here to avoid clutter.

        return traces

