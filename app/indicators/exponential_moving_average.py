from plotly.subplots import make_subplots

from app.indicators.BaseIndicatorInterface import BaseIndicatorInterface
import plotly.graph_objs as go

from app.utils.price_utils import get_price_source_data


class ExponentialMovingAverage(BaseIndicatorInterface):
    """Exponential Moving Average (EMA) indicator."""

    def __init__(self, name: str = "ExponentialMovingAverage", period: int = 20, **kwargs):
        super().__init__(name)
        self.name = name
        self.period = period
        self.input_columns = kwargs.get('input_columns', ['close'])
        self.source = kwargs.get('source', 'close')
        self.category = 'Moving Average'

    def calculate(self, data, **kwargs):
        """
        Calculate Exponential Moving Average on the specified data source.

        Supports various price inputs:
        - single column: 'close', 'open', 'high', 'low', etc.
        - combinations: 'hlc3' (high+low+close)/3, 'hl2' (high+low)/2, etc.

        :param data: DataFrame containing price data
        :param kwargs: Optional arguments to override class settings
        :return: DataFrame with EMA values added
        """
        source = kwargs.get('source', self.source)
        period = kwargs.get('period', self.period)

        # Create the source series based on the input type
        source_data = get_price_source_data(data, source)

        # Calculate EMA
        column_name = f'ema_{period}_{source}'
        data[column_name] = source_data.ewm(span=period, adjust=False).mean()

        return data

    def plot(self, data, **kwargs):
        """
        Plot the Exponential Moving Average on the provided data.

        :param data: DataFrame containing price data and calculated EMA
        :param kwargs: Optional parameters for plotting
        """

        if (data is None) or (kwargs.get("column_name", 'ema_20_close') not in data.columns):
            raise ValueError("Data is required for plotting")

        column_name = kwargs.get("column_name", 'ema_20_close')

        fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.02, subplot_titles=f"{self.name} Plot")

        fig.add_trace(
            go.Candlestick(
                x=data['open_time'] if 'open_time' in data.columns else data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='Candlestick'
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=data['open_time'] if 'open_time' in data.columns else data.index,
                y=data[column_name],
                mode='lines',
                line=dict(color='blue', width=1),
                name=f"EMA {self.period} ({self.source})"
            ),
            row=1, col=1
        )

        fig.update_layout(
            title=f"{self.name}",
            xaxis_title="Timestamp",
            yaxis_title="Price",
            height=600,
            xaxis_rangeslider_visible=True,
            template='plotly_dark'
        )

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
            'period': {'type': 'int', 'default': 20, 'min': 1, 'max': 100, 'description': 'EMA Length'},
            'source': {'type': 'select', 'default': 'close', 'options': ['close', 'hlc3', 'hl2', 'ohlc4'],
                       'description': 'Source'}
        }

    def _get_plot_trace(self, data, **kwargs):
        """Return plotly trace for EMA"""
        period = kwargs.get('period', self.period)
        source = kwargs.get('source', self.source)
        column_name = f'ema_{period}_{source}'

        # Ensure the column exists
        if column_name not in data.columns:
            # Calculate if not present
            data = self.calculate(data, **kwargs)

        if column_name not in data.columns:
            raise ValueError(f"Column {column_name} not found after calculation")

        # Clean data and ensure alignment with original timeframe
        clean_data = data[[column_name]].dropna()

        # Ensure we don't exceed the original data bounds
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
                'name': f'EMA({period})',
                'line': {
                    'color': '#4ecdc4',
                    'width': 2
                },
                'yaxis': 'y'
            }],
            'layout_update': {}
        }
