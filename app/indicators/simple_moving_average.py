from plotly.subplots import make_subplots

from app.indicators.BaseIndicatorInterface import BaseIndicatorInterface
import plotly.graph_objs as go


class SimpleMovingAverage(BaseIndicatorInterface):
    """Simple Moving Average (SMA) indicator."""

    def __init__(self, name: str = "SimpleMovingAverage", period: int = 20, **kwargs):
        super().__init__(name)
        self.name = name
        self.period = period
        self.input_columns = kwargs.get('input_columns', ['close'])
        self.source = kwargs.get('source', 'close')

    def calculate(self, data, **kwargs):
        """
        Calculate Simple Moving Average on the specified data source.

        Supports various price inputs:
        - single column: 'close', 'open', 'high', 'low', etc.
        - combinations: 'hlc3' (high+low+close)/3, 'hl2' (high+low)/2, etc.

        :param data: DataFrame containing price data
        :param kwargs: Optional arguments to override class settings
        :return: DataFrame with SMA values added
        """
        source = kwargs.get('source', self.source)
        period = kwargs.get('period', self.period)

        # Create the source series based on the input type
        if source == 'hlcc4':
            source_data = data['high'] + data['low'] + data['close'] + data['close'] / 4
        elif source == 'hlc3':
            source_data = (data['high'] + data['low'] + data['close']) / 3
        elif source == 'hl2':
            source_data = (data['high'] + data['low']) / 2
        elif source == 'ohlc4':
            source_data = (data['open'] + data['high'] + data['low'] + data['close']) / 4
        elif source in data.columns:
            source_data = data[source]
        else:
            raise ValueError(f"Invalid source '{source}'. Must be a column name or a valid combination.")

        # Calculate SMA
        column_name = f'sma_{period}_{source}'
        data[column_name] = source_data.rolling(window=period).mean()

        return data

    def plot(self, data, **kwargs):
        """
        Plot the Simple Moving Average on the provided data.

        :param data: DataFrame containing price data and calculated SMA
        :param kwargs: Optional parameters for plotting
        """

        if (data is None) or (kwargs.get("column_name", 'sma_20_close') not in data.columns):
            raise ValueError("Data is required for plotting")

        column_name = kwargs.get("column_name", 'sma_20_close')

        fig = make_subplots(rows= 1, cols=1, shared_xaxes=True, vertical_spacing=0.02, subplot_titles=f"{self.name} Plot")

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
                line=dict(color='red', width=1),
                name=f"SMA {self.period} ({self.source})"
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