from typing import Optional

from plotly.subplots import make_subplots

from app.indicators.simple_moving_average import SimpleMovingAverage
from app.strategy.base_strategy_interface import BaseStrategyInterface
import pandas as pd
import plotly.graph_objs as go


class SmaConfluence(BaseStrategyInterface):
    """
    A strategy that uses the confluence of two Simple Moving Averages (SMA) to make trading decisions.
    """

    def __init__(self, short_window: int = 20, long_window: int = 50):
        """
        Initialize the strategy with short and long SMA windows.

        :param short_window: The period for the short SMA.
        :param long_window: The period for the long SMA.
        """
        super().__init__("SMA Confluence")
        self.fast_ma = None
        self.slow_ma = None
        self.data = None
        self.signals = None
        self.fast_ma_column = None
        self.slow_ma_column = None

    def initialize(self, **kwargs):
        """
        Initialize the strategy with the provided data and parameters.
        """
        self.parameters = {
            'fast_period': kwargs.get('fast_period', 10),
            'slow_period': kwargs.get('slow_period', 30),
            'source': kwargs.get('source', 'close'),
            'signal_threshold': kwargs.get('signal_threshold', 0)
        }

        # Initialize MA indicators
        self.fast_ma = SimpleMovingAverage(
            name=f"FastMA_{self.parameters['fast_period']}",
            period=self.parameters['fast_period'],
            source=self.parameters['source']
        )

        self.slow_ma = SimpleMovingAverage(
            name=f"SlowMA_{self.parameters['slow_period']}",
            period=self.parameters['slow_period'],
            source=self.parameters['source']
        )

        # Define column names for the indicators
        self.fast_ma_column = f"sma_{self.parameters['fast_period']}_{self.parameters['source']}"
        self.slow_ma_column = f"sma_{self.parameters['slow_period']}_{self.parameters['source']}"

    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Execute the strategy on the provided data.

        :param data: Price data as pandas DataFrame
        :return: DataFrame with strategy calculations and signals
        """
        # Store a copy of the data
        self.data = data.copy()

        # Calculate moving averages
        self.data = self.fast_ma.calculate(self.data, period=self.parameters['fast_period'],
                                           source=self.parameters['source'])
        self.data = self.slow_ma.calculate(self.data, period=self.parameters['slow_period'],
                                           source=self.parameters['source'])

        # Generate signals
        self.data = self.generate_signals(self.data)

        self.signals = self.data

        return self.data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy/sell signals based on moving average crossovers.

        :param data: DataFrame with calculated indicators
        :return: DataFrame with added signal columns
        """
        df = data.copy()

        # Calculate the difference between fast and slow MAs
        df['ma_diff'] = df[self.fast_ma_column] - df[self.slow_ma_column]

        # Initialize signal column
        df['signal'] = 0

        # Generate crossover signals
        # Bullish crossover: fast MA crosses above slow MA
        bullish = (df['ma_diff'] > self.parameters['signal_threshold']) & (
                    df['ma_diff'].shift(1) <= self.parameters['signal_threshold'])

        # Bearish crossover: fast MA crosses below slow MA
        bearish = (df['ma_diff'] < self.parameters['signal_threshold']) & (
                    df['ma_diff'].shift(1) >= self.parameters['signal_threshold'])

        # Assign signals
        df.loc[bullish, 'signal'] = 1  # Buy signal
        df.loc[bearish, 'signal'] = -1  # Sell signal

        # Add position column for backtesting
        df['position'] = df['signal'].replace(to_replace=0, method='ffill')
        df['position'] = df['position'].fillna(0)

        return df

    def plot(self, data: Optional[pd.DataFrame] = None, **kwargs):
        """
        Plot the strategy signals and indicators.

        :param data: Optional data to plot (uses self.data if None)
        :param kwargs: Additional plotting parameters
        :return: Plotly figure object
        """
        plot_data = data if data is not None else self.data

        if plot_data is None:
            raise ValueError("No data available for plotting. Execute the strategy first.")

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],
            subplot_titles=[f"{self.strategy_name} - Price and Moving Averages", "Signals"]
        )

        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=plot_data['open_time'] if 'open_time' in plot_data.columns else plot_data.index,
                open=plot_data['open'],
                high=plot_data['high'],
                low=plot_data['low'],
                close=plot_data['close'],
                name='Price'
            ),
            row=1, col=1
        )

        # Add moving averages
        fig.add_trace(
            go.Scatter(
                x=plot_data['open_time'] if 'open_time' in plot_data.columns else plot_data.index,
                y=plot_data[self.fast_ma_column],
                mode='lines',
                line=dict(color='blue', width=1.5),
                name=f"Fast MA ({self.parameters['fast_period']})"
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=plot_data['open_time'] if 'open_time' in plot_data.columns else plot_data.index,
                y=plot_data[self.slow_ma_column],
                mode='lines',
                line=dict(color='red', width=1.5),
                name=f"Slow MA ({self.parameters['slow_period']})"
            ),
            row=1, col=1
        )

        # Add buy signals
        buy_signals = plot_data[plot_data['signal'] == 1]
        fig.add_trace(
            go.Scatter(
                x=buy_signals['open_time'] if 'open_time' in plot_data.columns else buy_signals.index,
                y=buy_signals['low'] * 0.99,  # Place below the candle
                mode='markers',
                marker=dict(color='green', size=10, symbol='triangle-up'),
                name='Buy Signal'
            ),
            row=1, col=1
        )

        # Add sell signals
        sell_signals = plot_data[plot_data['signal'] == -1]
        fig.add_trace(
            go.Scatter(
                x=sell_signals['open_time'] if 'open_time' in plot_data.columns else sell_signals.index,
                y=sell_signals['high'] * 1.01,  # Place above the candle
                mode='markers',
                marker=dict(color='red', size=10, symbol='triangle-down'),
                name='Sell Signal'
            ),
            row=1, col=1
        )

        # Add MA difference chart
        fig.add_trace(
            go.Scatter(
                x=plot_data['open_time'] if 'open_time' in plot_data.columns else plot_data.index,
                y=plot_data['ma_diff'],
                mode='lines',
                line=dict(color='purple', width=1),
                name='MA Difference'
            ),
            row=2, col=1
        )

        # Add zero line
        fig.add_trace(
            go.Scatter(
                x=plot_data['open_time'] if 'open_time' in plot_data.columns else plot_data.index,
                y=[0] * len(plot_data),
                mode='lines',
                line=dict(color='black', width=0.5, dash='dash'),
                name='Zero Line'
            ),
            row=2, col=1
        )

        # Update layout
        fig.update_layout(
            title=f"{self.strategy_name} - Moving Average Crossover Strategy",
            xaxis_rangeslider_visible=False,
            height=800,
            template='plotly_white'
        )

        # Update y-axis labels
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="MA Diff", row=2, col=1)

        return fig
