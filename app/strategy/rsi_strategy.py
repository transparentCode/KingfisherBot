from typing import Optional

from app.indicators.rsi import RSI
from app.strategy.base_strategy_interface import BaseStrategyInterface
import plotly.graph_objs as go
import pandas as pd

class RSIStrategy(BaseStrategyInterface):
    """
    A strategy that generates signals based on RSI overbought/oversold levels.
    """

    def __init__(self):
        super().__init__("RSI Strategy")
        self.rsi_indicator = None
        self.data = None
        self.signals = None
        self.rsi_column = None

    def initialize(self, **kwargs):
        """
        Initialize the strategy with the provided parameters.
        """
        self.parameters = {
            'period': kwargs.get('period', 14),
            'source': kwargs.get('source', 'close'),
            'overbought': kwargs.get('overbought', 70),
            'oversold': kwargs.get('oversold', 30),
            'exit_overbought': kwargs.get('exit_overbought', 50),
            'exit_oversold': kwargs.get('exit_oversold', 50)
        }

        # Initialize RSI indicator
        self.rsi_indicator = RSI(
            period=self.parameters['period'],
            source=self.parameters['source'],
            overbought=self.parameters['overbought'],
            oversold=self.parameters['oversold']
        )

        # Define column name for the RSI
        self.rsi_column = f"rsi_{self.parameters['period']}_{self.parameters['source']}"

    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Execute the strategy on the provided data.

        :param data: Price data as pandas DataFrame
        :return: DataFrame with strategy calculations and signals
        """
        # Store a copy of the data
        self.data = data.copy()

        # Calculate RSI
        self.data = self.rsi_indicator.calculate(
            self.data,
            period=self.parameters['period'],
            source=self.parameters['source']
        )

        # Generate signals
        self.data = self.generate_signals(self.data)

        self.signals = self.data

        return self.data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy/sell signals based on RSI levels.

        :param data: DataFrame with calculated RSI
        :return: DataFrame with added signal columns
        """
        df = data.copy()

        # Initialize signal column
        df['signal'] = 0

        # Buy signal: RSI crosses below oversold level
        buy_signal = (df[self.rsi_column] < self.parameters['oversold']) & \
                     (df[self.rsi_column].shift(1) >= self.parameters['oversold'])

        # Exit long: RSI crosses above exit_oversold level
        exit_long = (df[self.rsi_column] > self.parameters['exit_oversold']) & \
                    (df[self.rsi_column].shift(1) <= self.parameters['exit_oversold'])

        # Sell signal: RSI crosses above overbought level
        sell_signal = (df[self.rsi_column] > self.parameters['overbought']) & \
                      (df[self.rsi_column].shift(1) <= self.parameters['overbought'])

        # Exit short: RSI crosses below exit_overbought level
        exit_short = (df[self.rsi_column] < self.parameters['exit_overbought']) & \
                     (df[self.rsi_column].shift(1) >= self.parameters['exit_overbought'])

        # Assign signals
        df.loc[buy_signal, 'signal'] = 1  # Buy signal
        df.loc[sell_signal, 'signal'] = -1  # Sell signal
        df.loc[exit_long, 'signal'] = -1  # Exit long position
        df.loc[exit_short, 'signal'] = 1  # Exit short position

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

        # Use the RSI indicator's plot method and add signal markers
        fig = self.rsi_indicator.plot(plot_data, source=self.parameters['source'])

        # Add buy signals
        buy_signals = plot_data[plot_data['signal'] == 1]
        if not buy_signals.empty:
            x_axis = buy_signals['open_time'] if 'open_time' in plot_data.columns else buy_signals.index
            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=buy_signals['low'] * 0.99,  # Place below the candle
                    mode='markers',
                    marker=dict(color='green', size=10, symbol='triangle-up'),
                    name='Buy Signal'
                ),
                row=1, col=1
            )

        # Add sell signals
        sell_signals = plot_data[plot_data['signal'] == -1]
        if not sell_signals.empty:
            x_axis = sell_signals['open_time'] if 'open_time' in plot_data.columns else sell_signals.index
            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=sell_signals['high'] * 1.01,  # Place above the candle
                    mode='markers',
                    marker=dict(color='red', size=10, symbol='triangle-down'),
                    name='Sell Signal'
                ),
                row=1, col=1
            )

        fig.update_layout(title=f"{self.strategy_name} - RSI Strategy Analysis")

        return fig