from plotly.subplots import make_subplots

from app.indicators.BaseIndicatorInterface import BaseIndicatorInterface
import plotly.graph_objs as go

class RSI(BaseIndicatorInterface):
    """
    Relative Strength Index (RSI) indicator.
    """

    def __init__(self, name: str = "RSI", period: int = 14, **kwargs):
        super().__init__(name)
        self.name = name
        self.period = period
        self.input_columns = kwargs.get('input_columns', ['close'])
        self.source = kwargs.get('source', 'close')
        self.overbought = kwargs.get('overbought', 70)
        self.oversold = kwargs.get('oversold', 30)

    def calculate(self, data, **kwargs):
        """
        Calculate Relative Strength Index on the specified data source.

        :param data: DataFrame containing price data
        :param kwargs: Optional arguments to override class settings
        :return: DataFrame with RSI values added
        """
        source = kwargs.get('source', self.source)
        period = kwargs.get('period', self.period)

        # Create the source series based on the input type
        if source == 'hlc3':
            source_data = (data['high'] + data['low'] + data['close']) / 3
        elif source == 'hl2':
            source_data = (data['high'] + data['low']) / 2
        elif source == 'ohlc4':
            source_data = (data['open'] + data['high'] + data['low'] + data['close']) / 4
        elif source in data.columns:
            source_data = data[source]
        else:
            raise ValueError(f"Invalid source '{source}'. Must be a column name or a valid combination.")

        # Calculate price changes
        delta = source_data.diff()

        # Create gains (positive) and losses (negative) Series
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Calculate average gains and losses over the period
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # Add to dataframe
        column_name = f"rsi_{period}_{source}"
        data[column_name] = rsi

        return data

    def plot(self, data, **kwargs):
        """
        Plot the RSI indicator with Plotly.

        :param data: DataFrame containing price data and calculated RSI
        :param kwargs: Optional parameters for plotting
        :return: Plotly figure object
        """
        period = self.period
        source = kwargs.get('source', self.source)
        overbought = kwargs.get('overbought', self.overbought)
        oversold = kwargs.get('oversold', self.oversold)

        column_name = f"rsi_{period}_{source}"

        if column_name not in data.columns:
            raise ValueError(f"RSI column '{column_name}' not found in data")

        # Create subplot with price chart and RSI indicator
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.3,
            row_heights=[0.7, 0.3],
            subplot_titles=["Price", f"RSI ({period})"]
        )

        # Determine x-axis data
        x_axis = 'open_time' if 'open_time' in data.columns else data.index

        # Add price candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=x_axis,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='Price'
            ),
            row=1, col=1
        )

        # Add RSI line
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=data[column_name],
                line=dict(color='purple', width=1.5),
                name=f"RSI ({period})"
            ),
            row=2, col=1
        )

        # Add overbought line
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=[overbought] * len(data),
                line=dict(color='red', width=1, dash='dash'),
                name=f"Overbought ({overbought})"
            ),
            row=2, col=1
        )

        # Add oversold line
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=[oversold] * len(data),
                line=dict(color='green', width=1, dash='dash'),
                name=f"Oversold ({oversold})"
            ),
            row=2, col=1
        )

        # Add 50 centerline
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=[50] * len(data),
                line=dict(color='gray', width=1, dash='dot'),
                name="Centerline (50)"
            ),
            row=2, col=1
        )

        # Update layout
        fig.update_layout(
            title=f"{self.name} ({period}) Analysis",
            xaxis_rangeslider_visible=True,
            height=800,
            template="plotly_dark",
            legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center")
        )

        # Update y-axis ranges
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)

        return fig