import pandas as pd
import numpy as np
import plotly.graph_objects as go
from numba import njit, float64, boolean
from app.indicators.BaseIndicatorInterface import BaseIndicatorInterface


@njit
def _calculate_t_ema(close_prices, alpha_series):
    """Calculates the t-EMA using a Numba-accelerated loop."""
    t_ema = np.empty_like(close_prices)
    t_ema[:] = np.nan
    if len(close_prices) > 0:
        t_ema[0] = close_prices[0]
        for i in range(1, len(close_prices)):
            alpha = alpha_series[i]
            t_ema[i] = alpha * close_prices[i] + (1.0 - alpha) * t_ema[i - 1]
    return t_ema


@njit
def _calculate_supertrend(close, upper_band, lower_band):
    """Calculates the core SuperTrend line and direction."""
    st = np.full_like(close, np.nan)
    direction = np.full_like(close, True, dtype=boolean)

    if len(close) > 0:
        st[0] = lower_band[0] if close[0] > (upper_band[0] + lower_band[0]) / 2 else upper_band[0]

    for i in range(1, len(close)):
        if np.isnan(st[i - 1]):
            st[i - 1] = lower_band[i - 1]  # Fallback for initial NaN

        if close[i] > st[i - 1]:
            direction[i] = True
        elif close[i] < st[i - 1]:
            direction[i] = False
        else:
            direction[i] = direction[i - 1]

        st[i] = lower_band[i] if direction[i] else upper_band[i]

        if direction[i] and st[i] < st[i - 1]:
            st[i] = st[i - 1]
        elif not direction[i] and st[i] > st[i - 1]:
            st[i] = st[i - 1]

    return st, direction


class StudentTSuperTrend(BaseIndicatorInterface):
    """
    Student-t SuperTrend Indicator.

    This indicator calculates a SuperTrend using a Student's t-distribution
    to make its moving averages and ATR calculations adaptive to market volatility.
    """

    def __init__(self, name: str = "StudentTSuperTrend", **kwargs):
        super().__init__(name, **kwargs)
        self.atr_len = kwargs.get('atrLen', 10)
        self.atr_mult = kwargs.get('atrMult', 3.0)
        self.span = kwargs.get('span', 14)
        self.nu = kwargs.get('nu', 3.0)
        self.vol_window = kwargs.get('volWindow', 30)
        self.gamma = kwargs.get('gamma', 1.0)
        self.alpha_floor = kwargs.get('alphaFloor', 0.03)
        self.robust_atr = kwargs.get('robustATR_on', True)
        self.input_columns = ['high', 'low', 'close']
        self.category = 'Trend Indicators'

    def _calculate_adaptive_alpha(self, series: pd.Series, base_alpha: float) -> pd.Series:
        """Calculates the adaptive alpha based on Student-t distribution."""
        r = series.pct_change().fillna(0)
        sigma = r.rolling(window=self.vol_window).std().fillna(1e-10)
        alpha_raw = base_alpha / (1 + np.power(np.abs(r / sigma), self.nu))
        alpha_p = (self.gamma * alpha_raw).clip(lower=self.alpha_floor, upper=1.0)
        return alpha_p

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculates the Student-t SuperTrend values.

        :param data: DataFrame with high, low, close columns.
        :return: DataFrame with t-EMA, SuperTrend, and direction columns.
        """
        df = data.copy()

        # 1. t-EMA Calculation
        base_alpha_ema = 2.0 / (self.span + 1.0)
        alpha_p_ema = self._calculate_adaptive_alpha(df['close'], base_alpha_ema)
        df['t_ema'] = _calculate_t_ema(df['close'].to_numpy(), alpha_p_ema.to_numpy())

        # 2. ATR Calculation
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        if self.robust_atr:
            base_alpha_atr = 2.0 / (self.atr_len + 1.0)
            alpha_p_atr = self._calculate_adaptive_alpha(tr, base_alpha_atr)
            atr = _calculate_t_ema(tr.to_numpy(), alpha_p_atr.to_numpy())
        else:
            atr = tr.ewm(span=self.atr_len, adjust=False).mean().to_numpy()

        df['atr'] = atr

        # 3. SuperTrend Core Calculation
        hlcc4 = (df['high'] + df['low'] + df['close'] + df['close']) / 4
        upper_band = hlcc4 + self.atr_mult * df['atr']
        lower_band = hlcc4 - self.atr_mult * df['atr']

        st, direction = _calculate_supertrend(
            df['close'].to_numpy(),
            upper_band.to_numpy(),
            lower_band.to_numpy()
        )
        df['supertrend'] = st
        df['direction'] = direction

        return df

    def plot(self, data: pd.DataFrame, **kwargs) -> go.Figure:
        """
        Plots the Student-t SuperTrend indicator.

        :param data: DataFrame containing price data and calculated indicator values.
        :return: Plotly figure object.
        """
        if 'supertrend' not in data.columns:
            raise ValueError("SuperTrend column not found. Please run calculate() first.")

        fig = go.Figure()

        # Add Candlestick chart
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name='Price'
        ))

        # Add t-EMA
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['t_ema'],
            line=dict(color='orange', width=1.5),
            name=f't-EMA ({self.span})'
        ))

        # Create separate series for up and down trends for coloring
        st_up = data.where(data['direction'], np.nan)['supertrend']
        st_down = data.where(~data['direction'], np.nan)['supertrend']

        fig.add_trace(go.Scatter(
            x=data.index, y=st_up,
            line=dict(color='teal', width=2),
            name='Uptrend'
        ))
        fig.add_trace(go.Scatter(
            x=data.index, y=st_down,
            line=dict(color='red', width=2),
            name='Downtrend'
        ))

        # Add background color
        for i in range(1, len(data)):
            color = 'rgba(0, 128, 128, 0.1)' if data['direction'][i] else 'rgba(255, 0, 0, 0.1)'
            fig.add_vrect(
                x0=data.index[i - 1], x1=data.index[i],
                fillcolor=color,
                layer="below", line_width=0,
            )

        fig.update_layout(
            title=f"{self.name} Analysis",
            xaxis_rangeslider_visible=False,
            height=700,
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_yaxes(title_text="Price")

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
            'atr_len': {'type': 'int', 'default': 10, 'min': 2, 'max': 100, 'description': 'atr length'},
            'atr_mult': {'type': 'int', 'default': 3, 'min': 2, 'max': 100, 'description': 'atr multiplier'},
            'source': {'type': 'select', 'default': 'close', 'options': ['close', 'hlc3', 'hl2', 'ohlc4'],
                       'description': 'Source'},
            'span': {'type': 'int', 'default': '14', 'min': 1, 'max': 100, 'description': 'tema-0 length'}
        }

    def _get_plot_trace(self, data, **kwargs):
        """Return plotly trace for Supertrend"""
        import plotly.graph_objs as go

        period = kwargs.get('period', self.period)
        multiplier = kwargs.get('multiplier', self.multiplier)

        supertrend_col = f'supertrend_{period}_{multiplier}'
        trend_col = f'trend_{period}_{multiplier}'

        if supertrend_col not in data.columns or trend_col not in data.columns:
            raise ValueError(f"Supertrend columns not found in data")

        # Create traces for bullish and bearish trends
        bullish_data = data[data[trend_col] == 1]
        bearish_data = data[data[trend_col] == -1]

        traces = []

        if not bullish_data.empty:
            traces.append({
                'x': bullish_data.index.tolist(),
                'y': bullish_data[supertrend_col].tolist(),
                'type': 'scatter',
                'mode': 'lines',
                'name': 'Supertrend (Bullish)',
                'line': {
                    'color': '#00ff88',
                    'width': 2
                }
            })

        if not bearish_data.empty:
            traces.append({
                'x': bearish_data.index.tolist(),
                'y': bearish_data[supertrend_col].tolist(),
                'type': 'scatter',
                'mode': 'lines',
                'name': 'Supertrend (Bearish)',
                'line': {
                    'color': '#ff4757',
                    'width': 2
                }
            })

        return {
            'data': traces
        }