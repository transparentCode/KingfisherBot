import pandas as pd
import numpy as np
import plotly.graph_objects as go
from numba import njit, float64, boolean
from app.indicators.BaseIndicatorInterface import BaseIndicatorInterface
from app.utils.price_utils import get_price_source_data


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
        self.atr_len = kwargs.get('atr_len', kwargs.get('atrLen', 10))
        self.atr_mult = kwargs.get('atr_mult', kwargs.get('atrMult', 3.0))
        self.span = kwargs.get('span', 14)
        self.nu = kwargs.get('nu', 3.0)
        self.vol_window = kwargs.get('vol_window', kwargs.get('volWindow', 30))
        self.gamma = kwargs.get('gamma', 1.2)
        self.alpha_floor = kwargs.get('alpha_floor', kwargs.get('alphaFloor', 0.03))
        self.robust_atr = kwargs.get('robust_atr', kwargs.get('robustATR_on', True))
        self.input_columns = ['high', 'low', 'close', 'open']
        self.category = 'Trend Indicators'

    def _calculate_adaptive_alpha(self, series, base_alpha):
        r = series.pct_change().fillna(0)
        sigma = r.rolling(self.vol_window).std().fillna(1e-6)

        vol_pct = sigma.rank(pct=True) * 100
        nu_dyn = np.where(vol_pct > 70, self.nu * 0.7, np.where(vol_pct < 30, self.nu * 1.3, self.nu))

        alpha_raw = base_alpha / (1 + np.abs(r / sigma) ** nu_dyn)
        return (self.gamma * alpha_raw).clip(lower=self.alpha_floor, upper=1.0)

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculates the Student-t SuperTrend values.

        :param data: DataFrame with high, low, close columns.
        :return: DataFrame with t-EMA, SuperTrend, and direction columns.
        """
        df = data.copy()

        # Get parameters with proper fallbacks
        atr_len = kwargs.get('atr_len', self.atr_len)
        atr_mult = kwargs.get('atr_mult', self.atr_mult)
        span = kwargs.get('span', self.span)

        # 1. t-EMA Calculation
        base_alpha_ema = 2.0 / (span + 1.0)
        alpha_p_ema = self._calculate_adaptive_alpha(df['close'], base_alpha_ema)
        df['t_ema'] = _calculate_t_ema(df['close'].to_numpy(), alpha_p_ema.to_numpy())

        # 2. ATR Calculation
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        if self.robust_atr:
            base_alpha_atr = 2.0 / (atr_len + 1.0)
            alpha_p_atr = self._calculate_adaptive_alpha(tr, base_alpha_atr)
            atr = _calculate_t_ema(tr.to_numpy(), alpha_p_atr.to_numpy())
        else:
            atr = tr.ewm(span=atr_len, adjust=False).mean().to_numpy()

        price_eps = df['close'] * 0.0001
        df['atr'] = np.maximum(atr, price_eps.to_numpy())

        # 3. SuperTrend Core Calculation
        source = get_price_source_data(df, 'hlcc4')
        upper_band = source + atr_mult * df['atr']
        lower_band = source - atr_mult * df['atr']

        st, direction = _calculate_supertrend(
            df['close'].to_numpy(),
            upper_band.to_numpy(),
            lower_band.to_numpy()
        )

        # Use consistent column naming for both legacy and new format
        df['supertrend'] = st
        df['direction'] = direction
        df[f'supertrend_{atr_len}_{atr_mult}'] = st
        df[f'trend_{atr_len}_{atr_mult}'] = direction.astype(int) * 2 - 1  # Convert to 1/-1

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
            'atr_len': {'type': 'int', 'default': 10, 'min': 2, 'max': 100, 'description': 'ATR Length'},
            'atr_mult': {'type': 'float', 'default': 3.0, 'min': 1.0, 'max': 10.0, 'description': 'ATR Multiplier'},
            'span': {'type': 'int', 'default': 14, 'min': 1, 'max': 100, 'description': 'T-EMA Span'},
            'nu': {'type': 'float', 'default': 3.0, 'min': 1.0, 'max': 10.0, 'description': 'Student-t Nu Parameter'},
            'vol_window': {'type': 'int', 'default': 30, 'min': 5, 'max': 100, 'description': 'Volatility Window'},
            'gamma': {'type': 'float', 'default': 1.2, 'min': 0.1, 'max': 5.0, 'description': 'Gamma Parameter'},
            'alpha_floor': {'type': 'float', 'default': 0.03, 'min': 0.01, 'max': 0.5, 'description': 'Alpha Floor'},
            'robust_atr': {'type': 'bool', 'default': True, 'description': 'Use Robust ATR'}
        }

    def _get_plot_trace(self, data, **kwargs):
        """Return plotly trace for Supertrend"""
        atr_len = kwargs.get('atr_len', self.atr_len)
        atr_mult = kwargs.get('atr_mult', self.atr_mult)

        # Use the legacy column names that are actually created
        supertrend_col = 'supertrend'
        direction_col = 'direction'

        # Calculate if columns don't exist
        if supertrend_col not in data.columns or direction_col not in data.columns:
            data = self.calculate(data, **kwargs)

        if supertrend_col not in data.columns or direction_col not in data.columns:
            raise ValueError(f"Supertrend columns not found in data")

        # Create traces for bullish and bearish trends
        # Convert boolean direction to proper format
        bullish_mask = data[direction_col] == True
        bearish_mask = data[direction_col] == False

        bullish_data = data[bullish_mask]
        bearish_data = data[bearish_mask]

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

        return traces