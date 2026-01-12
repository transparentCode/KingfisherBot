import numpy as np
import pandas as pd
from typing import Dict, Any, List
import plotly.graph_objects as go
from numba import njit, prange
from app.indicators.BaseIndicatorInterface import BaseIndicatorInterface


# ---------------------------------------------------------
# 1. NUMBA KERNEL (Theil-Sen + MAD)
# ---------------------------------------------------------

@njit(parallel=True, fastmath=True)
def _calc_robust_channel(y, x, v, lookback):
    """
    Calculates Volume-Weighted Theil-Sen Slope, Intercept, and MAD.
    """
    n = len(y)
    out_slope = np.full(n, np.nan)
    out_intercept = np.full(n, np.nan)
    out_mad = np.full(n, np.nan)
    
    max_pairs = int((lookback * (lookback - 1)) / 2)
    
    for t in prange(lookback, n):
        # Extract Window
        y_w = y[t-lookback:t]
        x_w = x[t-lookback:t]
        v_w = v[t-lookback:t]
        
        slopes = np.zeros(max_pairs)
        weights = np.zeros(max_pairs)
        pair_idx = 0
        
        # 1. Pairwise Slopes
        for i in range(lookback):
            for j in range(i + 1, lookback):
                dx = x_w[j] - x_w[i]
                if dx == 0: continue
                slope = (y_w[j] - y_w[i]) / dx
                slopes[pair_idx] = slope
                weights[pair_idx] = min(v_w[i], v_w[j])
                pair_idx += 1
                
        # 2. Weighted Median Slope
        k = pair_idx
        if k < 2: 
            out_slope[t] = 0.0
            continue
            
        # Sort by slope
        sort_idxs = np.argsort(slopes[:k])
        sorted_slopes = slopes[sort_idxs]
        sorted_weights = weights[sort_idxs]
        
        total_weight = np.sum(sorted_weights)
        target = total_weight / 2.0
        current_w = 0.0
        median_slope = 0.0
        
        for i in range(k):
            current_w += sorted_weights[i]
            if current_w >= target:
                median_slope = sorted_slopes[i]
                break
                
        out_slope[t] = median_slope
        
        # 3. Intercept (Pass through weighted median point)
        # y = mx + c  =>  c = y - mx
        intercepts = np.zeros(lookback)
        for i in range(lookback):
            intercepts[i] = y_w[i] - median_slope * x_w[i]
            
        # We weight intercepts by volume
        i_sort = np.argsort(intercepts)
        s_int = intercepts[i_sort]
        s_w = v_w[i_sort] # Weight by volume of that point
        
        tot_w_int = np.sum(s_w)
        t_int = tot_w_int / 2.0
        curr_w_int = 0.0
        median_intercept = 0.0
        
        for i in range(lookback):
            curr_w_int += s_w[i]
            if curr_w_int >= t_int:
                median_intercept = s_int[i]
                break
                
        out_intercept[t] = median_intercept
        
        # 4. MAD (Median Absolute Deviation) - The Robust "Standard Deviation"
        residuals = np.zeros(lookback)
        for i in range(lookback):
            # Regression value at this point
            reg_val = median_slope * x_w[i] + median_intercept
            residuals[i] = abs(y_w[i] - reg_val)
            
        # Median of residuals
        # (Standard MAD doesn't need weighting, usually sufficient)
        residuals.sort()
        mad = residuals[lookback // 2]
        
        out_mad[t] = mad
        
    return out_slope, out_intercept, out_mad


# ---------------------------------------------------------
# 2. INDICATOR CLASS
# ---------------------------------------------------------

class RobustTrend(BaseIndicatorInterface):
    """
    Volume-Weighted Robust Trend Channel.
    - Center: Theil-Sen Estimator (Median Slope).
    - Bands: MAD (Median Absolute Deviation).
    """

    def __init__(self, name: str = "RobustTrend", **kwargs):
        super().__init__(name, **kwargs)
        self.lookback = kwargs.get('lookback', 50)
        self.band_mult = kwargs.get('band_mult', 3.0) # 3.0 MAD approx 2.0 Sigma
        self.metrics: Dict[str, float] = {}
        self.output_df = None

    def _get_default_params(self):
        return {
            'lookback': {'type': 'int', 'default': 50, 'min': 10, 'max': 200},
            'band_mult': {'type': 'float', 'default': 3.0, 'min': 1.0, 'max': 5.0}
        }

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        if len(df) < self.lookback: return df
            
        y = df['close'].values.astype(np.float64)
        v = df['volume'].values.astype(np.float64)
        x = np.arange(len(y), dtype=np.float64)
        
        slope_arr, intercept_arr, mad_arr = _calc_robust_channel(y, x, v, self.lookback)
        
        # NOTE: rt_line is the "Running" trend (the curve of rolling medians)
        # For visualization, we usually project the straight line of the *latest* calculation
        df['rt_slope'] = slope_arr
        df['rt_line'] = slope_arr * x + intercept_arr 
        df['rt_mad'] = mad_arr
        
        # Latest Values
        last_slope = slope_arr[-1] if not np.isnan(slope_arr[-1]) else 0.0
        last_line_val = df['rt_line'].iloc[-1]
        last_mad = mad_arr[-1] if not np.isnan(mad_arr[-1]) else 0.0
        
        self.metrics = {
            'slope': last_slope,
            'price_at_line': last_line_val,
            'mad': last_mad
        }
        
        self.output_df = df
        return df

    def get_features(self, df: pd.DataFrame, **kwargs) -> Dict[str, float]:
        if self.output_df is None or kwargs.get('force_recalc', False):
            self.calculate(df)
            
        current_price = df['close'].iloc[-1]
        line_price = self.metrics.get('price_at_line', current_price)
        slope = self.metrics.get('slope', 0.0)
        mad = self.metrics.get('mad', 0.0)
        
        # 1. Robust Z-Score
        # (Price - Trend) / (1.4826 * MAD)
        # 1.4826 converts MAD to Sigma equivalent for Normal Dist
        sigma_equiv = mad * 1.4826
        z_score = 0.0
        if sigma_equiv > 0:
            z_score = (current_price - line_price) / sigma_equiv
            
        # 2. Normalized Slope
        slope_pct = (slope / current_price) * 100
        
        return {
            'rt_z_score': float(z_score),      # > 2.0 = Overbought
            'rt_slope_pct': float(slope_pct),  # Trend Direction
            'rt_line_price': float(line_price),
            'rt_width_pct': float((mad * self.band_mult * 2) / current_price * 100) # Volatility width
        }

    def plot(self, data: pd.DataFrame, **kwargs) -> None:
        if self.output_df is None:
            self.calculate(data)
            
        subset = self.output_df.iloc[-self.lookback:]
        
        # Reconstruct Straight Line for Current Window
        slope = self.metrics['slope']
        mad = self.metrics['mad']
        y_end = self.metrics['price_at_line']
        y_start = y_end - slope * (self.lookback - 1)
        
        # Bands (Straight Parallel Lines)
        band_offset = mad * self.band_mult
        
        fig = go.Figure()
        
        # Candles
        fig.add_trace(go.Candlestick(
            x=subset.index, open=subset['open'], high=subset['high'],
            low=subset['low'], close=subset['close'], name='Price'
        ))
        
        # Center Line
        fig.add_trace(go.Scatter(
            x=[subset.index[0], subset.index[-1]], 
            y=[y_start, y_end],
            mode='lines', name='Robust Trend',
            line=dict(color='yellow', width=2)
        ))
        
        # Upper Band
        fig.add_trace(go.Scatter(
            x=[subset.index[0], subset.index[-1]], 
            y=[y_start + band_offset, y_end + band_offset],
            mode='lines', name=f'+{self.band_mult} MAD',
            line=dict(color='yellow', width=1, dash='dash')
        ))
        
        # Lower Band
        fig.add_trace(go.Scatter(
            x=[subset.index[0], subset.index[-1]], 
            y=[y_start - band_offset, y_end - band_offset],
            mode='lines', name=f'-{self.band_mult} MAD',
            line=dict(color='yellow', width=1, dash='dash'),
            fill='tonexty', fillcolor='rgba(255, 255, 0, 0.1)'
        ))
        
        fig.update_layout(title="Robust Regression Channel", template='plotly_dark')
        fig.show()

    def _get_plot_trace(self, data: pd.DataFrame, **kwargs) -> List[Dict[str, Any]]:
        # Returns traces for the frontend
        if self.output_df is None: self.calculate(data)
        
        slope = self.metrics.get('slope', 0.0)
        mad = self.metrics.get('mad', 0.0)
        y_end = self.metrics.get('price_at_line', 0.0)
        y_start = y_end - slope * (self.lookback - 1)
        
        start_date = data.index[-self.lookback]
        end_date = data.index[-1]
        offset = mad * self.band_mult
        
        return [
            {'x': [start_date, end_date], 'y': [y_start, y_end], 'type': 'scatter', 'mode': 'lines', 'name': 'Robust Center', 'line': {'color': 'yellow'}},
            {'x': [start_date, end_date], 'y': [y_start + offset, y_end + offset], 'type': 'scatter', 'mode': 'lines', 'name': 'Robust Upper', 'line': {'color': 'yellow', 'dash': 'dash'}},
            {'x': [start_date, end_date], 'y': [y_start - offset, y_end - offset], 'type': 'scatter', 'mode': 'lines', 'name': 'Robust Lower', 'line': {'color': 'yellow', 'dash': 'dash'}}
        ]
