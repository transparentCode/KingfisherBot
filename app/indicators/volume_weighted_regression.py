import numpy as np
import pandas as pd
from typing import Dict, Any, List
import plotly.graph_objects as go
from app.indicators.BaseIndicatorInterface import BaseIndicatorInterface

class VolumeWeightedRegression(BaseIndicatorInterface):
    """
    Calculates both Standard and Volume-Weighted Trendlines.
    Detects divergence between 'Geometric Trend' and 'Money Trend'.
    
    Enhanced with:
    - R² goodness-of-fit metrics
    - Configurable thresholds
    - Volume outlier clipping
    - Rolling calculation mode
    """

    def __init__(self, name: str = "VolumeWeightedRegression", **kwargs):
        super().__init__(name, **kwargs)
        self.lookback = kwargs.get('lookback', 50)
        
        # Configurable thresholds
        self.overbought_threshold = kwargs.get('overbought_threshold', 0.02)
        self.oversold_threshold = kwargs.get('oversold_threshold', -0.02)
        
        # Volume clipping (percentile, e.g., 95 = clip at 95th percentile)
        self.volume_clip_pct = kwargs.get('volume_clip_pct', None)
        
        self.metrics: Dict[str, float] = {}

    def _get_default_params(self):
        return {
            'lookback': {'type': 'int', 'default': 50, 'min': 10, 'max': 200, 'description': 'Regression Lookback'},
            'overbought_threshold': {'type': 'float', 'default': 0.02, 'min': 0.005, 'max': 0.10, 'description': 'Overbought gap threshold (e.g., 0.02 = 2%)'},
            'oversold_threshold': {'type': 'float', 'default': -0.02, 'min': -0.10, 'max': -0.005, 'description': 'Oversold gap threshold (e.g., -0.02 = -2%)'},
            'volume_clip_pct': {'type': 'float', 'default': None, 'min': 50, 'max': 99, 'description': 'Percentile to clip volume outliers (None = no clipping)'}
        }

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Returns slopes and current projected values for both regressions.
        
        Enhanced with:
        - R² goodness-of-fit metrics
        - Volume outlier clipping
        - Rolling calculation mode
        
        Args:
            data: DataFrame with 'close' and 'volume' columns
            rolling: If True, calculate for all bars (for backtesting)
        """
        df = data.copy()
        rolling = kwargs.get('rolling', False)
        lookback = kwargs.get('lookback', self.lookback)
        volume_clip_pct = kwargs.get('volume_clip_pct', self.volume_clip_pct)
        
        # Ensure sufficient data
        if len(df) < lookback:
            return df

        if rolling:
            # Rolling mode: calculate for each bar
            n = len(df)
            out_vw_center = np.full(n, np.nan)
            out_vw_upper = np.full(n, np.nan)
            out_vw_lower = np.full(n, np.nan)
            out_geom_line = np.full(n, np.nan)
            out_vw_r2 = np.full(n, np.nan)
            out_geom_r2 = np.full(n, np.nan)
            out_signal = np.zeros(n, dtype=int)
            
            closes = df['close'].values.astype(float)
            volumes = df['volume'].values.astype(float)
            
            for t in range(lookback, n + 1):
                y = np.nan_to_num(closes[t-lookback:t], nan=0.0)
                v = np.nan_to_num(volumes[t-lookback:t], nan=0.0)
                x = np.arange(lookback)
                
                # Optional volume clipping
                if volume_clip_pct is not None:
                    clip_val = np.percentile(v, volume_clip_pct)
                    v = np.clip(v, 0, clip_val)
                
                # Geometric regression
                geom_slope, geom_intercept = np.polyfit(x, y, 1)
                geom_price = geom_slope * (lookback - 1) + geom_intercept
                
                # Volume-weighted regression
                mean_vol = np.mean(v)
                weights = v / (mean_vol + 1e-9) if mean_vol > 0 else np.ones_like(v)
                vw_slope, vw_intercept = np.polyfit(x, y, 1, w=weights)
                vw_price = vw_slope * (lookback - 1) + vw_intercept
                
                # Residuals and std dev
                vw_line = vw_slope * x + vw_intercept
                residuals = y - vw_line
                std_dev = np.std(residuals)
                
                # R² calculations
                geom_line = geom_slope * x + geom_intercept
                geom_r2 = self._calc_r2(y, geom_line)
                vw_r2 = self._calc_r2(y, vw_line)
                
                # Store values
                idx = t - 1
                out_vw_center[idx] = vw_price
                out_vw_upper[idx] = vw_price + 2 * std_dev
                out_vw_lower[idx] = vw_price - 2 * std_dev
                out_geom_line[idx] = geom_price
                out_vw_r2[idx] = vw_r2
                out_geom_r2[idx] = geom_r2
                
                # Signal generation
                gap_pct = (geom_price - vw_price) / (geom_price + 1e-9)
                if gap_pct > self.overbought_threshold:
                    out_signal[idx] = -1  # Overbought
                elif gap_pct < self.oversold_threshold:
                    out_signal[idx] = 1   # Oversold
            
            df['vwr_center'] = out_vw_center
            df['vwr_upper'] = out_vw_upper
            df['vwr_lower'] = out_vw_lower
            df['vwr_geom'] = out_geom_line
            df['vwr_vw_r2'] = out_vw_r2
            df['vwr_geom_r2'] = out_geom_r2
            df['vwr_signal'] = out_signal
            
            return df

        # Non-rolling mode (original behavior, enhanced)
        subset = df.iloc[-lookback:]
        y = np.nan_to_num(subset['close'].values.astype(float), nan=0.0)
        v = np.nan_to_num(subset['volume'].values.astype(float), nan=0.0)
        x = np.arange(lookback)
        
        # Optional volume clipping
        if volume_clip_pct is not None:
            clip_val = np.percentile(v, volume_clip_pct)
            v = np.clip(v, 0, clip_val)

        # 1. Standard Regression (Geometric)
        geom_slope, geom_intercept = np.polyfit(x, y, 1)
        geom_price = geom_slope * (lookback - 1) + geom_intercept
        geom_line = geom_slope * x + geom_intercept

        # 2. Volume-Weighted Regression (Money)
        mean_vol = np.mean(v)
        if mean_vol == 0:
             weights = np.ones_like(v)
        else:
             weights = v / (mean_vol + 1e-9)
        
        vw_slope, vw_intercept = np.polyfit(x, y, 1, w=weights)
        vw_price = vw_slope * (lookback - 1) + vw_intercept
        vw_line = vw_slope * x + vw_intercept

        # 3. Residuals and std dev
        residuals = y - vw_line
        std_dev = np.std(residuals)

        # 4. R² calculations
        geom_r2 = self._calc_r2(y, geom_line)
        vw_r2 = self._calc_r2(y, vw_line)

        # 5. Divergence Metrics
        slope_divergence = geom_slope - vw_slope
        price_divergence = geom_price - vw_price
        price_divergence_pct = price_divergence / (geom_price + 1e-9)
        
        # 6. Confidence score (combines R² with slope consistency)
        avg_r2 = (geom_r2 + vw_r2) / 2
        slope_agreement = 1.0 if (geom_slope * vw_slope) > 0 else 0.5  # Same direction = bonus
        confidence = avg_r2 * slope_agreement

        self.metrics = {
            'geom_slope': geom_slope,
            'vw_slope': vw_slope,
            'geom_price': geom_price,
            'vw_price': vw_price,
            'std_dev': std_dev,
            'slope_divergence': slope_divergence,
            'price_gap_pct': price_divergence_pct,
            # NEW metrics
            'geom_r2': geom_r2,
            'vw_r2': vw_r2,
            'confidence': confidence
        }
        

        return df

    def _calc_r2(self, y_actual: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate R² (coefficient of determination) for regression fit.
        Returns value between 0 and 1 (clamped).
        """
        ss_res = np.sum((y_actual - y_pred) ** 2)
        ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
        r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
        return max(0.0, min(1.0, r2))

    def get_signal(self) -> str:
        """
        Interprets the metrics into a trading signal.
        """
        if not self.metrics:
            return "NEUTRAL"

        g_slope = self.metrics['geom_slope']
        v_slope = self.metrics['vw_slope']
        gap = self.metrics['price_gap_pct']

        # Scenario 1: Fake Pump (Price Up, Volume Flat/Down)
        if g_slope > 0 and v_slope < 0:
            return "BEARISH_DIVERGENCE_PUMP"
            
        # Scenario 2: Fake Dump (Price Down, Volume Flat/Up)
        if g_slope < 0 and v_slope > 0:
            return "BULLISH_DIVERGENCE_DUMP"

        # Scenario 3: Overextension (Price far above Volume Line)
        if gap > self.overbought_threshold:
            return "OVERBOUGHT_REVERSION"
            
        # Scenario 4: Undervalued (Price far below Volume Line)
        if gap < self.oversold_threshold:
            return "OVERSOLD_REVERSION"

        # Scenario 5: Trend Confirmation
        if g_slope > 0 and v_slope > 0:
            return "HEALTHY_UPTREND"
        if g_slope < 0 and v_slope < 0:
            return "HEALTHY_DOWNTREND"

        return "NEUTRAL"

    def get_features(self, df: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """
        Extracts normalized features for the Bot Brain.
        """
        # Ensure metrics are calculated
        if not self.metrics or kwargs.get('force_recalc', False):
            self.calculate(df)
            
        if not self.metrics:
            return {}

        # 1. Normalized Gap (Z-Score)
        # How many standard deviations is the current Geometric Price away from the Volume Weighted Price?
        # High Z-Score (> 2.0) = Unstable Extension
        gap_raw = self.metrics['geom_price'] - self.metrics['vw_price']
        std_dev = self.metrics['std_dev']
        
        # Avoid division by zero
        if std_dev == 0:
            z_score = 0.0
        else:
            z_score = gap_raw / std_dev

        # 2. Effort vs Result Ratio (Efficiency)
        # Result = Geometric Slope (Price movement)
        # Effort = Volume Weighted Slope (Money movement)
        
        g_slope = self.metrics['geom_slope']
        v_slope = self.metrics['vw_slope']
        
        # Divergence Strength
        # If g_slope is huge but v_slope is small -> Divergence (Fakeout)
        # We normalize this by price to make it comparable across assets
        current_price = df['close'].iloc[-1]
        
        slope_div_norm = (g_slope - v_slope) / (current_price + 1e-9) * 1000  # Scaled for readability & Safety
        
        # 3. Efficiency Ratios
        # Ratio of Volume Slope to Geometric Slope
        # > 1.0: Money is moving faster than price (Bullish/Accumulation)
        # < 1.0: Price is moving faster than money (Thin Liquidity/Fakeout)
        # Add epsilon to signs to avoid zero issues
        eff_ratio = 0.0
        if abs(g_slope) > 1e-9:
            eff_ratio = v_slope / g_slope
            
        return {
            'vwr_z_score': float(z_score),         # -2.0 to +2.0 (Extension)
            'vwr_slope_div': float(slope_div_norm),# +Val = Price outpacing Volume (Speculation)
            'vwr_money_slope': float(v_slope),     # The "True" Trend direction
            'vwr_eff_ratio': float(eff_ratio),     # Flow Efficiency
            'vwr_price_gap': float(self.metrics['price_gap_pct']), # Gap %
            'vwr_fair_value': float(self.metrics['vw_price']), # Absolute level for TP
            # NEW: Quality metrics
            'vwr_geom_r2': float(self.metrics.get('geom_r2', 0.0)),
            'vwr_vw_r2': float(self.metrics.get('vw_r2', 0.0)),
            'vwr_confidence': float(self.metrics.get('confidence', 0.0)),
            'vwr_signal': self.get_signal()
        }


    def _get_plot_trace(self, data: pd.DataFrame, **kwargs) -> List[Dict[str, Any]]:
        """
        Generates Plotly traces for the frontend.
        """
        if not self.metrics:
            self.calculate(data)
            
        if not self.metrics:
            return []

        subset = data.iloc[-self.lookback:]
        x_dates = subset.index.tolist()
        
        # Reconstruct lines
        x_nums = np.arange(self.lookback)
        
        # VW Line
        vw_slope = self.metrics['vw_slope']
        # Intercept at x=0 (start of window)
        # vw_price is at x=lookback-1
        # y = mx + c => c = y - mx
        vw_intercept_start = self.metrics['vw_price'] - vw_slope * (self.lookback - 1)
        vw_line = vw_slope * x_nums + vw_intercept_start

        # Geometric Line
        geom_slope = self.metrics['geom_slope']
        geom_intercept_start = self.metrics['geom_price'] - geom_slope * (self.lookback - 1)
        geom_line = geom_slope * x_nums + geom_intercept_start
        
        # Bands
        std_dev = self.metrics['std_dev']
        upper_band = vw_line + 2 * std_dev
        lower_band = vw_line - 2 * std_dev
        
        traces = []
        
        # VW Center
        traces.append({
            'x': x_dates,
            'y': vw_line.tolist(),
            'type': 'scatter',
            'mode': 'lines',
            'name': 'VWR Center',
            'line': {'color': 'cyan', 'width': 2}
        })

        # Geometric Trend
        traces.append({
            'x': x_dates,
            'y': geom_line.tolist(),
            'type': 'scatter',
            'mode': 'lines',
            'name': 'Geometric Trend',
            'line': {'color': 'orange', 'width': 2, 'dash': 'dash'}
        })
        
        # Upper Band
        traces.append({
            'x': x_dates,
            'y': upper_band.tolist(),
            'type': 'scatter',
            'mode': 'lines',
            'name': 'VWR Upper (2σ)',
            'line': {'color': 'cyan', 'width': 1, 'dash': 'dot'}
        })
        
        # Lower Band
        traces.append({
            'x': x_dates,
            'y': lower_band.tolist(),
            'type': 'scatter',
            'mode': 'lines',
            'name': 'VWR Lower (2σ)',
            'line': {'color': 'cyan', 'width': 1, 'dash': 'dot'}
        })
        
        return traces

    def plot(self, data: pd.DataFrame, **kwargs) -> None:
        """
        Plots the VWR Channel using Plotly.
        """
        if not self.metrics:
            self.calculate(data)
            
        if not self.metrics:
            print("Insufficient data for VWR plot.")
            return

        subset = data.iloc[-self.lookback:]
        
        fig = go.Figure()

        # 1. Candlesticks
        fig.add_trace(go.Candlestick(
            x=subset.index,
            open=subset['open'], high=subset['high'],
            low=subset['low'], close=subset['close'],
            name='Price'
        ))

        # 2. Volume Weighted Trend (Center)
        y_vw_end = self.metrics['vw_price']
        y_vw_start = y_vw_end - self.metrics['vw_slope'] * (self.lookback - 1)
        
        fig.add_trace(go.Scatter(
            x=[subset.index[0], subset.index[-1]], 
            y=[y_vw_start, y_vw_end],
            mode='lines', name='VWR Center',
            line=dict(color='cyan', width=2)
        ))

        # 2b. Geometric Trend
        y_geom_end = self.metrics['geom_price']
        y_geom_start = y_geom_end - self.metrics['geom_slope'] * (self.lookback - 1)

        fig.add_trace(go.Scatter(
            x=[subset.index[0], subset.index[-1]], 
            y=[y_geom_start, y_geom_end],
            mode='lines', name='Geometric Trend',
            line=dict(color='orange', width=2, dash='dash')
        ))

        # 3. Deviation Bands
        std_dev = self.metrics.get('std_dev', 0)
        std_multiplier = kwargs.get('std_multiplier', 2.0)
        
        # Upper Band
        y_up_start = y_vw_start + (std_dev * std_multiplier)
        y_up_end = y_vw_end + (std_dev * std_multiplier)
        
        fig.add_trace(go.Scatter(
            x=[subset.index[0], subset.index[-1]], 
            y=[y_up_start, y_up_end],
            mode='lines', name=f'+{std_multiplier} StdDev',
            line=dict(color='rgba(0, 255, 255, 0.3)', width=1, dash='dash')
        ))

        # Lower Band
        y_down_start = y_vw_start - (std_dev * std_multiplier)
        y_down_end = y_vw_end - (std_dev * std_multiplier)
        
        fig.add_trace(go.Scatter(
            x=[subset.index[0], subset.index[-1]], 
            y=[y_down_start, y_down_end],
            mode='lines', name=f'-{std_multiplier} StdDev',
            line=dict(color='rgba(0, 255, 255, 0.3)', width=1, dash='dash'),
            fill='tonexty', # Fill to the trace before it (Upper Band)
            fillcolor='rgba(0, 255, 255, 0.05)'
        ))

        slope_div = self.metrics.get('slope_divergence', 0)
        price_gap = self.metrics.get('price_gap_pct', 0) * 100
        
        title_text = f"Volume Weighted Regression Channel (Lookback: {self.lookback}, Std: {std_multiplier})<br>Slope Div: {slope_div:.6f} | Price Gap: {price_gap:.2f}%"

        fig.update_layout(
            title=title_text,
            template='plotly_dark',
            height=600
        )
        fig.show()
