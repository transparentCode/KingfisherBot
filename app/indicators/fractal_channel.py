import numpy as np
import pandas as pd
import plotly.graph_objects as go

from typing import Tuple, Dict, Any
from app.indicators.BaseIndicatorInterface import BaseIndicatorInterface


class FractalChannel(BaseIndicatorInterface):
    """
    Fractal Channel Indicator.
    
    Modes:
    - 'geometric': Fits separate regression lines to High and Low pivots.
    - 'dynamic': Fits a center regression line to pivots and adds a volatility-based envelope.
    
    Pivot Methods:
    - 'zigzag': Uses percentage deviation to find pivots (Recommended).
    - 'fractal': Uses Williams Fractal logic (window-based).
    """


    def __init__(self, name: str = "FractalChannel", **kwargs):
        super().__init__(name, **kwargs)
        # Configurable Parameters
        self.pivot_method = kwargs.get('pivot_method', 'fractal')   # 'zigzag' or 'fractal'
        self.zigzag_dev = kwargs.get('zigzag_dev', 0.05)           # Deviation for ZigZag (e.g., 0.02 = 2%)
        self.pivot_window = kwargs.get('pivot_window', 5)          # Window for Williams Fractal
        
        self.lookback = kwargs.get('lookback', 150)
        self.min_pivots = kwargs.get('min_pivots', 3)
        self.mode = kwargs.get('mode', 'geometric')
        self.mult = kwargs.get('mult', 2.0)
        self.source = kwargs.get('source', 'close')
        self.category = 'Trendlines'
        
        if self.mode == 'geometric':
            self.input_columns = ['high', 'low']
        else:
            self.input_columns = [self.source]


    def get_parameter_schema(self) -> Dict[str, Any]:
        return {
            'pivot_method': {
                'type': 'string',
                'default': 'fractal',
                'options': ['zigzag', 'fractal'],
                'description': 'Method to find pivots'
            },
            'zigzag_dev': {
                'type': 'number',
                'default': 0.05,
                'min': 0.001,
                'max': 0.2,
                'description': 'Deviation for ZigZag (e.g. 0.02 = 2%)'
            },
            'pivot_window': {
                'type': 'integer',
                'default': 5,
                'min': 2,
                'max': 50,
                'description': 'Window size for Williams Fractal'
            },
            'lookback': {
                'type': 'integer',
                'default': 150,
                'min': 10,
                'max': 500,
                'description': 'Minimum Lookback period'
            },
            'mode': {
                'type': 'string',
                'default': 'geometric',
                'options': ['geometric', 'dynamic'],
                'description': 'Channel calculation mode'
            },
            'mult': {
                'type': 'number',
                'default': 2.0,
                'min': 0.1,
                'max': 5.0,
                'description': 'Multiplier for dynamic mode'
            }
        }


    def _find_pivots_fractal(self, series_high: np.ndarray, series_low: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Williams Fractal Pivot Detection"""
        n = len(series_high)
        if n < 2 * self.pivot_window + 1:
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        high_idxs = []
        low_idxs = []
        
        # Scan valid middle range
        for i in range(self.pivot_window, n - self.pivot_window):
            if np.isnan(series_high[i]) or np.isnan(series_low[i]):
                continue


            window_high = series_high[i - self.pivot_window : i + self.pivot_window + 1]
            window_low = series_low[i - self.pivot_window : i + self.pivot_window + 1]
            
            # Use nanmax/nanmin to handle NaNs in window
            if series_high[i] == np.nanmax(window_high):
                high_idxs.append(i)
            if series_low[i] == np.nanmin(window_low):
                low_idxs.append(i)
                
        return (
            np.array(high_idxs, dtype=int), 
            series_high[high_idxs] if high_idxs else np.array([]),
            np.array(low_idxs, dtype=int), 
            series_low[low_idxs] if low_idxs else np.array([])
        )


    def _find_pivots_zigzag(self, series_high: np.ndarray, series_low: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ZigZag Pivot Detection (Percentage Deviation)"""
        n = len(series_high)
        if n < 2:
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        # Find first valid index to handle leading NaNs
        start_idx = 0
        while start_idx < n and (np.isnan(series_high[start_idx]) or np.isnan(series_low[start_idx])):
            start_idx += 1
            
        if start_idx >= n - 1:
             return np.array([]), np.array([]), np.array([]), np.array([])


        high_idxs = []
        low_idxs = []
        
        # Simple ZigZag Logic
        trend = 0 # 1=up, -1=down
        last_high_idx = start_idx
        last_low_idx = start_idx
        last_high_val = series_high[start_idx]
        last_low_val = series_low[start_idx]
        
        for i in range(start_idx + 1, n):
            if np.isnan(series_high[i]) or np.isnan(series_low[i]):
                continue


            # Check for reversal downwards
            if trend >= 0:
                if series_high[i] > last_high_val:
                    last_high_idx = i
                    last_high_val = series_high[i]
                elif series_low[i] < last_high_val * (1 - self.zigzag_dev):
                    high_idxs.append(last_high_idx)
                    trend = -1
                    last_low_idx = i
                    last_low_val = series_low[i]
            
            # Check for reversal upwards
            elif trend <= 0:
                if series_low[i] < last_low_val:
                    last_low_idx = i
                    last_low_val = series_low[i]
                elif series_high[i] > last_low_val * (1 + self.zigzag_dev):
                    low_idxs.append(last_low_idx)
                    trend = 1
                    last_high_idx = i
                    last_high_val = series_high[i]
                    
        return (
            np.array(high_idxs, dtype=int), series_high[high_idxs] if high_idxs else np.array([]),
            np.array(low_idxs, dtype=int), series_low[low_idxs] if low_idxs else np.array([])
        )


    def _fit_line(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        # Filter 1: Min Pivots
        if len(x) < self.min_pivots:
            return np.nan, np.nan
        
        # Filter 2: Min Duration (prevents steep noise lines)
        if len(x) > 1 and (x[-1] - x[0]) < 10:
            return np.nan, np.nan
        
        slope, intercept = np.polyfit(x, y, 1)
        return slope, intercept


    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        
        # Override Params from call
        pivot_method = kwargs.get('pivot_method', self.pivot_method)
        lookback = kwargs.get('lookback', self.lookback)
        mode = kwargs.get('mode', self.mode)
        mult = kwargs.get('mult', self.mult)
        source_name = kwargs.get('source', self.source)
        
        # Update instance params if provided (for helper methods)
        self.zigzag_dev = kwargs.get('zigzag_dev', self.zigzag_dev)
        self.pivot_window = kwargs.get('pivot_window', self.pivot_window)


        upper_col = f'fc_upper_{lookback}_{mode}'
        lower_col = f'fc_lower_{lookback}_{mode}'
        mid_col = f'fc_mid_{lookback}_{mode}'
        
        # Use numpy arrays for speed optimization inside loop
        n = len(df)
        out_upper = np.full(n, np.nan)
        out_lower = np.full(n, np.nan)
        out_mid = np.full(n, np.nan)


        # Determine Source Data
        volatility = None
        if source_name in df.columns and source_name != 'close':
            # Indicator Mode (RSI, MACD) -> Use the indicator for both High and Low
            highs = df[source_name].values
            lows = df[source_name].values
            volatility = df[source_name].rolling(window=lookback).std().values
        elif mode == 'geometric':
            # Price Mode -> Use actual High/Low
            highs = df['high'].values
            lows = df['low'].values
        else:
            # Dynamic Price Mode -> Use Close
            highs = df['close'].values
            lows = df['close'].values
            volatility = df['close'].rolling(window=lookback).std().values


        if volatility is None and mode == 'dynamic':
             target_col = source_name if source_name in df.columns else 'close'
             volatility = df[target_col].rolling(window=lookback).std().values


        # --- ANCHOR LOGIC ---
        # Pre-calculate rolling max for anchors to fix recency bias
        anchor_window = 200 
        rolling_max_indices = df['high'].rolling(anchor_window).apply(np.argmax, raw=True).fillna(0).astype(int)
        
        # Start loop later to accommodate anchor window
        start_t = max(lookback, anchor_window)


        # Rolling Calculation Loop
        for t in range(start_t, n):
            
            # 1. Determine Window Start (Anchored)
            rel_anchor = rolling_max_indices[t]
            abs_anchor_idx = (t - anchor_window + 1) + rel_anchor
            slice_start = max(0, abs_anchor_idx)
            
            # Fallback for anchor too close
            if (t - slice_start) < lookback:
                slice_start = t - lookback
                
            start_idx = slice_start
            end_idx = t + 1 
            
            h_slice = highs[start_idx:end_idx]
            l_slice = lows[start_idx:end_idx]
            
            # Choose Pivot Method
            if pivot_method == 'zigzag':
                h_idx, h_vals, l_idx, l_vals = self._find_pivots_zigzag(h_slice, l_slice)
            else: # 'fractal'
                h_idx, h_vals, l_idx, l_vals = self._find_pivots_fractal(h_slice, l_slice)
            
            curr_rel_idx = (end_idx - 1) - start_idx
            
            if mode == 'geometric':
                up_m, up_c = self._fit_line(h_idx, h_vals)
                low_m, low_c = self._fit_line(l_idx, l_vals)
                
                if not np.isnan(up_m):
                    out_upper[t] = up_m * curr_rel_idx + up_c
                if not np.isnan(low_m):
                    out_lower[t] = low_m * curr_rel_idx + low_c
                    
            elif mode == 'dynamic':
                all_idx = np.concatenate([h_idx, l_idx])
                all_vals = np.concatenate([h_vals, l_vals])
                
                if len(all_idx) >= self.min_pivots:
                    mid_m, mid_c = self._fit_line(all_idx, all_vals)
                    if not np.isnan(mid_m):
                        mid_val = mid_m * curr_rel_idx + mid_c
                        width = volatility[t] * mult
                        out_mid[t] = mid_val
                        out_upper[t] = mid_val + width
                        out_lower[t] = mid_val - width


        # Assign back to DataFrame
        df[upper_col] = out_upper
        df[lower_col] = out_lower
        if mode == 'dynamic':
            df[mid_col] = out_mid
            
        # --- SIGNAL GENERATION ---
        # Determine which column to check against the channel
        check_col = 'close'
        if mode == 'dynamic' and source_name in df.columns:
            check_col = source_name

        # 1: Bullish Breakout, -1: Bearish Breakdown, 0: Neutral
        df['fc_signal'] = 0
        
        # Fill NaNs for comparison (infinity prevents false signals at start)
        upper_filled = df[upper_col].fillna(np.inf)
        lower_filled = df[lower_col].fillna(-np.inf)
        
        series_to_check = df[check_col]

        # Bullish: Close Crosses Above Upper
        bullish_cond = (series_to_check > upper_filled) & (series_to_check.shift(1) <= upper_filled.shift(1))
        df.loc[bullish_cond, 'fc_signal'] = 1
        
        # Bearish: Close Crosses Below Lower
        bearish_cond = (series_to_check < lower_filled) & (series_to_check.shift(1) >= lower_filled.shift(1))
        df.loc[bearish_cond, 'fc_signal'] = -1
            
        return df


    def plot(self, data: pd.DataFrame, **kwargs) -> None:
        pass

    def _calculate_structural_projection(self, df_window: pd.DataFrame, pivots_idx: np.ndarray, pivots_val: np.ndarray, is_res: bool) -> Tuple[float, float, float, int]:
        """
        Calculates the projected value and slope of the Structural (DP) trendline.
        
        Enhanced Algorithm Features:
        - Touch count bonus: Lines touching more pivots score higher
        - Recency weighting: Recent pivots contribute more to score
        - Full-path regression: Uses all path points for projection
        - R² metric: Quantifies goodness-of-fit
        
        Returns:
            Tuple of (projected_value, slope, r_squared, touch_count)
            Returns (np.nan, np.nan, 0.0, 0) if no valid path is found.
        """
        if len(pivots_idx) < 2: 
            return np.nan, np.nan, 0.0, 0

        opens, closes = df_window['open'].values, df_window['close'].values
        n_pivots = len(pivots_idx)
        
        # Determine strictness for validation
        # Resistance: Line should be above Max(Open, Close) (Bodies)
        # Using Bodies allows for wick violations which is common in Structural Channels
        price_highs = np.maximum(opens, closes)
        price_lows = np.minimum(opens, closes)

        # DP Logic (Find Best Path with Enhanced Scoring)
        # stored as {pivot_index: {'score': total_score, 'prev': prev_pivot_index, 'touches': touch_count}}
        dp = {idx: {'score': 0.0, 'prev': -1, 'touches': 0} for idx in pivots_idx}
        
        for i in range(n_pivots):
            curr, curr_v = pivots_idx[i], pivots_val[i]
            
            # Look back at all previous pivots
            for j in range(i):
                prev, prev_v = pivots_idx[j], pivots_val[j]
                
                # Calculate Slope/Intercept
                if curr == prev: 
                    continue
                slope = (curr_v - prev_v) / (curr - prev)
                intercept = prev_v - (slope * prev)
                
                # Validation Scan (Vectorized)
                start_k, end_k = int(prev) + 1, int(curr)
                
                valid = True
                if start_k < end_k:
                    # Generate line values for the range
                    k_indices = np.arange(start_k, end_k)
                    line_vals = slope * k_indices + intercept
                    
                    if is_res:
                        # Resistance: Line must be >= Price Body Highs
                        if np.any(line_vals < price_highs[start_k:end_k]):
                            valid = False
                    else:
                        # Support: Line must be <= Price Body Lows
                        if np.any(line_vals > price_lows[start_k:end_k]):
                            valid = False
                
                if valid:
                    # Enhanced Scoring:
                    # 1. Base: Segment length (distance in X)
                    segment_length = float(curr - prev)
                    
                    # 2. Recency weight: Later pivots are more relevant (1.0 to 1.5x)
                    recency_weight = 1.0 + (i / n_pivots) * 0.5
                    
                    # 3. Touch bonus: Each valid connection counts as a touch
                    touch_bonus = 1.0
                    
                    # Combined score
                    score = dp[prev]['score'] + (segment_length * recency_weight) + touch_bonus
                    
                    if score > dp[curr]['score']:
                        dp[curr]['score'] = score
                        dp[curr]['prev'] = prev
                        dp[curr]['touches'] = dp[prev]['touches'] + 1

        # Find Best Path (Endpoint with highest total score)
        best = max(dp, key=lambda k: dp[k]['score'])
        
        if dp[best]['score'] == 0 or dp[best]['prev'] == -1: 
            return np.nan, np.nan, 0.0, 0

        # Backtrack to collect ALL path points for full-path regression
        idx_map = {idx: val for idx, val in zip(pivots_idx, pivots_val)}
        path_points = []
        curr = best
        while curr != -1:
            path_points.append((curr, idx_map[curr]))
            curr = dp[curr]['prev']
        
        path_points = path_points[::-1]  # Reverse to chronological order
        touch_count = len(path_points)
        
        # Full-Path Regression: Fit line through ALL path pivots (not just last 2)
        path_x = np.array([p[0] for p in path_points], dtype=float)
        path_y = np.array([p[1] for p in path_points], dtype=float)
        
        slope, intercept = np.polyfit(path_x, path_y, 1)
        
        # Calculate R² (Goodness-of-fit)
        y_pred = slope * path_x + intercept
        ss_res = np.sum((path_y - y_pred) ** 2)
        ss_tot = np.sum((path_y - np.mean(path_y)) ** 2)
        r_squared = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
        
        # Clamp R² to valid range
        r_squared = max(0.0, min(1.0, r_squared))
        
        # Project to Current Bar using full-path regression
        current_idx = len(df_window) - 1
        projected_val = slope * current_idx + intercept
        
        return float(projected_val), float(slope), r_squared, touch_count

    def get_features(self, df: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """
        Extracts normalized features for the Bot Brain.
        Returns dictionary of signals.
        """
        if df.empty: return {}
        
        lookback = kwargs.get('lookback', self.lookback)
        mode = kwargs.get('mode', self.mode)
        
        # Get Column Names
        upper_col = f'fc_upper_{lookback}_{mode}'
        lower_col = f'fc_lower_{lookback}_{mode}'
        
        # Ensure columns exist
        if upper_col not in df.columns:
            return {}

        # 1. Geometric Data (Fast Lookups)
        current_close = df['close'].iloc[-1]
        geo_upper = df[upper_col].iloc[-1]
        geo_lower = df[lower_col].iloc[-1]
        
        if np.isnan(geo_upper) or np.isnan(geo_lower):
            return {}

        # FEATURE 1: Normalized Position (The Oscillator)
        channel_height = geo_upper - geo_lower
        norm_pos = (current_close - geo_lower) / (channel_height + 1e-9) # Avoid div/0

        # FEATURE 2: Geometric Slope Strength (Upper & Lower)
        # Using previous value to determine slope
        geo_upper_prev = df[upper_col].iloc[-2]
        geo_lower_prev = df[lower_col].iloc[-2]
        
        slope_upper_raw = geo_upper - geo_upper_prev
        slope_lower_raw = geo_lower - geo_lower_prev
        
        slope_upper_pct = (slope_upper_raw / (current_close + 1e-9)) * 100 
        slope_lower_pct = (slope_lower_raw / (current_close + 1e-9)) * 100

        # FEATURE 3: Channel Width & Shape
        width_pct = (channel_height / (current_close + 1e-9))
        
        # Wedge Factor: Difference helps identify squeezing vs expanding
        # If Upper is pointing down (-) and Lower is pointing up (+), result is negative (Squeeze)
        # If Upper is pointing up (+) and Lower is pointing down (-), result is positive (Expansion)
        wedge_factor = slope_upper_pct - slope_lower_pct

        # FEATURE 4 & 5: Structural Divergence & Slope (The Alpha)
        # Re-extract window to find "Latest" structural pivots
        window = df.iloc[-lookback:].copy()
        
        if self.pivot_method == 'zigzag':
            h_idx, h_vals, l_idx, l_vals = self._find_pivots_zigzag(window['high'].values, window['low'].values)
        else:
            h_idx, h_vals, l_idx, l_vals = self._find_pivots_fractal(window['high'].values, window['low'].values)
            
        # Run DP Projection (Now returns 4 values: val, slope, r², touches)
        struct_res_val, struct_res_slope, res_r2, res_touches = self._calculate_structural_projection(window, h_idx, h_vals, is_res=True)
        struct_sup_val, struct_sup_slope, sup_r2, sup_touches = self._calculate_structural_projection(window, l_idx, l_vals, is_res=False)
        
        # Calculate Divergence (Difference between Structural and Geometric levels)
        sup_div = 0.0
        res_div = 0.0
        
        if not np.isnan(struct_sup_val):
            sup_div = (struct_sup_val - geo_lower) / (geo_lower + 1e-9)
            
        if not np.isnan(struct_res_val):
            res_div = (struct_res_val - geo_upper) / (geo_upper + 1e-9)

        # Normalize Structural Slopes
        norm_sup_slope = 0.0
        norm_res_slope = 0.0
        if not np.isnan(struct_sup_slope):
            norm_sup_slope = (struct_sup_slope / (current_close + 1e-9)) * 100
        if not np.isnan(struct_res_slope):
            norm_res_slope = (struct_res_slope / (current_close + 1e-9)) * 100

        # Signal State
        signal_state = 0.0
        if 'fc_signal' in df.columns:
            signal_state = float(df['fc_signal'].iloc[-1])

        # Pivot Metrics (Quality of Fit)
        pivots_count = len(h_idx) + len(l_idx)
        
        # Pivot Age: How "fresh" is the pattern? (Normalized by lookback)
        last_pivot_idx = 0
        if len(h_idx) > 0: last_pivot_idx = max(last_pivot_idx, h_idx[-1])
        if len(l_idx) > 0: last_pivot_idx = max(last_pivot_idx, l_idx[-1])
        
        # h_idx are relative to the window start (0 to lookback-1)
        # Calculate distance from end of window
        pivot_age_bars = len(window) - 1 - last_pivot_idx
        pivot_age_norm = pivot_age_bars / lookback

        # FEATURE 6: Slope Divergence (Structural Slope - Geometric Slope)
        # A positive value means Structure is more bullish than History.
        slope_div_up = norm_res_slope - slope_upper_pct
        slope_div_down = norm_sup_slope - slope_lower_pct
        
        # NEW: Confidence Score (Combines R² and touch count)
        # Higher R² + more touches = higher confidence
        res_confidence = res_r2 * min(1.0, res_touches / 5.0) if res_touches > 0 else 0.0
        sup_confidence = sup_r2 * min(1.0, sup_touches / 5.0) if sup_touches > 0 else 0.0

        return {
            'fc_norm_pos': float(norm_pos),
            'fc_slope_pct': float(slope_upper_pct),
            'fc_slope_lower_pct': float(slope_lower_pct),
            'fc_wedge_factor': float(wedge_factor),
            'fc_width_pct': float(width_pct),
            'fc_sup_div': float(sup_div),
            'fc_res_div': float(res_div),
            'fc_sup_slope': float(norm_sup_slope),
            'fc_res_slope': float(norm_res_slope),
            'fc_slope_div_up': float(slope_div_up),
            'fc_slope_div_down': float(slope_div_down),
            'fc_geo_upper': float(geo_upper),
            'fc_geo_lower': float(geo_lower),
            'fc_signal': float(signal_state),
            'fc_pivots_count': float(pivots_count),
            'fc_pivot_age': float(pivot_age_norm),
            # NEW: Enhanced quality metrics
            'fc_res_r2': float(res_r2),
            'fc_sup_r2': float(sup_r2),
            'fc_res_touches': float(res_touches),
            'fc_sup_touches': float(sup_touches),
            'fc_res_confidence': float(res_confidence),
            'fc_sup_confidence': float(sup_confidence)
        }



    def _get_plot_trace(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Build Plotly-ready traces including:
        - Rolling channel (upper/lower/mid)
        - Breakout/breakdown markers
        - Latest-window static channel (upper/lower) and pivot markers
        """

        lookback = kwargs.get('lookback', self.lookback)
        mode = kwargs.get('mode', self.mode)
        pivot_method = kwargs.get('pivot_method', self.pivot_method)
        zigzag_dev = kwargs.get('zigzag_dev', self.zigzag_dev)
        pivot_window = kwargs.get('pivot_window', self.pivot_window)

        upper_col = f'fc_upper_{lookback}_{mode}'
        lower_col = f'fc_lower_{lookback}_{mode}'
        mid_col = f'fc_mid_{lookback}_{mode}'

        dict_traces: list[Dict[str, Any]] = []

        # Rolling channel traces
        if upper_col in data.columns:
            dict_traces.append({
                'x': data.index.tolist(),
                'y': data[upper_col].tolist(),
                'mode': 'lines',
                'name': 'Rolling Resistance (History)',
                'line': {'color': 'rgba(0, 255, 0, 0.3)', 'width': 1},
                'type': 'scatter'
            })

        if lower_col in data.columns:
            dict_traces.append({
                'x': data.index.tolist(),
                'y': data[lower_col].tolist(),
                'mode': 'lines',
                'name': 'Rolling Support (History)',
                'line': {'color': 'rgba(255, 0, 0, 0.3)', 'width': 1},
                'type': 'scatter'
            })

        if mode == 'dynamic' and mid_col in data.columns:
            dict_traces.append({
                'x': data.index.tolist(),
                'y': data[mid_col].tolist(),
                'mode': 'lines',
                'name': f'FC Mid ({mode})',
                'line': {'width': 1, 'color': 'gray'}, 
                'type': 'scatter'
            })

        # Breakout/breakdown markers
        if 'fc_signal' in data.columns:
            bullish = data[data['fc_signal'] == 1]
            if not bullish.empty:
                dict_traces.append({
                    'x': bullish.index.tolist(),
                    'y': (bullish['low'] * 0.99).tolist(),
                    'mode': 'markers',
                    'name': 'Bullish Breakout',
                    'marker': {'symbol': 'triangle-up', 'size': 10, 'color': 'lime'},
                    'type': 'scatter'
                })

            bearish = data[data['fc_signal'] == -1]
            if not bearish.empty:
                dict_traces.append({
                    'x': bearish.index.tolist(),
                    'y': (bearish['high'] * 1.01).tolist(),
                    'mode': 'markers',
                    'name': 'Bearish Breakdown',
                    'marker': {'symbol': 'triangle-down', 'size': 10, 'color': 'red'},
                    'type': 'scatter'
                })

        # Latest-window static channel and pivot markers
        # Note: We allow len(data) < lookback to match notebook behavior (it just takes what's available)
        if lookback and not data.empty and {'high', 'low'}.issubset(data.columns):
            window = data.iloc[-lookback:]
            highs = window['high'].values
            lows = window['low'].values

            # Select pivot finder
            if pivot_method == 'zigzag':
                self.zigzag_dev = zigzag_dev
                h_idx, h_vals, l_idx, l_vals = self._find_pivots_zigzag(highs, lows)
            else:
                self.pivot_window = pivot_window
                h_idx, h_vals, l_idx, l_vals = self._find_pivots_fractal(highs, lows)

            # Fit lines on latest window pivots
            up_m, up_c = self._fit_line(h_idx, h_vals)
            low_m, low_c = self._fit_line(l_idx, l_vals)
            x_rel = np.arange(len(window))

            if not np.isnan(up_m):
                static_upper = up_m * x_rel + up_c
                dict_traces.append({
                    'x': window.index.tolist(),
                    'y': static_upper.tolist(),
                    'mode': 'lines',
                    'name': 'LATEST Resistance Pattern',
                    'line': {'color': 'lime', 'width': 3, 'dash': 'solid'},
                    'type': 'scatter'
                })

            if not np.isnan(low_m):
                static_lower = low_m * x_rel + low_c
                dict_traces.append({
                    'x': window.index.tolist(),
                    'y': static_lower.tolist(),
                    'mode': 'lines',
                    'name': 'LATEST Support Pattern',
                    'line': {'color': 'red', 'width': 3, 'dash': 'solid'},
                    'type': 'scatter'
                })

            # Pivot markers on latest window
            if len(h_idx) > 0:
                dict_traces.append({
                    'x': window.index[h_idx].tolist(),
                    'y': h_vals.tolist(),
                    'mode': 'markers',
                    'name': f'Latest High Pivots ({pivot_method})',
                    'marker': {'color': 'lime', 'size': 8, 'symbol': 'triangle-up'},
                    'type': 'scatter'
                })

            if len(l_idx) > 0:
                dict_traces.append({
                    'x': window.index[l_idx].tolist(),
                    'y': l_vals.tolist(),
                    'mode': 'markers',
                    'name': f'Latest Low Pivots ({pivot_method})',
                    'marker': {'color': 'red', 'size': 8, 'symbol': 'triangle-down'},
                    'type': 'scatter'
                })

        return {
            'data': dict_traces,
            'layout_update': {}
        }
    
    @staticmethod
    def compare_trendlines_plot(df: pd.DataFrame, h_idx: np.ndarray, h_vals: np.ndarray, l_idx: np.ndarray, l_vals: np.ndarray):
        """
        Plots Price + Geometric (Regression) + Structural (DP) Trendlines on one chart.
        
        Enhanced version with:
        - Recency-weighted scoring
        - Full-path regression for projections
        - R² display in legend
        
        Parameters:
        - df: DataFrame with 'open', 'high', 'low', 'close', index (datetime)
        - h_idx, h_vals: Arrays of High Pivot indices and values
        - l_idx, l_vals: Arrays of Low Pivot indices and values
        
        Returns:
        - plotly.graph_objects.Figure
        """
        
        # --- HELPER: Enhanced DP Logic ---
        def _find_path(pivots_idx, pivots_val, is_res):
            """
            Enhanced pathfinding with recency weighting and full-path regression.
            Returns: (path_points, slope, intercept, r_squared)
            """
            if len(pivots_idx) < 2: 
                return [], np.nan, np.nan, 0.0
            
            opens, closes = df['open'].values, df['close'].values
            n_pivots = len(pivots_idx)
            dp = {idx: {'score': 0.0, 'prev': -1, 'touches': 0} for idx in pivots_idx}
            
            for i in range(n_pivots):
                curr, curr_v = int(pivots_idx[i]), pivots_val[i]
                for j in range(i):
                    prev, prev_v = int(pivots_idx[j]), pivots_val[j]
                    if curr == prev:
                        continue
                    slope = (curr_v - prev_v) / (curr - prev)
                    intercept = prev_v - (slope * prev)
                    
                    # Vectorized validation
                    valid = True
                    if prev + 1 < curr:
                        k_range = np.arange(prev + 1, curr)
                        line_vals = slope * k_range + intercept
                        if is_res:
                            price_limit = np.maximum(opens[prev+1:curr], closes[prev+1:curr])
                            if np.any(line_vals < price_limit):
                                valid = False
                        else:
                            price_limit = np.minimum(opens[prev+1:curr], closes[prev+1:curr])
                            if np.any(line_vals > price_limit):
                                valid = False
                    
                    if valid:
                        # Enhanced scoring with recency weight
                        segment_length = float(curr - prev)
                        recency_weight = 1.0 + (i / n_pivots) * 0.5
                        touch_bonus = 1.0
                        score = dp[prev]['score'] + (segment_length * recency_weight) + touch_bonus
                        
                        if score > dp[curr]['score']:
                            dp[curr]['score'] = score
                            dp[curr]['prev'] = prev
                            dp[curr]['touches'] = dp[prev]['touches'] + 1
                            
            best = max(dp, key=lambda k: dp[k]['score'])
            if dp[best]['score'] == 0 or dp[best]['prev'] == -1:
                return [], np.nan, np.nan, 0.0
            
            # Backtrack to collect ALL path points
            idx_map = {idx: val for idx, val in zip(pivots_idx, pivots_val)}
            path = []
            curr = best
            while curr != -1:
                path.append((curr, idx_map[curr]))
                curr = dp[curr]['prev']
            path = path[::-1]
            
            # Full-path regression
            path_x = np.array([p[0] for p in path], dtype=float)
            path_y = np.array([p[1] for p in path], dtype=float)
            slope, intercept = np.polyfit(path_x, path_y, 1)
            
            # R² calculation
            y_pred = slope * path_x + intercept
            ss_res = np.sum((path_y - y_pred) ** 2)
            ss_tot = np.sum((path_y - np.mean(path_y)) ** 2)
            r_squared = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
            r_squared = max(0.0, min(1.0, r_squared))
            
            return path, slope, intercept, r_squared

        # --- PLOTTING ---
        fig = go.Figure()

        # 1. Price
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['open'], high=df['high'],
            low=df['low'], close=df['close'], name='Price'
        ))

        # 2. Geometric (Regression)
        x_rel = np.arange(len(df))
        if len(h_idx) >= 2:
            m, c = np.polyfit(h_idx, h_vals, 1)
            fig.add_trace(go.Scatter(x=df.index, y=m*x_rel + c, mode='lines', 
                                    name='Geometric Res', line=dict(color='lime', width=1)))
        if len(l_idx) >= 2:
            m, c = np.polyfit(l_idx, l_vals, 1)
            fig.add_trace(go.Scatter(x=df.index, y=m*x_rel + c, mode='lines', 
                                    name='Geometric Sup', line=dict(color='red', width=1)))

        # 3. Structural (Enhanced DP Pathfinding)
        res_path, res_slope, res_intercept, res_r2 = _find_path(h_idx, h_vals, True)
        if res_path and len(res_path) >= 2:
            # A. PLOT HISTORY (The Zig-Zag path)
            path_x = [df.index[p[0]] for p in res_path]
            path_y = [p[1] for p in res_path]
            
            fig.add_trace(go.Scatter(
                x=path_x, y=path_y,
                mode='lines+markers', name=f'Struct Res Path (R²={res_r2:.2f})',
                line=dict(color='yellow', width=1, dash='dot')
            ))
            
            # B. PLOT PROJECTION (Full-path regression line)
            proj_x = [df.index[0], df.index[-1]]
            proj_y = [res_slope * 0 + res_intercept, res_slope * (len(df)-1) + res_intercept]
            
            fig.add_trace(go.Scatter(
                x=proj_x, y=proj_y,
                mode='lines', name=f'Struct Res Projection',
                line=dict(color='yellow', width=3)
            ))

        sup_path, sup_slope, sup_intercept, sup_r2 = _find_path(l_idx, l_vals, False)
        if sup_path and len(sup_path) >= 2:
            # A. PLOT HISTORY (The Zig-Zag path)
            path_x = [df.index[p[0]] for p in sup_path]
            path_y = [p[1] for p in sup_path]
            
            fig.add_trace(go.Scatter(
                x=path_x, y=path_y,
                mode='lines+markers', name=f'Struct Sup Path (R²={sup_r2:.2f})',
                line=dict(color='cyan', width=1, dash='dot')
            ))
            
            # B. PLOT PROJECTION (Full-path regression line)
            proj_x = [df.index[0], df.index[-1]]
            proj_y = [sup_slope * 0 + sup_intercept, sup_slope * (len(df)-1) + sup_intercept]
            
            fig.add_trace(go.Scatter(
                x=proj_x, y=proj_y,
                mode='lines', name=f'Struct Sup Projection',
                line=dict(color='cyan', width=3)
            ))

        fig.update_layout(template='plotly_dark', title='Geometric vs Structural Trendlines (Enhanced)')
        return fig
