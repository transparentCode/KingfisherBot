import numpy as np
import pandas as pd
from typing import Dict, Any, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from numba import njit, prange
from app.indicators.BaseIndicatorInterface import BaseIndicatorInterface


@njit(fastmath=True)
def _calc_rs_hurst_numba(arr):
    """Calculates R/S Hurst for a single window slice."""
    n = len(arr)
    if n < 20: return 0.5
    
    # Log prices usually preferred for financial data, 
    # but classic R/S often uses simple returns or prices directly if stationary.
    # Assuming 'arr' is log-prices or prices. 
    
    # Calculate mean and centered values
    mean_x = np.mean(arr)
    y = arr - mean_x
    
    # Cumulative deviation
    z = np.cumsum(y)
    
    # Range
    R = np.max(z) - np.min(z)
    
    # Standard Deviation
    S = np.std(y) # Numba std uses ddof=0 by default usually, check version
    
    if S == 0 or R == 0:
        return 0.5
        
    return np.log(R / S) / np.log(n)

@njit(parallel=True)
def rolling_hurst_numba(prices, lookback):
    """
    Parallelized Rolling Hurst. 
    100x faster than Python Loop.
    """
    n = len(prices)
    result = np.full(n, np.nan)
    
    # Parallel loop over the array
    for i in prange(lookback, n):
        window = prices[i-lookback : i]
        result[i] = _calc_rs_hurst_numba(window)
        
    return result


class RegimeMetrics(BaseIndicatorInterface):
    """
    Computes regime statistics on price:
    - Rolling Hurst exponent (trendiness / memory)
    - Rolling skewness & kurtosis of log returns (asymmetry & tails)
    
    Produces:
    - hurst, skew, kurt columns
    - regime: 'TRENDING', 'MEAN_REVERTING', 'UNCERTAIN'
    - strategy_hint: high-level suggestion for which style to favor.
    """

    def __init__(
        self,
        name: str = "RegimeMetrics",
        hurst_lookback: int = 250,
        moment_lookback: int = 100,
        atr_period: int = 14,
        ema_period: int = 14,
        smoothing_type: str = "ema",
        volatility_lookback: int = 252,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.hurst_lookback = kwargs.get('hurst_lookback', hurst_lookback)
        self.moment_lookback = kwargs.get('moment_lookback', moment_lookback)
        
        # ATR Parameters
        self.atr_period = kwargs.get('atr_period', atr_period)
        self.ema_period = kwargs.get('ema_period', ema_period)
        self.smoothing_type = kwargs.get('smoothing_type', smoothing_type)
        self.volatility_lookback = kwargs.get('volatility_lookback', volatility_lookback)
        
        self.metrics_df = None

    # ---------- internal helpers ----------

    def _calculate_atr(self, data: pd.DataFrame) -> pd.Series:
        high = data['high']
        low = data['low']
        close = data['close']

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        if self.smoothing_type.lower() == 'ema':
            return true_range.ewm(span=self.atr_period, adjust=False).mean()
        if self.smoothing_type.lower() == 'sma':
            return true_range.rolling(self.atr_period).mean()
        if self.smoothing_type.lower() == 'rma':
            alpha = 1.0 / self.atr_period
            return true_range.ewm(alpha=alpha, adjust=False).mean()

        return true_range.ewm(span=self.atr_period, adjust=False).mean()


    def _calculate_atr_percentile(self, atr_series: pd.Series) -> pd.Series:
        """Calculates ATR Percentile Rank."""
        return atr_series.rolling(window=self.volatility_lookback).rank(pct=True) * 100

    def _rolling_window_view(self, arr: np.ndarray, window: int) -> np.ndarray:
        """Return a 2D rolling window view of a 1D array without copying."""
        if arr.ndim != 1:
            arr = np.asarray(arr).reshape(-1)
        n = arr.size
        if window > n:
            raise ValueError("window larger than array")
        shape = (n - window + 1, window)
        strides = (arr.strides[0], arr.strides[0])
        return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

    def _hurst_rs(self, series: np.ndarray) -> float:
        """
        Classic R/S Hurst estimate on a 1D array.
        H ≈ 0.5   : random
        H > 0.6   : trending/persistent
        H < 0.4   : mean-reverting/anti-persistent
        """
        x = np.log(series) if (series > 0).all() else series
        N = len(x)
        if N < 20:
            return np.nan

        mean_x = x.mean()
        y = x - mean_x
        z = y.cumsum()
        R = z.max() - z.min()
        S = y.std(ddof=1)

        if S == 0 or R == 0:
            return 0.5

        return np.log(R / S) / np.log(N)

    def _rolling_hurst(self, prices: pd.Series) -> pd.Series:
        arr = prices.values.astype(float)
        if len(arr) < self.hurst_lookback:
            return pd.Series(index=prices.index, dtype=float)

        windows = self._rolling_window_view(arr, self.hurst_lookback)
        hursts = np.array([self._hurst_rs(w) for w in windows])
        idx = prices.index[self.hurst_lookback - 1 :]
        return pd.Series(hursts, index=idx)

    def _rolling_moments(self, rets: pd.Series) -> pd.DataFrame:
        """
        Rolling skewness & excess kurtosis of returns.
        """
        r = rets.values.astype(float)
        n = len(r)
        if n < self.moment_lookback:
            return pd.DataFrame(
                index=rets.index, columns=["skew", "kurt"], dtype=float
            )

        windows = self._rolling_window_view(r, self.moment_lookback)
        skews = []
        kurts = []

        for w in windows:
            m = w.mean()
            s = w.std(ddof=1)
            if s == 0:
                skews.append(0.0)
                kurts.append(0.0)
                continue

            z = (w - m) / s
            skews.append((z ** 3).mean())
            kurts.append((z ** 4).mean() - 3.0)  # excess kurtosis

        idx = rets.index[self.moment_lookback - 1 :]
        return pd.DataFrame({"skew": skews, "kurt": kurts}, index=idx)


    def get_features(self, df: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """
        Extracts normalized features for the Bot Brain.
        Acts as the 'Gatekeeper' for strategy selection.
        """
        if self.metrics_df is None or kwargs.get('force_recalc', False):
            self.calculate(data=df)
            
        # Helper to get last value safely
        def get_last(col, default=0.0):
            if self.metrics_df is None or col not in self.metrics_df.columns: return default
            if self.metrics_df.empty: return default
            val = self.metrics_df[col].iloc[-1]
            return float(val) if not np.isnan(val) else default

        hurst = get_last('hurst', 0.5)
        skew = get_last('skew', 0.0)
        kurt = get_last('kurt', 0.0)
        atr_rank = get_last('atr_pct', 50.0)
        adaptive_L = get_last('adaptive_L', 100.0)

        # 1. Trend Quality Score (-1.0 to 1.0)
        # > 0.5: Strong Trending
        # < 0.0: Mean Reverting / Noise
        # We normalize Hurst (0.4-0.6 range) to a wider score
        trend_score = (hurst - 0.5) * 10.0 # 0.55 -> 0.5, 0.45 -> -0.5
        # Clamp to range -1 to 1
        trend_score = max(-1.0, min(1.0, trend_score))

        # 2. Volatility Stress (0.0 to 1.0)
        # 0.9+: Extreme Volatility (Stop Trading or Wide Stops)
        # 0.1-: Dead Market (Don't Scalp)
        vol_stress = atr_rank / 100.0
        vol_stress = max(0.0, min(1.0, vol_stress))

        # 3. Tail Risk (Kurtosis)
        # High Kurtosis (> 3.0) means "Fat Tails" -> Expect black swans / stop runs
        # We normalize it to a 'danger' metric
        # A kurtosis of 6.0 is extremely high, so we map 3.0 -> 0.5, 6.0 -> 1.0
        tail_risk = max(0.0, kurt / 6.0) 
        tail_risk = min(1.0, tail_risk) 

        return {
            'regime_hurst': float(hurst),
            'regime_trend_score': float(trend_score),
            'regime_vol_stress': float(vol_stress),
            'regime_skew': float(skew), # +Val = Bullish Bias
            'regime_tail_risk': float(tail_risk), # 0-1 (Risk of Black Swan)
            'regime_adaptive_L': float(adaptive_L) # Changed to float for consistency
        }



    # ---------- public API ----------

    def _get_default_params(self):
        return {
            'hurst_lookback': {'type': 'int', 'default': 250, 'min': 50, 'max': 500, 'description': 'Hurst Lookback'},
            'moment_lookback': {'type': 'int', 'default': 100, 'min': 20, 'max': 500, 'description': 'Skew/Kurt Lookback'},
            'atr_period': {'type': 'int', 'default': 14, 'min': 1, 'max': 100, 'description': 'ATR Period'},
            'volatility_lookback': {'type': 'int', 'default': 252, 'min': 50, 'max': 500, 'description': 'Vol Percentile Lookback'}
        }

    def _get_plot_trace(self, data: pd.DataFrame, **kwargs) -> List[Dict[str, Any]]:
        if self.metrics_df is None or self.metrics_df.empty:
            self.calculate(data)
            
        if self.metrics_df is None or self.metrics_df.empty:
            return []

        df = self.metrics_df.dropna()
        x_dates = df.index.tolist()
        
        traces = []
        
        # Hurst
        traces.append({
            'x': x_dates,
            'y': df['hurst'].tolist(),
            'type': 'scatter',
            'mode': 'lines',
            'name': 'Hurst Exponent',
            'line': {'color': 'cyan'}
        })
        
        # Skew
        traces.append({
            'x': x_dates,
            'y': df['skew'].tolist(),
            'type': 'scatter',
            'mode': 'lines',
            'name': 'Skewness',
            'line': {'color': 'orange'}
        })
        
        # Kurt
        traces.append({
            'x': x_dates,
            'y': df['kurt'].tolist(),
            'type': 'scatter',
            'mode': 'lines',
            'name': 'Kurtosis',
            'line': {'color': 'purple'}
        })
        
        # ATR Pct
        traces.append({
            'x': x_dates,
            'y': df['atr_pct'].tolist(),
            'type': 'scatter',
            'mode': 'lines',
            'name': 'ATR % Rank',
            'line': {'color': 'yellow'}
        })
        
        return traces

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        
        # 1. High-Performance Hurst (Numba)
        # Using Log Prices for better stationarity handling in R/S
        log_prices = np.log(df['close'].values)
        hurst_arr = rolling_hurst_numba(log_prices, self.hurst_lookback)
        df['hurst'] = hurst_arr

        # 2. Vectorized Moments (Pandas C-backend is fast enough here)
        # Using Log Returns
        log_rets = np.log(df['close']).diff()
        df['skew'] = log_rets.rolling(self.moment_lookback).skew()
        df['kurt'] = log_rets.rolling(self.moment_lookback).kurt() # Excess kurtosis by default

        # 3. Vectorized ATR & Rank
        high, low, close_s = df['high'], df['low'], df['close']
        tr = np.maximum(high - low, 
               np.maximum(abs(high - close_s.shift(1)), 
                          abs(low - close_s.shift(1))))
        
        # Apply smoothing based on configuration
        if self.smoothing_type.lower() == 'sma':
            df['atr'] = tr.rolling(self.atr_period).mean()
        elif self.smoothing_type.lower() == 'rma':
            alpha = 1.0 / self.atr_period
            df['atr'] = tr.ewm(alpha=alpha, adjust=False).mean()
        else: # Default to EMA
            df['atr'] = tr.ewm(span=self.atr_period, adjust=False).mean()

        # Rank of ATR over volatility_lookback
        df['atr_pct'] = df['atr'].rolling(self.volatility_lookback).rank(pct=True) * 100

        # 4. Vectorized Classification (No .apply!)
        # Create conditions
        cond_trend = df['hurst'] > 0.6
        cond_mean_rev = df['hurst'] < 0.4
        
        # Default to UNCERTAIN
        df['regime'] = 'UNCERTAIN'
        df.loc[cond_trend, 'regime'] = 'TRENDING'
        df.loc[cond_mean_rev, 'regime'] = 'MEAN_REVERTING'
        
        # 5. NEW: Calculate Adaptive Lookback (The "Actuator")
        # This prepares the L value for your other indicators
        df['adaptive_L'] = self._calculate_adaptive_length(
            df['hurst'].values, 
            df['atr_pct'].values
        )

        self.metrics_df = df[['hurst', 'skew', 'kurt', 'atr_pct', 'regime', 'adaptive_L']]
        return df

    def _calculate_adaptive_length(self, hurst, atr_rank, base_L=100, L_min=50, L_max=200):
        """
        Calculates the Adaptive Lookback (L) with Inertia Smoothing.
        Prevents rapid flip-flopping of indicator lengths.
        """
        # 1. Handle NaNs
        h = np.nan_to_num(hurst, nan=0.5)
        v = np.nan_to_num(atr_rank, nan=50.0)

        # 2. Raw Score Calculation
        h_norm = np.clip(h, 0.3, 0.8)
        v_norm = np.clip(v / 100.0, 0.0, 1.0)

        trend_factor = (h_norm - 0.3) / (0.5) # 0..1
        vol_factor = 1.0 - v_norm             # High Vol -> Short L
        
        # Weighted Score
        raw_score = 0.6 * trend_factor + 0.4 * vol_factor
        
        # Map to Raw Length
        raw_L = L_min + raw_score * (L_max - L_min)
        
        # 3. INERTIA SMOOTHING (Vectorized)
        # Use Pandas EWM for O(N) C-optimized smoothing
        # alpha=0.05 corresponds to span ≈ 39
        smooth_L = pd.Series(raw_L).ewm(alpha=0.05, adjust=False).mean().values
        
        # Handle initial NaN if any
        smooth_L = np.nan_to_num(smooth_L, nan=base_L)
            
        return np.round(smooth_L).astype(int)


    def plot(self, data: pd.DataFrame, **kwargs) -> None:
        """
        Plots the Regime Metrics: Hurst, Skewness, Kurtosis, and Volatility Percentile.
        """
        if self.metrics_df is None or self.metrics_df.empty:
            self.calculate(data)
            
        if self.metrics_df is None or self.metrics_df.empty:
            print("Insufficient data for Regime Metrics plot.")
            return

        # Align data
        plot_df = self.metrics_df.dropna()
        if plot_df.empty:
            return

        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                "Hurst Exponent (Trendiness)", 
                "Skewness (Asymmetry)", 
                "Kurtosis (Tail Risk)",
                "Volatility Percentile (ATR Rank)"
            )
        )

        # 1. Hurst Exponent
        fig.add_trace(go.Scatter(
            x=plot_df.index, y=plot_df['hurst'],
            mode='lines', name='Hurst',
            line=dict(color='cyan')
        ), row=1, col=1)
        
        # Hurst Thresholds
        fig.add_hline(y=0.5, line_dash="dot", line_color="gray", row=1, col=1)
        fig.add_hline(y=0.6, line_dash="dash", line_color="green", annotation_text="Trending (>0.6)", row=1, col=1)
        fig.add_hline(y=0.4, line_dash="dash", line_color="red", annotation_text="Mean Rev (<0.4)", row=1, col=1)

        # 2. Skewness
        fig.add_trace(go.Scatter(
            x=plot_df.index, y=plot_df['skew'],
            mode='lines', name='Skewness',
            line=dict(color='orange')
        ), row=2, col=1)
        fig.add_hline(y=0, line_color="gray", row=2, col=1)

        # 3. Kurtosis
        fig.add_trace(go.Scatter(
            x=plot_df.index, y=plot_df['kurt'],
            mode='lines', name='Excess Kurtosis',
            line=dict(color='purple')
        ), row=3, col=1)
        fig.add_hline(y=0, line_color="gray", row=3, col=1)
        fig.add_hline(y=3, line_dash="dot", line_color="red", annotation_text="Fat Tails (>3)", row=3, col=1)

        # 4. Volatility Percentile
        fig.add_trace(go.Scatter(
            x=plot_df.index, y=plot_df['atr_pct'],
            mode='lines', name='ATR % Rank',
            line=dict(color='yellow')
        ), row=4, col=1)
        fig.add_hline(y=80, line_dash="dot", line_color="red", annotation_text="High Vol (>80)", row=4, col=1)
        fig.add_hline(y=20, line_dash="dot", line_color="green", annotation_text="Low Vol (<20)", row=4, col=1)

        fig.update_layout(
            title="Regime Metrics Analysis (Trend + Volatility)",
            template='plotly_dark',
            height=1000,
            showlegend=True
        )
        fig.show()

    def atr(series: pd.DataFrame, length: int = 14) -> pd.Series:
        high = series['high']
        low = series['low']
        close = series['close']
        prev_close = close.shift(1)

        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)

        return tr.rolling(length).mean()

    def atr_percentile_rank(atr_series: pd.Series, lookback: int = 200) -> pd.Series:
        # percentile rank of current ATR vs last lookback values
        def _rank(x):
            last = x.iloc[-1]
            return (x <= last).mean()
        return atr_series.rolling(lookback).apply(_rank, raw=False)
