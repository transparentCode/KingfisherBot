import numpy as np
import pandas as pd
from typing import Dict, Any, List
import plotly.graph_objects as go
from numba import njit
from app.indicators.BaseIndicatorInterface import BaseIndicatorInterface

# ---------------------------------------------------------
# 1. NUMBA KERNEL (EHLERS DOMINANT CYCLE)
# ---------------------------------------------------------
@njit(fastmath=True)
def _calc_hilbert_cycle_numba(prices):
    """
    Calculates Dominant Cycle Period using Ehlers Homodyne Discriminator.
    Source: Cycle Analytics for Traders, pp. 60-61.
    """
    n = len(prices)
    period = np.full(n, 20.0)  # Default period initialization
    
    # Intermediate arrays
    smooth = np.zeros(n)
    detrender = np.zeros(n)
    q1 = np.zeros(n)
    i1 = np.zeros(n)
    
    # Smoothed Real/Imag parts for Homodyne
    smooth_real = np.zeros(n)
    smooth_imag = np.zeros(n)

    # Begin Loop (Need history for lags)
    # Start at index 6 to accommodate lag-6 in detrender
    for i in range(6, n):
        
        # 1. Smooth Price (4-bar WMA)
        smooth[i] = (4*prices[i] + 3*prices[i-1] + 2*prices[i-2] + prices[i-3]) / 10.0
        
        # 2. Detrend (Hilbert Transform requires detrended data)
        # The multiplier adjusts for the variable period to keep the HT tuned
        adj_prev = 0.075 * period[i-1] + 0.54
        
        detrender[i] = (0.0962*smooth[i] + 0.5769*smooth[i-2] - 0.5769*smooth[i-4] - 0.0962*smooth[i-6]) * adj_prev
        
        # 3. Compute In-Phase (I) and Quadrature (Q)
        # Quadrature is the Hilbert Transform of the detrender (90 deg phase shift)
        q1[i] = (0.0962*detrender[i] + 0.5769*detrender[i-2] - 0.5769*detrender[i-4] - 0.0962*detrender[i-6]) * adj_prev
        
        # In-Phase is simply the detrender delayed by 3 bars (approx 90 deg at typical freq)
        i1[i] = detrender[i-3]
        
        # 4. Homodyne Discriminator (Complex Conjugate Multiplication)
        # Measures the rotation angle between the current phasor and the previous phasor
        # (I + jQ) * (I_prev - jQ_prev)
        
        real_part = (i1[i] * i1[i-1]) + (q1[i] * q1[i-1])
        imag_part = (i1[i] * q1[i-1]) - (q1[i] * i1[i-1])
        
        # 5. Smooth the Phase Components
        smooth_real[i] = 0.2 * real_part + 0.8 * smooth_real[i-1]
        smooth_imag[i] = 0.2 * imag_part + 0.8 * smooth_imag[i-1]
        
        # 6. Compute Cycle Period
        if smooth_imag[i] != 0 and smooth_real[i] != 0:
            # Arctan gives the phase change per bar (in degrees)
            # 360 / phase_change = bars per cycle
            cycle_deg = np.arctan2(smooth_imag[i], smooth_real[i]) * (180.0 / np.pi)
            
            # Ensure positive cycle
            if cycle_deg < 0: cycle_deg += 360
            if cycle_deg > 0:
                 inst_period = 360.0 / cycle_deg
            else:
                 inst_period = period[i-1]
                 
            period[i] = inst_period
            
        else:
            period[i] = period[i-1]

        # 7. Clamp and Smooth Period (Stability Checks)
        # Limit change to <50% per bar to prevent "jitter"
        period[i] = min(period[i], 1.5 * period[i-1])
        period[i] = max(period[i], 0.67 * period[i-1])
        
        # Hard bounds for trading utility
        period[i] = min(max(period[i], 6.0), 50.0)
        
        # Final Smoothing
        period[i] = 0.2 * period[i] + 0.8 * period[i-1]

    return period

# ---------------------------------------------------------
# 2. INDICATOR CLASS
# ---------------------------------------------------------

class HilbertCycle(BaseIndicatorInterface):
    """
    Computes the Dominant Cycle Period using the Hilbert Transform.
    Used for adaptive lookbacks in Oscillators and Channels.
    """
    
    def __init__(
        self, 
        name: str = "HilbertCycle", 
        source: str = "close",
        smooth_factor: float = 0.2,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.source = source
        self.smooth_factor = smooth_factor
        self.output_df = None

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Returns DataFrame with:
        - dc_period: Dominant Cycle Length (e.g., 22.4)
        - dc_phase: (Optional extension) Phase angle
        """
        df = data.copy()
        
        # Validate Input
        if self.source not in df.columns:
            return df
            
        prices = df[self.source].values.astype(np.float64)
        
        # Execute Numba Kernel
        dc_period = _calc_hilbert_cycle_numba(prices)
        
        # Assign to DataFrame
        df['dc_period'] = dc_period
        
        # Optional: Half-Period for Oscillators (e.g. RSI Length)
        df['half_period'] = (dc_period * 0.5).astype(int)
        
        self.output_df = df
        return df

    def get_features(self, df: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """
        Extracts Cycle features for the Bot Brain.
        """
        if self.output_df is None or kwargs.get('force_recalc', False):
            self.calculate(df)
            
        period = self.output_df['dc_period'].iloc[-1]
        
        # 1. Cycle Regime
        # Low Period (<15) = High Frequency Noise or Compressed Volatility
        # High Period (>40) = Strong Trending (Cycle is elongated)
        
        cycle_state = 0.0
        if period < 12: cycle_state = -1.0 # Too fast to trade cycles
        elif period > 35: cycle_state = 1.0 # Trend Mode
        
        # 2. Rate of Change (Speed)
        # Fast changing period = Market Instability
        # If the period is jumping around, the market is unsure of its own heartbeat
        period_prev = self.output_df['dc_period'].iloc[-2]
        period_change = period - period_prev

        return {
            'hilbert_period': float(period),
            'hilbert_half_period': float(period * 0.5), # Useful for RSI length (as float)
            'hilbert_state': float(cycle_state),
            'hilbert_change': float(period_change)
        }


    def plot(self, data: pd.DataFrame, **kwargs) -> None:
        if self.output_df is None:
            self.calculate(data)
            
        fig = go.Figure()
        
        # Plot Cycle Period
        fig.add_trace(go.Scatter(
            x=self.output_df.index, 
            y=self.output_df['dc_period'],
            mode='lines',
            name='Dominant Cycle (Period)',
            line=dict(color='orange', width=2)
        ))
        
        # Plot Reference Bands (Typical Bounds)
        fig.add_hline(y=20, line_dash="dot", line_color="gray", annotation_text="Standard 20")
        fig.add_hline(y=10, line_dash="dot", line_color="gray", annotation_text="Fast Cycle")
        fig.add_hline(y=40, line_dash="dot", line_color="gray", annotation_text="Slow Cycle")

        fig.update_layout(
            title="Hilbert Dominant Cycle Period",
            yaxis_title="Period (Bars)",
            template="plotly_dark",
            height=400
        )
        fig.show()

    def _get_plot_trace(self, data: pd.DataFrame, **kwargs) -> List[Dict[str, Any]]:
        if self.output_df is None:
            self.calculate(data)
            
        return [{
            'x': self.output_df.index.tolist(),
            'y': self.output_df['dc_period'].tolist(),
            'type': 'scatter',
            'mode': 'lines',
            'name': 'Hilbert Cycle',
            'line': {'color': 'orange'}
        }]

    def _get_default_params(self):
        return {
            'source': {'type': 'string', 'default': 'close', 'options': ['close', 'hl2', 'hlc3']},
            'smooth_factor': {'type': 'float', 'default': 0.2, 'min': 0.01, 'max': 1.0}
        }
