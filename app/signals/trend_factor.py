import pandas as pd
import talib
import numpy as np
from app.signals.base_factor import BaseSignalFactor

class TrendFactor(BaseSignalFactor):
    """
    Returns +1 if Price > SuperTrend/EMA, -1 if < SuperTrend/EMA.
    """
    def __init__(self):
        super().__init__("TrendFactor")

    def _calculate_supertrend(self, high, low, close, length=10, multiplier=3.0):
        atr = talib.ATR(high, low, close, timeperiod=length)
        hl2 = (high + low) / 2
        
        basic_upper = hl2 + (multiplier * atr)
        basic_lower = hl2 - (multiplier * atr)
        
        final_upper = np.zeros(len(close))
        final_lower = np.zeros(len(close))
        supertrend = np.zeros(len(close))
        
        # Initialize
        final_upper[0] = basic_upper[0]
        final_lower[0] = basic_lower[0]
        
        for i in range(1, len(close)):
            # Final Upper Band
            if basic_upper[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]:
                final_upper[i] = basic_upper[i]
            else:
                final_upper[i] = final_upper[i-1]
                
            # Final Lower Band
            if basic_lower[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]:
                final_lower[i] = basic_lower[i]
            else:
                final_lower[i] = final_lower[i-1]
                
            # SuperTrend
            if supertrend[i-1] == final_upper[i-1]:
                if close[i] > final_upper[i]:
                    supertrend[i] = final_lower[i]
                else:
                    supertrend[i] = final_upper[i]
            else:
                if close[i] < final_lower[i]:
                    supertrend[i] = final_upper[i]
                else:
                    supertrend[i] = final_lower[i]
                    
        return supertrend

    def calculate(self, data: pd.DataFrame) -> float:
        if data is None or data.empty:
            return 0.0
            
        # Parameters
        ema_period = self.params.get('ema_period', 50)
        st_length = self.params.get('st_length', 10)
        st_multiplier = self.params.get('st_multiplier', 3.0)
        
        # Calculate Indicators
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        
        # EMA
        try:
            ema = talib.EMA(close, timeperiod=ema_period)
        except Exception:
            return 0.0
            
        # SuperTrend
        try:
            st = self._calculate_supertrend(high, low, close, length=st_length, multiplier=st_multiplier)
        except Exception:
            return 0.0
            
        current_close = close[-1]
        current_ema = ema[-1]
        current_st = st[-1]
        
        score = 0.0
        
        # EMA Check
        if current_close > current_ema:
            score += 0.5
        else:
            score -= 0.5
            
        # SuperTrend Check
        # If Close > SuperTrend (and SuperTrend is acting as support/lower band) -> Bullish
        # Note: In my implementation, ST value switches between Upper and Lower band.
        # If ST < Close, it's bullish (Lower Band). If ST > Close, it's bearish (Upper Band).
        
        if current_close > current_st:
            score += 0.5
        else:
            score -= 0.5
            
        return score
            
        # SuperTrend Check
        if current_close > current_st:
            score += 0.5
        else:
            score -= 0.5
            
        return score
