import pandas as pd
import talib
from app.signals.base_factor import BaseSignalFactor

class MomentumFactor(BaseSignalFactor):
    """
    Returns +1 if RSI crosses 50 up, etc.
    """
    def __init__(self):
        super().__init__("MomentumFactor")

    def calculate(self, data: pd.DataFrame) -> float:
        if data is None or data.empty:
            return 0.0
            
        rsi_period = self.params.get('rsi_period', 14)
        
        try:
            rsi = talib.RSI(data['close'].values, timeperiod=rsi_period)
        except Exception:
            return 0.0
            
        if rsi is None or len(rsi) == 0:
            return 0.0
            
        current_rsi = rsi[-1]
        
        # Simple Logic:
        # > 60 -> Bullish (+0.5 to +1.0)
        # < 40 -> Bearish (-0.5 to -1.0)
        # 40-60 -> Neutral
        
        if current_rsi > 60:
            # Scale 60-100 to 0.5-1.0
            return 0.5 + (current_rsi - 60) / 40 * 0.5
        elif current_rsi < 40:
            # Scale 40-0 to -0.5 to -1.0
            return -0.5 - (40 - current_rsi) / 40 * 0.5
        else:
            # Neutral zone
            return 0.0
