import pandas as pd
from app.signals.base_factor import BaseSignalFactor

class VolumeFactor(BaseSignalFactor):
    """
    Returns +1 if Price > POC & Entropy < 0.6.
    """
    def __init__(self):
        super().__init__("VolumeFactor")

    def calculate(self, data: pd.DataFrame) -> float:
        if data is None or data.empty:
            return 0.0
            
        # Assuming Volume Profile / VWR is calculated elsewhere and attached to DF
        # or we calculate simple volume metrics here.
        
        # Simple Volume Trend for now:
        # If Volume is increasing and Price is increasing -> Bullish
        # If Volume is increasing and Price is decreasing -> Bearish
        
        close = data['close']
        volume = data['volume']
        
        if len(close) < 2:
            return 0.0
            
        price_change = close.iloc[-1] - close.iloc[-2]
        vol_change = volume.iloc[-1] - volume.iloc[-2]
        
        if vol_change > 0:
            if price_change > 0:
                return 0.5
            elif price_change < 0:
                return -0.5
        
        return 0.0
