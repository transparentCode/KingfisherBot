from dataclasses import dataclass
from typing import Dict


@dataclass
class AssetIndicatorConfig:
    """Configuration for indicators per asset"""
    asset: str
    enabled: bool = True
    regime_adaptation_enabled: bool = True
    
    # Moving Average configs
    ma_configs: Dict[str, Dict] = None
    
    # SuperTrend configs  
    supertrend_configs: Dict[str, Dict] = None
    
    # Oscillator configs
    oscillator_configs: Dict[str, Dict] = None
    
    # Timeframe overrides
    timeframe_overrides: Dict[str, list] = None
    
    def __post_init__(self):
        if self.ma_configs is None:
            self.ma_configs = {
                'fast': {'period': 14, 'source': 'close'},
                'medium': {'period': 21, 'source': 'close'}, 
                'slow': {'period': 50, 'source': 'close'}
            }
        if self.supertrend_configs is None:
            self.supertrend_configs = {
                'default': {'atr_len': 10, 'atr_mult': 3.0, 'span': 14}
            }
        if self.oscillator_configs is None:
            self.oscillator_configs = {
                'rsi': {'period': 14, 'gaussian_weights': True}
            }
        if self.timeframe_overrides is None:
            self.timeframe_overrides = {}