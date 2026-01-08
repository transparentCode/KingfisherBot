from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

class TradingStyle(Enum):
    SCALPING = "scalping"
    DAY_TRADING = "day_trading"
    SWING = "swing"
    POSITION = "position"

class StopLossType(Enum):
    FIXED_PERCENT = "fixed_percent"
    ATR = "atr"
    MARKET_STRUCTURE = "market_structure" # Recent Swing High/Low

class TakeProfitType(Enum):
    FIXED_RR = "fixed_rr" # Risk:Reward Ratio
    ATR_MULTIPLIER = "atr_multiplier"
    MULTI_TIER = "multi_tier"

@dataclass
class RiskProfile:
    """
    Configuration for Risk Management based on Trading Style.
    """
    style: TradingStyle
    max_risk_per_trade: float = 0.01  # 1% of equity
    risk_reward_ratio: float = 2.0    # Target 1:2
    
    # Stop Loss Config
    sl_type: StopLossType = StopLossType.ATR
    sl_atr_multiplier: float = 1.5
    sl_fixed_pct: float = 0.02
    
    # Take Profit Config
    tp_type: TakeProfitType = TakeProfitType.MULTI_TIER
    tp_tiers: List[float] = field(default_factory=lambda: [0.5, 0.3, 0.2]) # 50% at TP1, 30% at TP2, 20% at TP3
    tp_rr_multipliers: List[float] = field(default_factory=lambda: [1.5, 2.5, 4.0]) # TP1=1.5R, TP2=2.5R, TP3=4R
    
    # RL Integration
    use_dynamic_adjustment: bool = False # If True, RL agent can override parameters

    # Pyramiding (Future Support)
    pyramiding_enabled: bool = False
    max_pyramiding_adds: int = 0
    pyramiding_threshold_r: float = 1.0 # Add to position every 1R move in favor
    
    # Leverage (Style Default)
    max_leverage: float = 1.0

@dataclass
class AssetRiskSettings:
    """
    Per-Asset Risk Constraints.
    Overrides or caps the global/style settings.
    """
    symbol: str
    max_leverage: float = 20.0 # Exchange limit or user safety limit
    max_position_size_usd: float = 100000.0
    min_position_size_usd: float = 10.0
    leverage_mode: str = "ISOLATED" # CROSS or ISOLATED

@dataclass
class TradePlan:
    """
    The output of the Risk Manager. A fully calculated trade setup.
    """
    symbol: str
    direction: str # "LONG" or "SHORT"
    entry_price: float
    
    # Risk Calculations
    stop_loss_price: float
    risk_amount: float # Currency amount at risk
    position_size: float # Quantity of asset
    
    # Take Profits
    take_profit_prices: List[float]
    take_profit_quantities: List[float]
    
    # Metadata
    risk_reward_ratio: float
    style: TradingStyle
    
    # Fields with defaults must come LAST
    leverage: float = 1.0
    confidence_score: float = 1.0 # Derived from MTF/Confluence (0.5 to 1.5)
    notes: str = ""
