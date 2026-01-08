from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class StrategySignal:
    """
    Standardized output from a strategy.
    """
    strategy_name: str
    signal: int  # 1 (Buy), -1 (Sell), 0 (Neutral)
    confidence: float  # 0.0 to 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AggregatedSignal:
    """
    Combined signal from the Brain.
    """
    asset: str
    action: str  # 'BUY', 'SELL', 'NEUTRAL'
    confidence: float
    contributing_strategies: Dict[str, StrategySignal] = field(default_factory=dict)
