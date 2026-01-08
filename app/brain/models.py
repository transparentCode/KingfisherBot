from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List
from enum import Enum


class SetupType(Enum):
    """Type of trading setup identified."""
    TREND_PULLBACK = "trend_pullback"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    NONE = "none"


class Direction(Enum):
    """Trade direction."""
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


@dataclass
class TradeSignal:
    """
    Output of the BotBrain. A complete trade setup recommendation.
    """
    timestamp: datetime
    symbol: str
    direction: str          # "LONG", "SHORT", "NEUTRAL"
    setup_type: str         # "TREND_PULLBACK", "MEAN_REVERSION", "BREAKOUT"
    timeframe: str          # Execution timeframe (e.g., "15m")
    
    # Price Levels
    entry_price: float
    stop_loss: float
    take_profit: float
    
    # Quality Metrics
    confidence: float       # 0.0 to 1.0 (combines indicator + confluence quality)
    risk_reward_ratio: float
    
    # MTF Confluence Context
    mtf_score: float = 0.0          # Weighted MTF alignment (-1 to +1)
    mtf_conflict: float = 0.0       # Timeframe disagreement (0 = aligned, 1 = opposed)
    
    # Debug info (Why did we take this trade?)
    reasoning: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'direction': self.direction,
            'setup_type': self.setup_type,
            'timeframe': self.timeframe,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'confidence': self.confidence,
            'risk_reward_ratio': self.risk_reward_ratio,
            'mtf_score': self.mtf_score,
            'mtf_conflict': self.mtf_conflict,
            'reasoning': self.reasoning
        }


@dataclass
class MTFConfluenceResult:
    """
    Result of multi-timeframe confluence analysis.
    """
    score: float            # Weighted average trend score (-1 to +1)
    conflict: float         # Disagreement level (0 = aligned, 1 = max conflict)
    confidence: float       # 1.0 - conflict
    direction: str          # "LONG", "SHORT", "NEUTRAL"
    
    # Per-timeframe breakdown
    tf_scores: Dict[str, float] = field(default_factory=dict)
    
    def allows_trade(self, direction: str) -> bool:
        """Check if confluence allows a trade in the given direction."""
        if self.conflict > 0.5:
            return False
        if direction == "LONG" and self.direction == "SHORT":
            return False
        if direction == "SHORT" and self.direction == "LONG":
            return False
        return True
