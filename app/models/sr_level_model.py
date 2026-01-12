from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
import uuid
import numpy as np 


class LevelType(Enum):
    SUPPORT = "SUPPORT"
    RESISTANCE = "RESISTANCE"


class LevelStatus(Enum):
    ACTIVE = "ACTIVE"
    BROKEN = "BROKEN"
    FLIPPED = "FLIPPED"


@dataclass
class PivotPoint:
    price: float
    index: int
    timestamp: datetime
    volume: float
    timeframe: str
    is_high: bool  # True = resistance, False = support


@dataclass
class TouchEvent:
    timestamp: datetime
    price: float
    volume: float
    wick_size: float
    body_size: float
    is_rejection: bool
    bar_index: int


@dataclass
class SRLevel:
    level_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    level_type: LevelType = LevelType.SUPPORT
    center: float = 0.0
    upper_bound: float = 0.0
    lower_bound: float = 0.0
    zone_width: float = 0.0
    strength: float = 0.0
    touch_count: int = 0
    rejection_count: int = 0
    avg_touch_volume: float = 0.0
    timeframes: List[str] = field(default_factory=list)
    timeframe_str: str = ""
    confluence_scores: Dict[str, float] = field(default_factory=dict)
    formation_time: Optional[datetime] = None
    last_touch_time: Optional[datetime] = None
    age_bars: int = 0
    status: LevelStatus = LevelStatus.ACTIVE
    pivots: List[PivotPoint] = field(default_factory=list)
    touches: List[TouchEvent] = field(default_factory=list)
    atr_at_formation: float = 0.0
    avg_volume_at_formation: float = 0.0
    
    # Additional fields for confluence
    liquidity_score: float = 0.0
    priority: str = "LOW"  # CRITICAL, HIGH, MEDIUM, LOW
    confluence_metadata: Dict = field(default_factory=dict)
    
    def add_touch(self, touch: TouchEvent) -> None:
        """Add a touch event"""
        
        if self.touches and (touch.bar_index - self.touches[-1].bar_index) <= 1:
         # Consecutive touch - treat as same event
         return 

        self.touches.append(touch)
        self.last_touch_time = touch.timestamp
        
        # Update average touch volume
        total_volume = sum(t.volume for t in self.touches)
        self.avg_touch_volume = total_volume / len(self.touches)
    
    def check_breakout(
        self,
        close_price: float,
        volume: float,
        avg_volume: float,
        volume_multiplier: float
    ) -> bool:
        """
        Check if level was broken
        
        Returns:
            True if breakout detected
        """
        volume_confirmed = volume >= (avg_volume * volume_multiplier)
        
        if self.level_type == LevelType.SUPPORT:
            # Support broken: close below lower bound
            if close_price < self.lower_bound and volume_confirmed:
                self.status = LevelStatus.BROKEN
                return True
        else:
            # Resistance broken: close above upper bound
            if close_price > self.upper_bound and volume_confirmed:
                self.status = LevelStatus.BROKEN
                return True
        
        return False
    
    def flip(self) -> None:
        """Flip support to resistance or vice versa"""
        if self.level_type == LevelType.SUPPORT:
            self.level_type = LevelType.RESISTANCE
        else:
            self.level_type = LevelType.SUPPORT
        
        self.status = LevelStatus.FLIPPED
    
    def update_strength(
        self,
        tf_weights: Dict[str, float],
        touch_bonus: float,
        age_lambda: float
    ) -> None:
        """
        Calculate strength with confluence formula
        
        strength = Σ(timeframe_weight) × (1 + touch_bonus × touches) × e^(-λ × age)
        """
        # Base strength from timeframes
        base_strength = sum(
            tf_weights.get(tf, 1.0) for tf in self.timeframes
        )
        
        # Touch bonus (capped)
        max_bonus = 2.0
        touch_multiplier = min(1 + (touch_bonus * self.touch_count), 1 + max_bonus)
        
        # Age decay
        age_decay = np.exp(-age_lambda * self.age_bars)
        
        self.strength = base_strength * touch_multiplier * age_decay
    
    def to_dict(self) -> dict:
        """Export as dictionary"""
        return {
            'level_id': self.level_id,
            'type': self.level_type.value,
            'center': round(self.center, 4),
            'upper_bound': round(self.upper_bound, 4),
            'lower_bound': round(self.lower_bound, 4),
            'zone_width': round(self.zone_width, 4),
            'strength': round(self.strength, 2),
            'liquidity_score': round(self.liquidity_score, 2),
            'priority': self.priority,
            'touch_count': self.touch_count,
            'rejection_count': self.rejection_count,
            'timeframes': self.timeframes,
            'status': self.status.value,
            'formation_time': self.formation_time.isoformat() if self.formation_time else None,
            'last_touch_time': self.last_touch_time.isoformat() if self.last_touch_time else None,
            'age_bars': self.age_bars,
            'confluence_metadata': self.confluence_metadata
        }