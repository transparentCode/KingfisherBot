import logging
from typing import List, Dict
from app.models.sr_level_model import SRLevel


logger = logging.getLogger(__name__)


class LiquidityScorer:
    """
    Calculates liquidity scores and assigns priority levels
    based on multi-factor confluence
    """
    
    def __init__(self, config: dict):
        self.config = config['priority_scoring']
        self.components = self.config['components']
        self.thresholds = self.config['thresholds']
        
        logger.info("Initialized Liquidity Scorer")
    
    def calculate_score(
        self,
        level: SRLevel,
        fib_matches: int = 0,
        ema_weight: float = 0.0,
        round_number_weight: float = 0.0,
        volume_quality: float = 1.0
    ) -> float:
        """
        Calculate comprehensive liquidity score
        
        Formula:
        score = Σ(component_value × component_weight)
        """
        score = 0.0
        
        # Base strength
        score += level.strength * self.components['base_strength']
        
        # Touch count
        score += level.touch_count * self.components['touch_count']
        
        # Rejection count
        score += level.rejection_count * self.components['rejection_count']
        
        # Timeframe confluence
        tf_count = len(level.timeframes)
        score += tf_count * self.components['timeframe_count']
        
        # Fibonacci confluence
        score += fib_matches * self.components['fibonacci_bonus']
        
        # EMA confluence
        score += ema_weight * self.components['ema_bonus']
        
        # Round number
        score += round_number_weight * self.components['round_number_bonus']
        
        # Volume quality
        score += volume_quality * self.components['volume_quality']
        
        return round(score, 2)
    
    def assign_priority(self, score: float) -> str:
        """
        Assign priority tier based on liquidity score
        
        Returns:
            CRITICAL, HIGH, MEDIUM, or LOW
        """
        if score >= self.thresholds['critical']:
            return "CRITICAL"
        elif score >= self.thresholds['high']:
            return "HIGH"
        elif score >= self.thresholds['medium']:
            return "MEDIUM"
        else:
            return "LOW"
    
    def calculate_volume_quality(
        self,
        level: SRLevel
    ) -> float:
        """
        Calculate volume quality score (0-2 scale)
        
        Based on:
        - avg_touch_volume vs market average
        - Consistency of volume across touches
        """
        if level.avg_volume_at_formation == 0:
            return 1.0  # Neutral
        
        volume_ratio = level.avg_touch_volume / level.avg_volume_at_formation
        
        # Score: >1.5 = excellent (2.0), 1.0-1.5 = good (1.5), <1.0 = weak (0.5)
        if volume_ratio >= 1.5:
            return 2.0
        elif volume_ratio >= 1.2:
            return 1.5
        elif volume_ratio >= 1.0:
            return 1.0
        else:
            return 0.5
