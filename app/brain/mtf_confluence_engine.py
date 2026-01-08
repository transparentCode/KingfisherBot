import logging
from typing import Dict, Any, List, Tuple

from app.brain.models import MTFConfluenceResult

logger = logging.getLogger(__name__)


class MTFConfluenceEngine:
    """
    Multi-Timeframe Confluence Engine.
    
    Analyzes regime alignment across multiple timeframes and returns:
    - Weighted trend score
    - Conflict level (timeframe disagreement)
    - Trade permission based on alignment
    
    Key Insight: Higher timeframes get more weight (4H > 1H > 30m > 15m).
    """
    
    # Timeframe weights (higher TF = more authority)
    TF_WEIGHTS = {
        '4h': 0.35,
        '1h': 0.30,
        '30m': 0.20,
        '15m': 0.15
    }
    
    # Thresholds
    TREND_THRESHOLD = 0.15      # Score above this = directional bias
    CONFLICT_THRESHOLD = 0.5   # Conflict above this = no trade
    STRONG_TREND = 0.3         # Score above this = strong conviction
    
    # Feature prefixes by timeframe
    TF_PREFIXES = {
        '4h': 'htf_',
        '1h': 'mtf_',
        '30m': 'ttf_',
        '15m': 'ttf_'
    }

    def calculate_confluence(self, features: Dict[str, Any]) -> MTFConfluenceResult:
        """
        Calculate multi-timeframe confluence from extracted features.
        
        Args:
            features: Dict from FeatureExtractor containing {prefix}trend_score etc.
            
        Returns:
            MTFConfluenceResult with score, conflict, confidence, direction
        """
        scores = []
        weights = []
        tf_scores = {}
        
        for tf, weight in self.TF_WEIGHTS.items():
            prefix = self.TF_PREFIXES.get(tf, '')
            trend_key = f'{prefix}trend_score'
            
            trend_score = features.get(trend_key)
            
            if trend_score is not None:
                scores.append(float(trend_score))
                weights.append(weight)
                tf_scores[tf] = float(trend_score)
        
        # No data = neutral
        if not scores:
            return MTFConfluenceResult(
                score=0.0,
                conflict=1.0,
                confidence=0.0,
                direction='NEUTRAL',
                tf_scores={}
            )
        
        # 1. Weighted average of trend scores
        total_weight = sum(weights)
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
        
        # 2. Conflict detection (are timeframes disagreeing?)
        directions = self._get_directions(scores)
        conflict = self._calculate_conflict(directions)
        
        # 3. Determine dominant direction
        if weighted_score > self.TREND_THRESHOLD:
            direction = 'LONG'
        elif weighted_score < -self.TREND_THRESHOLD:
            direction = 'SHORT'
        else:
            direction = 'NEUTRAL'
        
        # 4. Confidence = inverse of conflict
        confidence = max(0.0, 1.0 - conflict)
        
        # 5. Boost confidence if all timeframes agree strongly
        if conflict < 0.2 and abs(weighted_score) > self.STRONG_TREND:
            confidence = min(1.0, confidence + 0.1)
        
        return MTFConfluenceResult(
            score=weighted_score,
            conflict=conflict,
            confidence=confidence,
            direction=direction,
            tf_scores=tf_scores
        )

    def _get_directions(self, scores: List[float]) -> List[int]:
        """Convert scores to direction indicators: +1 (long), -1 (short), 0 (neutral)."""
        directions = []
        for score in scores:
            if score > self.TREND_THRESHOLD:
                directions.append(1)
            elif score < -self.TREND_THRESHOLD:
                directions.append(-1)
            else:
                directions.append(0)
        return directions

    def _calculate_conflict(self, directions: List[int]) -> float:
        """
        Calculate conflict level based on direction disagreement.
        
        0.0 = All timeframes agree
        0.5 = Mixed signals (some neutral)
        1.0 = Direct opposition (long vs short)
        """
        if not directions:
            return 1.0
        
        unique_dirs = set(directions)
        
        # All same direction = no conflict
        if len(unique_dirs) == 1:
            return 0.0
        
        # Check for direct opposition (both +1 and -1 present)
        has_long = 1 in unique_dirs
        has_short = -1 in unique_dirs
        
        if has_long and has_short:
            return 1.0  # Maximum conflict
        
        # Mixed with neutral = moderate conflict
        return 0.4

    def can_trade(self, features: Dict[str, Any], direction: str) -> Tuple[bool, str]:
        """
        Check if a trade is allowed in the given direction.
        
        Args:
            features: Feature dict from FeatureExtractor
            direction: "LONG" or "SHORT"
            
        Returns:
            (allowed: bool, reason: str)
        """
        confluence = self.calculate_confluence(features)
        
        # Rule 1: High conflict = No trade
        if confluence.conflict > self.CONFLICT_THRESHOLD:
            return False, f"MTF conflict too high ({confluence.conflict:.2f})"
        
        # Rule 2: Direction must match or be neutral
        if direction == "LONG" and confluence.direction == "SHORT":
            return False, "MTF bias is SHORT, blocking LONG"
        if direction == "SHORT" and confluence.direction == "LONG":
            return False, "MTF bias is LONG, blocking SHORT"
        
        # Rule 3: Check 4H veto (highest TF has veto power)
        htf_trend = features.get('htf_trend_score', 0.0)
        if direction == "LONG" and htf_trend < -self.STRONG_TREND:
            return False, f"4H trend strongly bearish ({htf_trend:.2f}), blocking LONG"
        if direction == "SHORT" and htf_trend > self.STRONG_TREND:
            return False, f"4H trend strongly bullish ({htf_trend:.2f}), blocking SHORT"
        
        return True, "Trade allowed by MTF confluence"

    def get_position_scale(self, confluence: MTFConfluenceResult) -> float:
        """
        Get position size scale factor based on confluence quality.
        
        Returns:
            0.0 to 1.0 multiplier for position size
        """
        if confluence.conflict > 0.5:
            return 0.0  # No trade
        elif confluence.conflict > 0.3:
            return 0.5  # Half size
        elif confluence.conflict > 0.1:
            return 0.75  # 3/4 size
        else:
            return 1.0  # Full size
