from typing import Dict, Any
from app.enums.MarketRegime import MarketRegime

class ConfluenceEngine:
    """
    Layer B: The Weighted Cortex.
    Aggregates signals with dynamic weighting based on Market Regime.
    """
    
    def calculate_score(self, signals: Dict[str, float], regime_metrics: Dict[str, Any]) -> float:
        """
        Calculate final score (-1.0 to 1.0) based on signals and regime.
        """
        # 1. Base Weights
        weights = {
            'TrendFactor': 0.4,
            'MomentumFactor': 0.3,
            'VolumeFactor': 0.3
        }
        
        # 2. DYNAMIC RE-WEIGHTING (The "Adaptive" Part)
        # Extract metrics. Assuming regime_metrics contains 'hurst', 'regime'
        hurst = regime_metrics.get('hurst', 0.5)
        regime = regime_metrics.get('regime', MarketRegime.CONSOLIDATION)
        
        # Hurst-based adaptation
        if hurst > 0.6: # Strong Trend
            weights['TrendFactor'] = 0.7      # Trust trend more
            weights['MomentumFactor'] = 0.1   # Ignore overbought/sold
            weights['VolumeFactor'] = 0.2
        elif hurst < 0.4: # Mean Reversion
            weights['TrendFactor'] = 0.1      # Ignore moving averages
            weights['MomentumFactor'] = 0.6   # Trust oscillators
            weights['VolumeFactor'] = 0.3     # Trust Value Area edges
            
        # Regime-based Strict Filters (Veto Power)
        # If Strategic Regime is Bearish, block Longs (Score > 0 becomes 0)
        # If Strategic Regime is Bullish, block Shorts (Score < 0 becomes 0)
        
        # Normalize weights to sum to 1.0 (optional but good practice)
        total_weight = sum(weights.get(k, 0) for k in signals.keys())
        if total_weight > 0:
            for k in weights:
                weights[k] /= total_weight

        # 3. Calculate Weighted Sum
        raw_score = 0.0
        used_weight = 0.0
        
        for name, score in signals.items():
            w = weights.get(name, 0.0)
            raw_score += score * w
            used_weight += w
            
        if used_weight == 0:
            return 0.0
            
        final_score = raw_score # Already normalized if weights sum to 1
        
        # 4. Apply Strategic Filters
        if regime == MarketRegime.TRENDING_BEAR:
            if final_score > 0:
                final_score = 0.0 # Veto Longs
        elif regime == MarketRegime.TRENDING_BULL:
            if final_score < 0:
                final_score = 0.0 # Veto Shorts
                
        return final_score
