import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

from app.models.sr_level_model import SRLevel


logger = logging.getLogger(__name__)


class EMAConfluence:
    """
    Calculates multi-timeframe EMA levels and checks confluence
    Can optionally create EMA-based dynamic S/R zones
    """
    
    def __init__(self, config: dict):
        self.config = config['ema_confluence']
        self.periods = self.config['periods']
        self.timeframes = self.config['timeframes']
        
        # Storage for EMA values per TF
        self.ema_data: Dict[str, Dict[int, float]] = {}
        
        # Track recent crosses
        self.recent_crosses: Dict[str, List[Dict]] = {}
        
        logger.info(
            f"Initialized EMA Confluence with periods: {self.periods}, "
            f"timeframes: {self.timeframes}"
        )
    
    def calculate_emas(
        self,
        data: Dict[str, pd.DataFrame],
        current_timeframe: str
    ) -> Dict[str, Dict[int, float]]:
        """
        Calculate EMAs for all configured periods and timeframes
        
        Args:
            data: Dict of DataFrames per timeframe
            current_timeframe: Primary timeframe for 'current' reference
        
        Returns:
            Nested dict: {timeframe: {period: ema_value}}
        """
        ema_results = {}
        
        for tf_key in self.timeframes:
            tf = current_timeframe if tf_key == "current" else tf_key
            
            if tf not in data:
                logger.warning(f"Timeframe {tf} not found in data")
                continue
            
            df = data[tf]
            ema_values = {}
            
            for period in self.periods:
                if len(df) < period:
                    logger.warning(f"Insufficient data for EMA({period}) on {tf}")
                    continue
                
                ema = df['close'].ewm(span=period, adjust=False).mean().iloc[-1]
                ema_values[period] = ema
            
            ema_results[tf] = ema_values
            
            # Detect recent crosses
            self._detect_crosses(df, tf)
        
        self.ema_data = ema_results
        return ema_results
    
    def _detect_crosses(
        self,
        df: pd.DataFrame,
        timeframe: str
    ) -> None:
        """
        Detect Golden/Death Cross (50/200 EMA crossovers)
        """
        if not self.config['golden_death_cross']['enabled']:
            return
        
        if 50 not in self.periods or 200 not in self.periods:
            return
        
        lookback = self.config['golden_death_cross']['lookback_bars']
        
        if len(df) < 200 + lookback:
            return
        
        # Calculate EMAs
        ema_50 = df['close'].ewm(span=50, adjust=False).mean()
        ema_200 = df['close'].ewm(span=200, adjust=False).mean()
        
        # Check recent bars for cross
        recent_ema50 = ema_50.tail(lookback)
        recent_ema200 = ema_200.tail(lookback)
        
        crosses = []
        
        for i in range(1, len(recent_ema50)):
            prev_50 = recent_ema50.iloc[i-1]
            curr_50 = recent_ema50.iloc[i]
            prev_200 = recent_ema200.iloc[i-1]
            curr_200 = recent_ema200.iloc[i]
            
            # Golden Cross (50 crosses above 200)
            if prev_50 <= prev_200 and curr_50 > curr_200:
                crosses.append({
                    'type': 'golden',
                    'timestamp': recent_ema50.index[i],
                    'price': df['close'].iloc[recent_ema50.index[i]]
                })
            
            # Death Cross (50 crosses below 200)
            elif prev_50 >= prev_200 and curr_50 < curr_200:
                crosses.append({
                    'type': 'death',
                    'timestamp': recent_ema50.index[i],
                    'price': df['close'].iloc[recent_ema50.index[i]]
                })
        
        if crosses:
            self.recent_crosses[timeframe] = crosses
            logger.info(f"Detected {len(crosses)} EMA crosses on {timeframe}")
    
    def check_confluence(
        self,
        sr_level: SRLevel,
        atr: float
    ) -> Tuple[float, List[Dict]]:
        """
        Check if S/R level aligns with any EMA levels
        
        Args:
            sr_level: Support/Resistance level to check
            atr: Current ATR for tolerance calculation
        
        Returns:
            (total_weight, list of matching EMA details)
        """
        tolerance = atr * self.config['tolerance_multiplier']
        matches = []
        total_weight = 0.0
        
        weights = self.config['weights']
        
        for tf, ema_values in self.ema_data.items():
            for period, ema_price in ema_values.items():
                # Check proximity
                price_diff = abs(sr_level.center - ema_price)
                
                if price_diff <= tolerance:
                    weight = weights.get(period, 0.0)
                    total_weight += weight
                    
                    matches.append({
                        'timeframe': tf,
                        'period': period,
                        'ema_price': ema_price,
                        'distance': price_diff,
                        'weight': weight
                    })
                    
                    logger.debug(
                        f"EMA confluence: {sr_level.level_id} matches "
                        f"EMA({period}) on {tf} at {ema_price:.2f}"
                    )
        
        # Add bonus for recent crosses
        if self.config['golden_death_cross']['enabled']:
            cross_bonus = self._check_cross_proximity(sr_level, tolerance)
            total_weight += cross_bonus
        
        return total_weight, matches
    
    def _check_cross_proximity(
        self,
        sr_level: SRLevel,
        tolerance: float
    ) -> float:
        """
        Check if level near recent Golden/Death Cross
        """
        bonus = 0.0
        cross_weight = self.config['golden_death_cross']['bonus_weight']
        
        for tf, crosses in self.recent_crosses.items():
            for cross in crosses:
                if abs(sr_level.center - cross['price']) <= tolerance:
                    bonus += cross_weight
                    logger.info(
                        f"Cross bonus: {sr_level.level_id} near {cross['type']} "
                        f"cross on {tf}"
                    )
        
        return bonus
    
    def create_ema_zones(
        self,
        atr: float,
        zone_width_multiplier: float = 0.3
    ) -> List[SRLevel]:
        """
        Optional: Create explicit S/R zones at EMA levels
        (For use when create_ema_zones is enabled in config)
        """
        if not self.config.get('create_ema_zones', False):
            return []
        
        # Implementation for creating EMA-based SRLevel objects
        # Not commonly used - EMAs typically used for confluence only
        pass
    
    def get_all_emas(self) -> Dict[str, Dict[int, float]]:
        """Export all EMA values"""
        return self.ema_data


# ============================================================================
# Step 4C: Round Number Detector
# ============================================================================

class RoundNumberDetector:
    """
    Detects psychological round number levels
    """
    
    def __init__(self, config: dict):
        self.config = config['round_numbers']
        self.rules = self.config['rules']
        self.min_price = self.config.get('min_price', 10.0)
        
        logger.info("Initialized Round Number Detector")
    
    def check_confluence(
        self,
        sr_level: SRLevel
    ) -> Tuple[float, Optional[Dict]]:
        """
        Check if S/R level is near a round number
        
        Returns:
            (weight, round_number_info)
        """
        price = sr_level.center
        
        # Skip if below minimum price
        if price < self.min_price:
            return 0.0, None
        
        tolerance_pct = self.config['tolerance_pct']
        tolerance = price * tolerance_pct
        
        # Check each rule
        for rule in self.rules:
            rule_type = rule['type']
            weight = rule['weight']
            
            if rule_type == 'integer':
                # Check if close to integer
                rounded = round(price)
                if abs(price - rounded) <= tolerance:
                    return weight, {
                        'type': 'integer',
                        'round_price': rounded,
                        'distance': abs(price - rounded)
                    }
            
            elif rule_type == 'ends_with':
                # Check if ends with specific values
                for value in rule['values']:
                    # Check different scales
                    for scale in [1, 10, 100, 1000, 10000]:
                        round_price = round(price / scale) * scale
                        
                        if str(int(round_price)).endswith(str(value)):
                            if abs(price - round_price) <= tolerance:
                                return weight, {
                                    'type': 'ends_with',
                                    'value': value,
                                    'round_price': round_price,
                                    'distance': abs(price - round_price)
                                }
            
            elif rule_type == 'decimals':
                # Check specific decimal places (e.g., 100.00)
                places = rule['places']
                round_price = round(price, places)
                
                if abs(price - round_price) <= tolerance:
                    return weight, {
                        'type': 'decimals',
                        'places': places,
                        'round_price': round_price,
                        'distance': abs(price - round_price)
                    }
        
        return 0.0, None
