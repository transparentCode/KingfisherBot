import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import logging

from app.models.sr_level_model import SRLevel, LevelType


logger = logging.getLogger(__name__)


class FibonacciLevel:
    """Represents a single Fibonacci level"""
    def __init__(
        self,
        price: float,
        ratio: float,
        timeframe: str,
        swing_high: float,
        swing_low: float,
        timestamp: datetime
    ):
        self.price = price
        self.ratio = ratio
        self.timeframe = timeframe
        self.swing_high = swing_high
        self.swing_low = swing_low
        self.timestamp = timestamp
        self.is_valid = True
    
    def invalidate(self):
        """Mark level as invalid (swing broken)"""
        self.is_valid = False


class FibonacciConfluence:
    """
    Calculates multi-timeframe Fibonacci retracements and 
    identifies confluence with S/R levels
    """
    
    def __init__(self, config: dict):
        self.config = config['fibonacci']
        self.fib_levels = self.config['levels']
        self.timeframes = self.config['timeframes']
        self.lookback = self.config['swing_lookback']
        
        # Storage for calculated Fibs per timeframe
        self.fib_data: Dict[str, List[FibonacciLevel]] = {}
        
        logger.info(f"Initialized Fibonacci Confluence with levels: {self.fib_levels}")
    
    def calculate_fibonacci_levels(
        self, 
        data: Dict[str, pd.DataFrame],
        current_timeframe: str
    ) -> Dict[str, List[FibonacciLevel]]:
        """
        Calculate Fibonacci retracements for all configured timeframes
        
        Args:
            data: Dict of DataFrames per timeframe
            current_timeframe: Primary timeframe for 'current' reference
        
        Returns:
            Dictionary of Fibonacci levels per timeframe
        """
        fib_results = {}
        
        for tf_key in self.timeframes:
            # Map 'current' to actual timeframe
            tf = current_timeframe if tf_key == "current" else tf_key
            
            if tf not in data:
                logger.warning(f"Timeframe {tf} not found in data")
                continue
            
            df = data[tf]
            
            if len(df) < self.lookback:
                logger.warning(f"Insufficient data for Fibonacci on {tf}")
                continue
            
            # Find swing high/low
            swing_high, swing_low, timestamp = self._find_swing_points(df)
            
            if swing_high is None or swing_low is None:
                continue
            
            # Calculate Fib levels
            fib_levels = self._calculate_fib_retracements(
                swing_high, swing_low, tf, timestamp
            )
            
            fib_results[tf] = fib_levels
        
        self.fib_data = fib_results
        return fib_results
    
    def _find_swing_points(
        self, 
        df: pd.DataFrame
    ) -> Tuple[Optional[float], Optional[float], Optional[datetime]]:
        """
        Find most recent swing high and swing low using lookback period
        """
        if len(df) < self.lookback:
            return None, None, None
        
        # Use last N bars
        recent_data = df.tail(self.lookback)
        
        swing_high = recent_data['high'].max()
        swing_low = recent_data['low'].min()
        
        # Get timestamp of swing formation
        timestamp = recent_data.index[-1] if isinstance(
            recent_data.index, pd.DatetimeIndex
        ) else datetime.now()
        
        return swing_high, swing_low, timestamp
    
    def _calculate_fib_retracements(
        self,
        swing_high: float,
        swing_low: float,
        timeframe: str,
        timestamp: datetime
    ) -> List[FibonacciLevel]:
        """
        Calculate Fibonacci retracement levels
        
        Formula: fib_price = swing_low + (swing_high - swing_low) * ratio
        """
        swing_range = swing_high - swing_low
        fib_levels = []
        
        for ratio in self.fib_levels:
            fib_price = swing_low + (swing_range * ratio)
            
            level = FibonacciLevel(
                price=fib_price,
                ratio=ratio,
                timeframe=timeframe,
                swing_high=swing_high,
                swing_low=swing_low,
                timestamp=timestamp
            )
            
            fib_levels.append(level)
        
        return fib_levels
    
    def check_confluence(
        self,
        sr_level: SRLevel,
        atr: float
    ) -> Tuple[int, List[Dict]]:
        """
        Check if S/R level aligns with any Fibonacci levels
        
        Args:
            sr_level: Support/Resistance level to check
            atr: Current ATR for tolerance calculation
        
        Returns:
            (match_count, list of matching Fib details)
        """
        tolerance = atr * self.config['tolerance_multiplier']
        matches = []
        match_count = 0
        
        for tf, fib_levels in self.fib_data.items():
            for fib in fib_levels:
                if not fib.is_valid:
                    continue
                
                # Check if SR level center is within tolerance of Fib level
                price_diff = abs(sr_level.center - fib.price)
                
                if price_diff <= tolerance:
                    match_count += 1
                    matches.append({
                        'timeframe': tf,
                        'fib_ratio': fib.ratio,
                        'fib_price': fib.price,
                        'distance': price_diff,
                        'swing_high': fib.swing_high,
                        'swing_low': fib.swing_low
                    })
                    
                    logger.debug(
                        f"Fib confluence: {sr_level.level_id} matches "
                        f"{fib.ratio:.3f} on {tf} at {fib.price:.2f}"
                    )
        
        return match_count, matches
    
    def invalidate_broken_swings(
        self,
        data: Dict[str, pd.DataFrame],
        break_threshold: float = 0.02
    ) -> None:
        """
        Invalidate Fibonacci levels when swing is broken
        
        Args:
            data: Current price data per timeframe
            break_threshold: % break beyond swing to invalidate (default 2%)
        """
        if not self.config['dynamic_invalidation']['enabled']:
            return
        
        threshold_pct = self.config['dynamic_invalidation']['break_threshold']
        
        for tf, fib_levels in self.fib_data.items():
            if tf not in data:
                continue
            
            current_price = data[tf]['close'].iloc[-1]
            
            for fib in fib_levels:
                if not fib.is_valid:
                    continue
                
                # Check if price broke swing high
                high_break = current_price > fib.swing_high * (1 + threshold_pct)
                
                # Check if price broke swing low
                low_break = current_price < fib.swing_low * (1 - threshold_pct)
                
                if high_break or low_break:
                    fib.invalidate()
                    logger.info(
                        f"Invalidated Fib {fib.ratio:.3f} on {tf} "
                        f"(swing broken at {current_price:.2f})"
                    )
    
    def get_all_levels(self) -> Dict[str, List[Dict]]:
        """Export all Fibonacci levels for visualization/analysis"""
        export = {}
        
        for tf, fib_levels in self.fib_data.items():
            export[tf] = [
                {
                    'price': fib.price,
                    'ratio': fib.ratio,
                    'swing_high': fib.swing_high,
                    'swing_low': fib.swing_low,
                    'is_valid': fib.is_valid,
                    'timestamp': fib.timestamp.isoformat()
                }
                for fib in fib_levels
            ]
        
        return export
