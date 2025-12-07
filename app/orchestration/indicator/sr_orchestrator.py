import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging
from datetime import datetime

from app.indicators.support_and_resistance import SupportResistanceDetector
from app.indicators.fibonacci_confluence import FibonacciConfluence
from app.indicators.ema_confluence import EMAConfluence, RoundNumberDetector
from app.indicators.liquidity_scorer import LiquidityScorer
from app.models.sr_level_model import SRLevel, LevelType
from app.utils.config_loader import ConfigLoader


logger = logging.getLogger(__name__)


class SupportResistanceOrchestrator:
    """
    Orchestrates multi-timeframe Support & Resistance detection
    with advanced confluence scoring
    """
    
    def __init__(
        self,
        timeframes: List[str],
        config_path: str = "config/sr_config.yaml"
    ):
        # Load configuration
        self.config = ConfigLoader.load(config_path)
        
        # Apply asset-specific overrides
        self._apply_asset_overrides()
        
        self.timeframes = timeframes
        
        # Create detectors for each timeframe
        self.detectors: Dict[str, SupportResistanceDetector] = {
            tf: SupportResistanceDetector(tf, self.config)
            for tf in timeframes
        }
        
        # Initialize confluence components
        self.fib_confluence = FibonacciConfluence(self.config)
        self.ema_confluence = EMAConfluence(self.config)
        self.round_number_detector = RoundNumberDetector(self.config)
        self.liquidity_scorer = LiquidityScorer(self.config)
        
        # Merged levels cache
        self.merged_levels: List[SRLevel] = []
        
        # Last update timestamp
        self.last_update: Optional[datetime] = None
        
        logger.info(
            f"Initialized SR Orchestrator with {len(timeframes)} timeframes: "
            f"{timeframes}"
        )
    
    def _apply_asset_overrides(self) -> None:
        """Apply asset-specific configuration overrides"""
        asset_type = self.config['general']['asset_type']
        
        if asset_type not in self.config.get('asset_overrides', {}):
            return
        
        overrides = self.config['asset_overrides'][asset_type]
        
        # Deep merge overrides (simplified - production would use recursive merge)
        for section, params in overrides.items():
            if section in self.config:
                self.config[section].update(params)
        
        logger.info(f"Applied {asset_type} configuration overrides")
    
    def update(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        Main update method - processes MTF data
        
        Args:
            data: Dict mapping timeframe to OHLCV DataFrame
                  e.g., {"5m": df_5m, "15m": df_15m, "1h": df_1h}
        """
        logger.info(f"Starting MTF update with {len(data)} timeframes")
        
        # Phase 1: Update individual detectors
        for tf, df in data.items():
            if tf in self.detectors:
                self.detectors[tf].update(df)
            else:
                logger.warning(f"No detector configured for {tf}")
        
        # Phase 2: Calculate confluence components
        primary_tf = self.timeframes[0] if self.timeframes else "15m"
        
        # Fibonacci
        if self.config['fibonacci']['enabled']:
            self.fib_confluence.calculate_fibonacci_levels(data, primary_tf)
            self.fib_confluence.invalidate_broken_swings(data)
        
        # EMA
        if self.config['ema_confluence']['enabled']:
            self.ema_confluence.calculate_emas(data, primary_tf)
        
        # Phase 3: Merge levels across timeframes
        self._merge_levels(data, primary_tf)
        
        # Phase 4: Apply confluence scoring
        self._apply_confluence_scoring(data, primary_tf)
        
        # Phase 5: Filter by minimum score
        self._filter_weak_levels()
        
        self.last_update = datetime.now()
        
        logger.info(
            f"Update complete: {len(self.merged_levels)} merged levels "
            f"({self._count_by_priority()})"
        )
    
    def _merge_levels(
        self,
        data: Dict[str, pd.DataFrame],
        primary_tf: str
    ) -> None:
        """
        Merge levels across timeframes with intelligent deduplication
        """
        all_levels = []
        
        # Collect all levels from detectors
        for tf, detector in self.detectors.items():
            levels = detector.get_active_levels()
            all_levels.extend(levels)
        
        logger.debug(f"Collected {len(all_levels)} total levels before merging")
        
        # Sort by strength (descending) for keep_strongest logic
        all_levels.sort(key=lambda x: x.strength, reverse=True)
        
        merged = []
        price_tol_pct = self.config['merging']['price_tolerance_pct']
        overlap_threshold = self.config['merging'].get('overlap_threshold_pct', 0.5)
        
        for level in all_levels:
            # Check for duplicates/overlaps
            match = None
            
            for existing in merged:
                if level.level_type != existing.level_type:
                    continue
                
                # Check price proximity
                price_diff_pct = abs(level.center - existing.center) / existing.center
                
                if price_diff_pct < price_tol_pct:
                    match = existing
                    break
                
                # Check zone overlap
                if self.config['merging'].get('merge_overlapping_zones', True):
                    overlap = self._calculate_overlap(level, existing)
                    
                    if overlap > overlap_threshold:
                        match = existing
                        break
            
            if match:
                # Merge into existing level
                self._merge_into_existing(level, match)
            else:
                # Add as new level
                merged.append(level)
        
        self.merged_levels = merged
        
        logger.debug(f"Merged into {len(merged)} unique levels")
    
    def _calculate_overlap(
        self,
        level1: SRLevel,
        level2: SRLevel
    ) -> float:
        """
        Calculate overlap percentage between two zones
        
        Returns:
            Overlap ratio (0.0 to 1.0)
        """
        # Find intersection
        overlap_start = max(level1.lower_bound, level2.lower_bound)
        overlap_end = min(level1.upper_bound, level2.upper_bound)
        
        if overlap_start >= overlap_end:
            return 0.0  # No overlap
        
        overlap_size = overlap_end - overlap_start
        
        # Use smaller zone as denominator
        zone1_size = level1.upper_bound - level1.lower_bound
        zone2_size = level2.upper_bound - level2.lower_bound
        min_zone_size = min(zone1_size, zone2_size)
        
        return overlap_size / min_zone_size if min_zone_size > 0 else 0.0
    
    def _merge_into_existing(
        self,
        new_level: SRLevel,
        existing_level: SRLevel
    ) -> None:
        """
        Merge new level data into existing level
        Updates timeframes, pivots, touches, and recalculates center
        """
        # Add timeframe if not present
        if new_level.timeframe_str not in existing_level.timeframes:
            existing_level.timeframes.append(new_level.timeframe_str)
        
        # Merge pivots
        existing_level.pivots.extend(new_level.pivots)
        
        # Merge touches
        existing_level.touches.extend(new_level.touches)
        
        # Update counts
        existing_level.touch_count = len(existing_level.pivots)
        
        # Recalculate center (volume-weighted)
        if self.config['zone_formation']['volume_weighted_center']:
            total_volume = sum(p.volume for p in existing_level.pivots)
            
            if total_volume > 0:
                existing_level.center = sum(
                    p.price * p.volume for p in existing_level.pivots
                ) / total_volume
                
                # Update bounds
                half_width = existing_level.zone_width / 2
                existing_level.upper_bound = existing_level.center + half_width
                existing_level.lower_bound = existing_level.center - half_width
        
        # Merge confluence scores
        for tf, score in new_level.confluence_scores.items():
            if tf in existing_level.confluence_scores:
                # Take max score for that timeframe
                existing_level.confluence_scores[tf] = max(
                    existing_level.confluence_scores[tf],
                    score
                )
            else:
                existing_level.confluence_scores[tf] = score
    
    def _apply_confluence_scoring(
        self,
        data: Dict[str, pd.DataFrame],
        primary_tf: str
    ) -> None:
        """
        Apply all confluence components to merged levels
        """
        # Get current ATR from primary timeframe
        if primary_tf in data:
            atr = self._calculate_atr(data[primary_tf])
        else:
            atr = 0.01  # Fallback
        
        for level in self.merged_levels:
            # Fibonacci confluence
            fib_count = 0
            fib_matches = []
            if self.config['fibonacci']['enabled']:
                fib_count, fib_matches = self.fib_confluence.check_confluence(
                    level, atr
                )
                level.strength += fib_count * self.config['fibonacci']['confluence_weight']
            
            # EMA confluence
            ema_weight = 0.0
            ema_matches = []
            if self.config['ema_confluence']['enabled']:
                ema_weight, ema_matches = self.ema_confluence.check_confluence(
                    level, atr
                )
                level.strength += ema_weight
            
            # Round number confluence
            round_weight = 0.0
            round_info = None
            if self.config['round_numbers']['enabled']:
                round_weight, round_info = self.round_number_detector.check_confluence(
                    level
                )
                level.strength += round_weight
            
            # Calculate volume quality
            volume_quality = self.liquidity_scorer.calculate_volume_quality(level)
            
            # Calculate liquidity score and priority
            liquidity_score = self.liquidity_scorer.calculate_score(
                level,
                fib_matches=fib_count,
                ema_weight=ema_weight,
                round_number_weight=round_weight,
                volume_quality=volume_quality
            )
            
            level.liquidity_score = liquidity_score
            level.priority = self.liquidity_scorer.assign_priority(liquidity_score)
            
            # Store confluence metadata
            if not hasattr(level, 'confluence_metadata'):
                level.confluence_metadata = {}
            
            level.confluence_metadata.update({
                'fibonacci_matches': fib_matches,
                'ema_matches': ema_matches,
                'round_number': round_info,
                'volume_quality': volume_quality
            })
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR for tolerance calculations"""
        if len(df) < period:
            return df['close'].iloc[-1] * 0.01
        
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]
        
        return atr if not np.isnan(atr) else df['close'].iloc[-1] * 0.01
    
    def _filter_weak_levels(self) -> None:
        """Remove levels below minimum liquidity score"""
        min_score = self.config['priority_scoring']['min_liquidity_score']
        
        before_count = len(self.merged_levels)
        
        self.merged_levels = [
            level for level in self.merged_levels
            if level.liquidity_score >= min_score
        ]
        
        removed = before_count - len(self.merged_levels)
        
        if removed > 0:
            logger.debug(f"Filtered {removed} weak levels (min score: {min_score})")
    
    def _count_by_priority(self) -> str:
        """Helper to count levels by priority"""
        counts = {
            'CRITICAL': 0,
            'HIGH': 0,
            'MEDIUM': 0,
            'LOW': 0
        }
        
        for level in self.merged_levels:
            counts[level.priority] += 1
        
        return ", ".join([f"{k}: {v}" for k, v in counts.items()])
    
    # ========================================================================
    # PUBLIC API METHODS
    # ========================================================================
    
    def get_all_levels(self) -> List[SRLevel]:
        """Get all merged levels"""
        return self.merged_levels
    
    def get_levels_by_priority(self, priority: str) -> List[SRLevel]:
        """
        Get levels filtered by priority
        
        Args:
            priority: CRITICAL, HIGH, MEDIUM, or LOW
        """
        return [
            level for level in self.merged_levels
            if level.priority == priority
        ]
    
    def get_levels_by_type(self, level_type: LevelType) -> List[SRLevel]:
        """Get levels filtered by type (SUPPORT or RESISTANCE)"""
        return [
            level for level in self.merged_levels
            if level.level_type == level_type
        ]
    
    def get_nearest_levels(
        self,
        current_price: float,
        count: int = 3,
        level_type: Optional[LevelType] = None
    ) -> List[SRLevel]:
        """
        Get nearest levels to current price
        
        Args:
            current_price: Current market price
            count: Number of levels to return
            level_type: Optional filter by SUPPORT or RESISTANCE
        """
        levels = self.merged_levels
        
        if level_type:
            levels = [l for l in levels if l.level_type == level_type]
        
        # Sort by distance to current price
        levels.sort(key=lambda x: abs(x.center - current_price))
        
        return levels[:count]
    
    def to_dict(self) -> List[dict]:
        """Export all levels as dictionary list"""
        return [level.to_dict() for level in self.merged_levels]
    
    def get_statistics(self) -> Dict:
        """Get summary statistics"""
        if not self.merged_levels:
            return {
                'total_levels': 0,
                'by_priority': {},
                'by_type': {},
                'avg_strength': 0,
                'avg_liquidity_score': 0
            }
        
        return {
            'total_levels': len(self.merged_levels),
            'by_priority': {
                'CRITICAL': len([l for l in self.merged_levels if l.priority == 'CRITICAL']),
                'HIGH': len([l for l in self.merged_levels if l.priority == 'HIGH']),
                'MEDIUM': len([l for l in self.merged_levels if l.priority == 'MEDIUM']),
                'LOW': len([l for l in self.merged_levels if l.priority == 'LOW'])
            },
            'by_type': {
                'SUPPORT': len([l for l in self.merged_levels if l.level_type == LevelType.SUPPORT]),
                'RESISTANCE': len([l for l in self.merged_levels if l.level_type == LevelType.RESISTANCE])
            },
            'avg_strength': round(np.mean([l.strength for l in self.merged_levels]), 2),
            'avg_liquidity_score': round(np.mean([l.liquidity_score for l in self.merged_levels]), 2),
            'last_update': self.last_update.isoformat() if self.last_update else None
        }
