import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass
from sklearn.cluster import DBSCAN

from app.models.sr_level_model import (
    SRLevel,
    PivotPoint,
    TouchEvent,
    LevelType,
    LevelStatus,
)


logger = logging.getLogger(__name__)


class SupportResistanceDetector:
    """
    Core detector for single-timeframe S/R identification
    Implements zone-based pivot detection with validation
    """

    def __init__(self, timeframe: str, config: dict):
        self.timeframe = timeframe
        self.config = config

        # Extract config sections
        self.pivot_config = config["pivot_detection"]
        self.zone_config = config["zone_formation"]
        self.validation_config = config["validation"]
        self.confluence_config = config["confluence"]

        # Get timeframe-specific lookback
        lookback_defaults = self.pivot_config["lookback"]["defaults"]
        tf_lookback = lookback_defaults.get(timeframe, {"n1": 8, "n2": 6})
        self.n1 = tf_lookback["n1"]
        self.n2 = tf_lookback["n2"]

        # Historical depth
        self.historical_depth = self.pivot_config["historical_depth"].get(
            timeframe, 200
        )

        # State
        self.levels: List[SRLevel] = []
        self.support_pivots: List[PivotPoint] = []
        self.resistance_pivots: List[PivotPoint] = []

        self.current_bar = 0
        self.data: Optional[pd.DataFrame] = None

        logger.info(
            f"Initialized SR Detector for {timeframe} (n1={self.n1}, n2={self.n2})"
        )

    def update(self, df: pd.DataFrame) -> None:
        """
        Main update method - processes new data

        Args:
            df: OHLCV DataFrame with columns [open, high, low, close, volume]
        """
        if df is None or len(df) < self.n1 + self.n2:
            logger.warning(f"Insufficient data for {self.timeframe}")
            return

        # Apply smoothing if enabled
        self.data = self._smooth_price_data(df.copy())
        self.current_bar = len(self.data) - 1

        # Phase 1: Detect pivots
        self._detect_pivots()

        # Phase 2: Form zones from pivots
        self._form_zones()

        # Phase 3: Validate levels
        self._validate_levels()

        # Phase 4: Update existing levels
        self._update_existing_levels()

        # Phase 5: Calculate strength & age decay
        self._calculate_strength()

        # Phase 6: Cleanup weak levels
        self._cleanup_levels()

        logger.debug(
            f"{self.timeframe}: {len(self.levels)} active levels "
            f"({len([l for l in self.levels if l.level_type == LevelType.SUPPORT])} S, "
            f"{len([l for l in self.levels if l.level_type == LevelType.RESISTANCE])} R)"
        )

    def _smooth_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply EMA smoothing to reduce noise"""
        if not self.pivot_config["smoothing"]["enabled"]:
            return df

        period = self.pivot_config["smoothing"]["period"]

        for col in ["high", "low", "close"]:
            df[f"{col}_smooth"] = df[col].ewm(span=period, adjust=False).mean()

        return df

    def _detect_pivots(self) -> None:
        """
        Phase 1: Identify pivot highs/lows
        """
        df = self.data

        # Use smoothed prices if available
        high_col = "high_smooth" if "high_smooth" in df.columns else "high"
        low_col = "low_smooth" if "low_smooth" in df.columns else "low"

        support_pivots = []
        resistance_pivots = []

        # Scan historical window
        start_idx = max(0, self.current_bar - self.historical_depth)
        end_idx = self.current_bar - self.n2  # Can't detect pivot at edge

        for i in range(start_idx + self.n1, end_idx):
            # Resistance (Pivot High)
            window_high = df[high_col].iloc[i - self.n1 : i + self.n2 + 1]
            if df[high_col].iloc[i] == window_high.max():
                pivot = PivotPoint(
                    price=df["high"].iloc[i],  # Use original price, not smoothed
                    index=i,
                    timestamp=df.index[i]
                    if isinstance(df.index, pd.DatetimeIndex)
                    else datetime.now(),
                    volume=df["volume"].iloc[i],
                    timeframe=self.timeframe,
                    is_high=True,
                )
                resistance_pivots.append(pivot)

            # Support (Pivot Low)
            window_low = df[low_col].iloc[i - self.n1 : i + self.n2 + 1]
            if df[low_col].iloc[i] == window_low.min():
                pivot = PivotPoint(
                    price=df["low"].iloc[i],
                    index=i,
                    timestamp=df.index[i]
                    if isinstance(df.index, pd.DatetimeIndex)
                    else datetime.now(),
                    volume=df["volume"].iloc[i],
                    timeframe=self.timeframe,
                    is_high=False,
                )
                support_pivots.append(pivot)

        self.support_pivots = support_pivots
        self.resistance_pivots = resistance_pivots

        logger.debug(
            f"{self.timeframe}: Found {len(resistance_pivots)} resistance pivots, "
            f"{len(support_pivots)} support pivots"
        )

    def _form_zones(self) -> None:
        """
        Phase 2: Cluster pivots into zones
        """
        # Calculate ATR for zone width
        atr = self._calculate_atr()
        zone_width_base = atr * self.zone_config["width_multiplier"]["default"]

        # Clustering tolerance
        cluster_tol = (
            zone_width_base * self.zone_config["clustering"]["tolerance_multiplier"]
        )

        # Form resistance zones
        resistance_zones = self._cluster_pivots(
            self.resistance_pivots, cluster_tol, zone_width_base, LevelType.RESISTANCE
        )

        # Form support zones
        support_zones = self._cluster_pivots(
            self.support_pivots, cluster_tol, zone_width_base, LevelType.SUPPORT
        )

        # Combine and replace existing levels
        new_levels = resistance_zones + support_zones

        # Merge with existing levels (preserve state)
        self._merge_with_existing(new_levels)

    def _cluster_pivots(
        self,
        pivots: List[PivotPoint],
        tolerance: float,
        zone_width: float,
        level_type: LevelType,
    ) -> List[SRLevel]:
        if not pivots: 
            return []

        # Convert to numpy for sklearn
        prices = np.array([p.price for p in pivots]).reshape(-1, 1)
        
        # Use DBSCAN (tolerance = eps)
        # min_samples = min_touch_count
        db = DBSCAN(eps=tolerance, min_samples=self.zone_config["min_touch_count"]).fit(prices)
        labels = db.labels_
        
        zones = []
        unique_labels = set(labels)
        
        for label in unique_labels:
            if label == -1: continue # Noise
            
            # Get pivots in this cluster
            cluster_indices = np.where(labels == label)[0]
            cluster_pivots = [pivots[i] for i in cluster_indices]
            
            # Create Zone
            zone = self._create_zone_from_cluster(cluster_pivots, zone_width, level_type)
            zones.append(zone)
            
        return zones



    def _create_zone_from_cluster(
        self, cluster: List[PivotPoint], zone_width_input: float, level_type: LevelType
    ) -> SRLevel:
        """
        Create SRLevel from clustered pivots
        
        Args:
            cluster: List of pivot points
            zone_width_input: Base width calculated from ATR (usually 1.0 * ATR)
            level_type: Expected type (Support/Resistance)
        """
        if not cluster: raise ValueError("Empty cluster")

        # 1. Calculate Center (Volume Weighted or Mean)
        if self.zone_config.get("volume_weighted_center", True):
            total_volume = sum(p.volume for p in cluster)
            if total_volume > 0:
                center = sum(p.price * p.volume for p in cluster) / total_volume
            else:
                center = np.mean([p.price for p in cluster])
        else:
            center = np.mean([p.price for p in cluster])

        # 2. Calculate Bounds using Actual Spread + Minimum ATR Buffer
        # zone_width_input usually comes from 1.0 * ATR or similar
        # We enforce a minimum thickness of 0.2 * ATR (20% of the input base width) or whatever the user config implies.
        # Let's assume zone_width_input is "1 ATR".
        
        # Calculate raw spread of the pivots
        prices = [p.price for p in cluster]
        p_min, p_max = min(prices), max(prices)
        spread = p_max - p_min
        
        # Enforce minimum width if the cluster is too tight (0 variance)
        # We ensure the zone serves as a buffer. 
        # Using 0.2 * zone_width_input as the minimum thickness foundation
        min_thickness = zone_width_input * 0.2
        
        if spread < min_thickness:
            # Spread is too thin. Center on 'center' and apply min thickness
            half_width = min_thickness / 2
            upper_bound = center + half_width
            lower_bound = center - half_width
        else:
            # Cluster has variance. Use the spread, but pad it slightly 
            # so the extreme pivots aren't exactly on the edge.
            padding = min_thickness * 0.5 # 10% of ATR padding
            upper_bound = p_max + padding
            lower_bound = p_min - padding

        # Recalculate width
        final_width = upper_bound - lower_bound

        # Formation time (earliest pivot)
        formation_time = min(p.timestamp for p in cluster)
        
        # Determine Current Status (Check against LAST PRICE to handle Backtest Flip logic)
        # Only relevant if we have data loaded
        status = LevelStatus.ACTIVE
        current_type = level_type
        
        if self.data is not None and not self.data.empty:
            last_close = self.data['close'].iloc[-1]
            
            # If Support is NOW above price -> It's Resistance (Flipped)
            if level_type == LevelType.SUPPORT and last_close < lower_bound:
                current_type = LevelType.RESISTANCE
                status = LevelStatus.FLIPPED
                
            # If Resistance is NOW below price -> It's Support (Flipped)
            elif level_type == LevelType.RESISTANCE and last_close > upper_bound:
                current_type = LevelType.SUPPORT
                status = LevelStatus.FLIPPED

        # Create level
        level = SRLevel(
            level_id="",  # Will be auto-generated
            level_type=current_type,
            center=center,
            upper_bound=upper_bound,
            lower_bound=lower_bound,
            zone_width=final_width,
            strength=0.0,  # Calculated later
            touch_count=len(cluster),
            rejection_count=0,  # Calculated in validation
            avg_touch_volume=np.mean([p.volume for p in cluster]),
            timeframes=[self.timeframe],
            timeframe_str=self.timeframe,
            confluence_scores={self.timeframe: 0.0},
            formation_time=formation_time,
            last_touch_time=max(p.timestamp for p in cluster),
            age_bars=self.current_bar - cluster[0].index,
            status=status,
            pivots=cluster,
            atr_at_formation=self._calculate_catr(),
            avg_volume_at_formation=self.data["volume"].tail(20).mean(),
        )

        return level

    def _calculate_atr(self, period: Optional[int] = None) -> float:
        """Calculate Average True Range"""
        if period is None:
            period = self.zone_config["atr"]["period"]

        df = self.data

        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]

        return atr if not np.isnan(atr) else df["close"].iloc[-1] * 0.01

    def _validate_levels(self) -> None:
        """
        Phase 3: Validate zones with wick rejection & volume
        """
        if not self.validation_config["wick_rejection"]["enabled"]:
            return

        df = self.data
        wick_ratio_threshold = self.validation_config["wick_rejection"][
            "wick_to_body_ratio"
        ]

        for level in self.levels:
            rejection_count = 0

            for touch in level.touches:
                if touch.wick_size > touch.body_size * wick_ratio_threshold:
                    rejection_count += 1
                    touch.is_rejection = True

            level.rejection_count = rejection_count

            # Check rejection ratio
            if level.touch_count > 0:
                rejection_ratio = rejection_count / level.touch_count
                min_ratio = self.validation_config["wick_rejection"][
                    "min_rejection_ratio"
                ]

                if rejection_ratio >= min_ratio:
                    # Boost strength for validated levels
                    level.strength += 1.0

    def _update_existing_levels(self) -> None:
        """
        Phase 4: Check for new touches and breakouts
        """
        if self.data is None or len(self.data) == 0:
            return

        current_bar = self.data.iloc[-1]
        avg_volume = (
            self.data["volume"]
            .tail(self.validation_config["volume_confirmation"]["volume_lookback"])
            .mean()
        )

        for level in self.levels:
            # Check if price touched zone
            if self._is_price_in_zone(current_bar, level):
                touch = self._create_touch_event(current_bar, level)
                level.add_touch(touch)

            # Check for breakout
            if level.check_breakout(
                current_bar["close"],
                current_bar["volume"],
                avg_volume,
                self.validation_config["volume_confirmation"][
                    "breakout_volume_multiplier"
                ],
            ):
                logger.info(f"Breakout detected: {level.level_id}")

                if self.config["dynamic_updates"]["breakout"][
                    "flip_support_resistance"
                ]:
                    level.flip()
                else:
                    level.status = LevelStatus.BROKEN

    def _is_price_in_zone(self, bar: pd.Series, level: SRLevel) -> bool:
        """Check if price action touched the zone"""
        return bar["low"] <= level.upper_bound and bar["high"] >= level.lower_bound

    def _create_touch_event(self, bar: pd.Series, level: SRLevel) -> TouchEvent:
        """Create touch event from bar data"""
        body_size = abs(bar["close"] - bar["open"])

        if level.level_type == LevelType.RESISTANCE:
            wick_size = abs(bar["high"] - max(bar["open"], bar["close"]))
        else:
            wick_size = abs(min(bar["open"], bar["close"]) - bar["low"])

        return TouchEvent(
            timestamp=bar.name
            if isinstance(bar.name, pd.Timestamp)
            else datetime.now(),
            price=bar["close"],
            volume=bar["volume"],
            wick_size=wick_size,
            body_size=body_size if body_size > 0 else 0.0001,
            is_rejection=False,  # Determined in validation
            bar_index=self.current_bar,
        )

    def _calculate_strength(self) -> None:
        """
        Phase 5: Calculate level strength with confluence formula
        """
        tf_weights = self.confluence_config["timeframe_weights"]
        age_lambda = self.confluence_config["age_decay"]["lambda"]
        touch_bonus = self.confluence_config["touch_bonus"]["multiplier"]

        for level in self.levels:
            level.age_bars = (
                self.current_bar - level.pivots[0].index if level.pivots else 0
            )
            level.update_strength(tf_weights, touch_bonus, age_lambda)

            # Apply inactivity penalty if enabled
            if self.confluence_config["inactivity_penalty"]["enabled"]:
                self._apply_inactivity_penalty(level)

    def _apply_inactivity_penalty(self, level: SRLevel) -> None:
        """Penalize levels without recent touches"""
        if not level.last_touch_time:
            return

        # Calculate bars since last touch (simplified - assumes uniform timeframe)
        # In production, convert timestamps properly
        bars_since_touch = self.current_bar - (
            level.touches[-1].bar_index if level.touches else level.pivots[0].index
        )

        threshold = self.confluence_config["inactivity_penalty"]["threshold_bars"]

        if bars_since_touch > threshold:
            decay_rate = self.confluence_config["inactivity_penalty"]["decay_rate"]
            penalty = decay_rate ** (bars_since_touch / threshold)
            level.strength *= penalty

    def _cleanup_levels(self) -> None:
        """
        Phase 6: Remove weak/old levels
        """
        min_threshold = self.validation_config["strength"]["min_threshold"]
        max_age = self.confluence_config["age_decay"]["max_age_bars"]

        self.levels = [
            level
            for level in self.levels
            if level.strength >= min_threshold and level.age_bars < max_age
        ]

    def _merge_with_existing(self, new_zones: List[SRLevel]) -> None:
        """
        Map new zones to old zones to preserve history (Strength/Status).
        """
        # Create a dict of old zones for fast lookup by ID or Center
        # Since centers drift, we use spatial overlap
        
        merged_list = []
        
        for new_z in new_zones:
            # Try to find a matching old zone
            match = None
            for old_z in self.levels:
                # Check overlap (Are these the same zone?)
                overlap = max(0, min(new_z.upper_bound, old_z.upper_bound) - 
                                 max(new_z.lower_bound, old_z.lower_bound))
                if overlap > 0 and new_z.level_type == old_z.level_type:
                    match = old_z
                    break
            
            if match:
                # We found the old version of this zone.
                # Update its geometry (pivots might have shifted slightly)
                match.center = new_z.center
                match.upper_bound = new_z.upper_bound
                match.lower_bound = new_z.lower_bound
                
                # Careful not to lose old pivots if they are still relevant
                # But DBSCAN runs on *all* pivots, so new_z.pivots likely contains valid ones?
                # Actually, detect_pivots finds recent history. 
                # If we overwrite `match.pivots = new_z.pivots`, we might lose very old pivots that fell out of `historical_depth`
                # but are still part of the level.
                
                # De-duplication Logic:
                existing_indices = {p.index for p in match.pivots}
                for p in new_z.pivots:
                    if p.index not in existing_indices:
                        match.pivots.append(p)

                # Keep its 'Strength', 'Touches', 'Status' from history
                merged_list.append(match)
            else:
                # Totally new zone found by DBSCAN
                merged_list.append(new_z)
                
        self.levels = merged_list


    def get_active_levels(self) -> List[SRLevel]:
        """Get all active levels"""
        return [l for l in self.levels if l.status == LevelStatus.ACTIVE]

    def to_dict(self) -> List[dict]:
        """Export levels as dictionary list"""
        return [level.to_dict() for level in self.levels]
