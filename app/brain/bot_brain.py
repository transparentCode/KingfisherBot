import logging
from typing import Dict, Optional, Any
from datetime import datetime
import pandas as pd

from app.brain.models import TradeSignal
from app.brain.feature_extractor import FeatureExtractor
from app.brain.mtf_confluence_engine import MTFConfluenceEngine
from app.db.redis_handler import RedisHandler

logger = logging.getLogger(__name__)


class BotBrain:
    """
    The Trading Brain - Central decision-making component.
    
    Orchestrates:
    1. Feature extraction (cached regimes + tactical indicators)
    2. MTF confluence analysis
    3. Playbook selection and execution
    4. Signal generation
    
    Playbooks:
    - TREND_PULLBACK: Buy dips in uptrend, sell rallies in downtrend
    - MEAN_REVERSION: Fade extremes when market is ranging
    """
    
    # Configuration
    MIN_CONFIDENCE = 0.5            # Minimum signal confidence to emit
    HTF_TREND_THRESH = 0.2          # Trend score threshold for directional bias
    SCALP_Z_SCORE = 2.0             # VWR deviation threshold for mean reversion
    MIN_RR = 1.5                    # Minimum risk-reward ratio
    
    def __init__(
        self,
        redis_handler: Optional[RedisHandler] = None,
        db_pool = None,
        execution_tf: str = '15m'
    ):
        self.extractor = FeatureExtractor(
            redis_handler=redis_handler,
            db_pool=db_pool,
            execution_tf=execution_tf
        )
        self.confluence = MTFConfluenceEngine()
        self.execution_tf = execution_tf

    async def analyze_market(
        self, 
        asset: str, 
        mtf_data: Dict[str, pd.DataFrame]
    ) -> Optional[TradeSignal]:
        """
        Main entry point. Analyzes market and returns a TradeSignal if setup found.
        
        Args:
            asset: Symbol name (e.g., "BTCUSDT")
            mtf_data: Dict of timeframe -> DataFrame from MTFDataManager
            
        Returns:
            TradeSignal if valid setup found, None otherwise
        """
        try:
            # 1. Extract all features
            features = await self.extractor.extract_all(asset, mtf_data)
            
            if not features or 'current_price' not in features:
                logger.warning(f"Insufficient features for {asset}")
                return None
            
            # 2. Calculate MTF confluence
            confluence = self.confluence.calculate_confluence(features)
            
            # 3. Check if market conditions allow trading
            if confluence.conflict > 0.5:
                logger.debug(f"Skipping {asset}: MTF conflict too high ({confluence.conflict:.2f})")
                return None
            
            # 4. Check volatility stress
            vol_stress = features.get('htf_vol_stress', 0.0)
            if vol_stress > 0.9:
                logger.info(f"Skipping {asset}: Extreme volatility (stress {vol_stress:.2f})")
                return None
            
            # 5. Select and execute playbook based on regime
            signal = None
            htf_trend = features.get('htf_trend_score', 0.0)
            
            # PLAYBOOK A: TREND_PULLBACK
            if htf_trend > self.HTF_TREND_THRESH:
                signal = self._check_long_trend_setup(asset, features, confluence)
            elif htf_trend < -self.HTF_TREND_THRESH:
                signal = self._check_short_trend_setup(asset, features, confluence)
            
            # PLAYBOOK B: MEAN_REVERSION (if no trend signal)
            if signal is None:
                signal = self._check_mean_reversion_setup(asset, features, confluence)
            
            # 6. Validate signal quality
            if signal and signal.confidence >= self.MIN_CONFIDENCE:
                return signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error in analyze_market for {asset}: {e}", exc_info=True)
            return None

    def _check_long_trend_setup(
        self, 
        asset: str, 
        f: Dict[str, Any],
        confluence
    ) -> Optional[TradeSignal]:
        """
        PLAYBOOK: Buy the dip in an uptrend.
        
        Conditions:
        - HTF trend is bullish
        - Price is near support (fc_norm_pos < 0.2)
        - Structural divergence positive (fc_sup_div > 0) = buyers stepping in
        - Volume profile not showing distribution (vp_skew > -0.3)
        """
        # 1. Location: Are we near support?
        norm_pos = f.get('fc_norm_pos', 0.5)
        at_support = norm_pos < 0.2
        
        if not at_support:
            return None
        
        # 2. Alpha: Structural divergence (hidden strength)
        sup_div = f.get('fc_sup_div', 0.0)
        has_alpha = sup_div > 0.0
        
        # 3. Context: Volume Profile not showing distribution
        vp_skew = f.get('vp_skew', 0.0)
        valid_context = vp_skew > -0.3
        
        # 4. MTF permission
        allowed, reason = self.confluence.can_trade(f, "LONG")
        if not allowed:
            logger.debug(f"Long blocked for {asset}: {reason}")
            return None
        
        if has_alpha and valid_context:
            entry = f['current_price']
            atr = f.get('current_atr', entry * 0.01)
            
            # Stop Loss: Below geometric support or 2 ATR
            geo_lower = f.get('fc_geo_lower', entry - 2 * atr)
            stop_loss = min(geo_lower, entry - 2 * atr)
            
            # Take Profit: Geometric upper band
            geo_upper = f.get('fc_geo_upper', entry + 3 * atr)
            take_profit = geo_upper
            
            # Calculate RR
            risk = entry - stop_loss
            reward = take_profit - entry
            rr = reward / risk if risk > 0 else 0
            
            if rr < self.MIN_RR:
                return None
            
            # Confidence based on multiple factors
            confidence = self._calculate_confidence(f, confluence, "LONG")
            
            return TradeSignal(
                timestamp=datetime.now(),
                symbol=asset,
                direction="LONG",
                setup_type="TREND_PULLBACK",
                timeframe=self.execution_tf,
                entry_price=entry,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=confidence,
                risk_reward_ratio=rr,
                mtf_score=confluence.score,
                mtf_conflict=confluence.conflict,
                reasoning={
                    'htf_trend_score': f.get('htf_trend_score', 0.0),
                    'fc_norm_pos': norm_pos,
                    'fc_sup_div': sup_div,
                    'vp_skew': vp_skew,
                    'mtf_confidence': confluence.confidence
                }
            )
        
        return None

    def _check_short_trend_setup(
        self, 
        asset: str, 
        f: Dict[str, Any],
        confluence
    ) -> Optional[TradeSignal]:
        """
        PLAYBOOK: Sell the rally in a downtrend.
        
        Conditions:
        - HTF trend is bearish
        - Price is near resistance (fc_norm_pos > 0.8)
        - Structural divergence negative (fc_res_div < 0) = sellers stepping in
        - Volume profile not showing accumulation (vp_skew < 0.3)
        """
        # 1. Location: Are we near resistance?
        norm_pos = f.get('fc_norm_pos', 0.5)
        at_resistance = norm_pos > 0.8
        
        if not at_resistance:
            return None
        
        # 2. Alpha: Structural divergence (hidden weakness)
        res_div = f.get('fc_res_div', 0.0)
        has_alpha = res_div < 0.0
        
        # 3. Context: Volume Profile not showing accumulation
        vp_skew = f.get('vp_skew', 0.0)
        valid_context = vp_skew < 0.3
        
        # 4. MTF permission
        allowed, reason = self.confluence.can_trade(f, "SHORT")
        if not allowed:
            logger.debug(f"Short blocked for {asset}: {reason}")
            return None
        
        if has_alpha and valid_context:
            entry = f['current_price']
            atr = f.get('current_atr', entry * 0.01)
            
            # Stop Loss: Above geometric resistance or 2 ATR
            geo_upper = f.get('fc_geo_upper', entry + 2 * atr)
            stop_loss = max(geo_upper, entry + 2 * atr)
            
            # Take Profit: Geometric lower band
            geo_lower = f.get('fc_geo_lower', entry - 3 * atr)
            take_profit = geo_lower
            
            # Calculate RR
            risk = stop_loss - entry
            reward = entry - take_profit
            rr = reward / risk if risk > 0 else 0
            
            if rr < self.MIN_RR:
                return None
            
            confidence = self._calculate_confidence(f, confluence, "SHORT")
            
            return TradeSignal(
                timestamp=datetime.now(),
                symbol=asset,
                direction="SHORT",
                setup_type="TREND_PULLBACK",
                timeframe=self.execution_tf,
                entry_price=entry,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=confidence,
                risk_reward_ratio=rr,
                mtf_score=confluence.score,
                mtf_conflict=confluence.conflict,
                reasoning={
                    'htf_trend_score': f.get('htf_trend_score', 0.0),
                    'fc_norm_pos': norm_pos,
                    'fc_res_div': res_div,
                    'vp_skew': vp_skew,
                    'mtf_confidence': confluence.confidence
                }
            )
        
        return None

    def _check_mean_reversion_setup(
        self, 
        asset: str, 
        f: Dict[str, Any],
        confluence
    ) -> Optional[TradeSignal]:
        """
        PLAYBOOK: Fade the extremes when market is ranging.
        
        Conditions:
        - VWR Z-Score > 2.0 (price way above fair value) -> SHORT
        - VWR Z-Score < -2.0 (price way below fair value) -> LONG
        """
        z_score = f.get('vwr_z_score', 0.0)
        
        # SHORT: Price stretched above fair value
        if z_score > self.SCALP_Z_SCORE:
            entry = f['current_price']
            atr = f.get('current_atr', entry * 0.01)
            
            # Target: Return to fair value
            target = f.get('vwr_fair_value', entry * 0.99)
            stop_loss = entry + 1.5 * atr  # Tight stop for scalps
            
            risk = stop_loss - entry
            reward = entry - target
            rr = reward / risk if risk > 0 else 0
            
            if rr < 1.0:  # Lower RR threshold for scalps
                return None
            
            return TradeSignal(
                timestamp=datetime.now(),
                symbol=asset,
                direction="SHORT",
                setup_type="MEAN_REVERSION",
                timeframe=self.execution_tf,
                entry_price=entry,
                stop_loss=stop_loss,
                take_profit=target,
                confidence=0.6,  # Scalps get lower base confidence
                risk_reward_ratio=rr,
                mtf_score=confluence.score,
                mtf_conflict=confluence.conflict,
                reasoning={
                    'vwr_z_score': z_score,
                    'vwr_fair_value': f.get('vwr_fair_value', 0.0),
                    'cycle_state': f.get('ttf_cycle_state', 0.0)
                }
            )
        
        # LONG: Price stretched below fair value
        elif z_score < -self.SCALP_Z_SCORE:
            entry = f['current_price']
            atr = f.get('current_atr', entry * 0.01)
            
            target = f.get('vwr_fair_value', entry * 1.01)
            stop_loss = entry - 1.5 * atr
            
            risk = entry - stop_loss
            reward = target - entry
            rr = reward / risk if risk > 0 else 0
            
            if rr < 1.0:
                return None
            
            return TradeSignal(
                timestamp=datetime.now(),
                symbol=asset,
                direction="LONG",
                setup_type="MEAN_REVERSION",
                timeframe=self.execution_tf,
                entry_price=entry,
                stop_loss=stop_loss,
                take_profit=target,
                confidence=0.6,
                risk_reward_ratio=rr,
                mtf_score=confluence.score,
                mtf_conflict=confluence.conflict,
                reasoning={
                    'vwr_z_score': z_score,
                    'vwr_fair_value': f.get('vwr_fair_value', 0.0),
                    'cycle_state': f.get('ttf_cycle_state', 0.0)
                }
            )
        
        return None

    def _calculate_confidence(
        self, 
        f: Dict[str, Any], 
        confluence,
        direction: str
    ) -> float:
        """
        Calculate confidence score based on multiple factors.
        
        Components:
        - MTF alignment (confluence.confidence)
        - Indicator quality (R² values)
        - Volume confirmation (vp_skew agreement)
        """
        base = 0.5
        
        # 1. MTF alignment bonus
        base += confluence.confidence * 0.2
        
        # 2. Structural quality (R² if available)
        if direction == "LONG":
            r2 = f.get('fc_sup_r2', 0.5)
        else:
            r2 = f.get('fc_res_r2', 0.5)
        base += r2 * 0.1
        
        # 3. Volume skew agreement
        vp_skew = f.get('vp_skew', 0.0)
        if direction == "LONG" and vp_skew > 0.5:
            base += 0.1
        elif direction == "SHORT" and vp_skew < -0.5:
            base += 0.1
        
        # 4. VWR confidence
        vwr_conf = f.get('vwr_confidence', 0.5)
        base += vwr_conf * 0.1
        
        return min(1.0, base)
