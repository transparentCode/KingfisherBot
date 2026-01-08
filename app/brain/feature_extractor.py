import logging
import json
from typing import Dict, Any, Optional, List
import pandas as pd

from app.db.redis_handler import RedisHandler
from app.indicators.fractal_channel import FractalChannel
from app.indicators.volume_profile import VolumeProfile
from app.indicators.volume_weighted_regression import VolumeWeightedRegression

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Responsible for extracting features for the trading brain.
    
    Key Design:
    - Regime metrics (Hurst, trend score, etc.) are READ from Redis/DB (pre-computed on bar close)
    - Tactical indicators (FractalChannel, VolumeProfile, VWR) are calculated on-demand
    
    This avoids expensive re-computation of Hurst exponent on every tick.
    """
    
    # Timeframe hierarchy
    STRATEGIC_TFS = ['4h', '1h']     # Higher timeframes for context
    TACTICAL_TFS = ['30m', '15m']    # Execution timeframes
    
    # Feature prefixes by timeframe type
    TF_PREFIXES = {
        '4h': 'htf_',    # Higher TimeFrame
        '1h': 'mtf_',    # Mid TimeFrame  
        '30m': 'ttf_',   # Tactical TimeFrame
        '15m': 'ttf_'    # Tactical TimeFrame
    }

    def __init__(
        self, 
        redis_handler: Optional[RedisHandler] = None,
        db_pool = None,
        execution_tf: str = '15m'
    ):
        self.redis_handler = redis_handler
        self.db_pool = db_pool
        self.execution_tf = execution_tf
        
        # Tactical indicators (calculated on-demand for execution TF)
        self.fractal = FractalChannel(mode='geometric', pivot_method='zigzag', lookback=100)
        self.volume_profile = VolumeProfile(session_mode=True, bins=100)
        self.vwr = VolumeWeightedRegression(lookback=50)

    async def extract_all(
        self, 
        asset: str, 
        mtf_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Main entry point. Extracts all features for the brain.
        
        Args:
            asset: Symbol name (e.g., "BTCUSDT")
            mtf_data: Dict of timeframe -> DataFrame (from MTFDataManager)
            
        Returns:
            Dict of feature name -> value
        """
        features = {}
        
        # --- 1. REGIME METRICS: Read from cache (not recalculated) ---
        cached_regimes = await self._get_cached_regimes(asset)
        
        for tf, regime_data in cached_regimes.items():
            prefix = self.TF_PREFIXES.get(tf, 'unk_')
            
            features[f'{prefix}hurst'] = regime_data.get('hurst', 0.5)
            features[f'{prefix}trend_score'] = regime_data.get('trend_strength', 0.0)
            features[f'{prefix}vol_stress'] = regime_data.get('volatility', 0.0)
            features[f'{prefix}regime'] = regime_data.get('regime', 'UNCERTAIN')
            features[f'{prefix}skew'] = regime_data.get('skew', 0.0)
            features[f'{prefix}tail_risk'] = regime_data.get('kurtosis', 0.0)
            
            # Hilbert cycle if available
            features[f'{prefix}cycle_period'] = regime_data.get('cycle_period', 20.0)
            features[f'{prefix}cycle_state'] = regime_data.get('cycle_state', 0.0)
        
        # --- 2. TACTICAL INDICATORS: Calculate on execution timeframe ---
        df_exec = mtf_data.get(self.execution_tf)
        
        if df_exec is not None and not df_exec.empty and len(df_exec) > 50:
            try:
                # Fractal Channel (structure, support/resistance)
                fc_feats = self.fractal.get_features(df_exec)
                features.update(fc_feats)
                
                # Volume Profile (liquidity, POC, skew)
                vp_feats = self.volume_profile.get_features(df_exec)
                features.update(vp_feats)
                
                # Volume Weighted Regression (money flow, fair value)
                vwr_feats = self.vwr.get_features(df_exec)
                features.update(vwr_feats)
                
                # Current price and ATR for SL/TP calculation
                features['current_price'] = float(df_exec['close'].iloc[-1])
                features['current_atr'] = self._calculate_atr(df_exec)
                
            except Exception as e:
                logger.error(f"Error calculating tactical indicators for {asset}: {e}")
        else:
            logger.warning(f"Insufficient data for tactical indicators: {asset} {self.execution_tf}")
        
        return features

    async def _get_cached_regimes(self, asset: str) -> Dict[str, Dict]:
        """
        Fetch pre-computed regime metrics from Redis (primary) or DB (fallback).
        
        Returns:
            Dict of timeframe -> regime data dict
        """
        regimes = {}
        all_tfs = self.STRATEGIC_TFS + self.TACTICAL_TFS
        
        for tf in all_tfs:
            redis_key = f"regime:{asset}:{tf}"
            data = None
            
            # 1. Try Redis first (faster)
            if self.redis_handler:
                try:
                    raw = await self.redis_handler.get(redis_key)
                    if raw:
                        # Redis returns bytes or string, parse JSON
                        if isinstance(raw, bytes):
                            raw = raw.decode('utf-8')
                        data = json.loads(raw) if isinstance(raw, str) else raw
                except Exception as e:
                    logger.debug(f"Redis miss for {redis_key}: {e}")
            
            # 2. Fallback to DB
            if data is None and self.db_pool:
                try:
                    data = await self.db_pool.get_latest_regime_metrics(asset, tf)
                except Exception as e:
                    logger.debug(f"DB miss for regime {asset} {tf}: {e}")
            
            if data:
                regimes[tf] = data
            else:
                # Use neutral defaults if no cached data
                regimes[tf] = {
                    'hurst': 0.5,
                    'trend_strength': 0.0,
                    'volatility': 0.5,
                    'regime': 'UNCERTAIN'
                }
                logger.debug(f"No cached regime for {asset} {tf}, using defaults")
        
        return regimes

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR for stop loss sizing."""
        if len(df) < period:
            return 0.0
            
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift(1)).abs()
        low_close = (df['low'] - df['close'].shift(1)).abs()
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]
        
        return float(atr) if pd.notna(atr) else 0.0

    def get_feature_names(self) -> List[str]:
        """Return list of all possible feature names for documentation."""
        return [
            # Regime features (per TF prefix)
            '{prefix}hurst', '{prefix}trend_score', '{prefix}vol_stress',
            '{prefix}regime', '{prefix}skew', '{prefix}tail_risk',
            '{prefix}cycle_period', '{prefix}cycle_state',
            # Fractal Channel
            'fc_norm_pos', 'fc_geo_upper', 'fc_geo_lower',
            'fc_slope_pct', 'fc_sup_div', 'fc_res_div',
            'fc_res_r2', 'fc_sup_r2', 'fc_res_confidence', 'fc_sup_confidence',
            # Volume Profile
            'vp_pos', 'vp_skew', 'vp_kurt', 'vp_entropy', 'vp_liq_type',
            'vp_poc', 'vp_vah', 'vp_val',
            # Volume Weighted Regression
            'vwr_z_score', 'vwr_slope_div', 'vwr_money_slope',
            'vwr_eff_ratio', 'vwr_price_gap', 'vwr_fair_value',
            'vwr_geom_r2', 'vwr_vw_r2', 'vwr_confidence',
            # Meta
            'current_price', 'current_atr'
        ]
