import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
import logging
from dataclasses import dataclass
from scipy.special import softmax

from app.enums.MarketRegime import MarketRegime

@dataclass
class RegimeConfig:
    """Configuration for different market regimes"""
    regime: MarketRegime
    ema_periods: Dict[str, list]  # MTF EMA periods
    supertrend_config: Dict[str, Any]  # SuperTrend parameters
    confidence_threshold: float
    description: str

class RegimeClassifier:
    """
    Market regime classifier that analyzes multiple indicators to determine
    current market conditions and suggest optimal parameters.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger("app")
        
        # Default regime configurations
        self.regime_configs = self._initialize_regime_configs()
        
        # Classification parameters
        self.volatility_window = self.config.get('volatility_window', 20)
        self.trend_window = self.config.get('trend_window', 50)
        self.atr_lookback = self.config.get('atr_lookback', 100)
        self.smoothing_factor = self.config.get('smoothing_factor', 0.7)  # For temporal smoothing (0-1)
        
        # Configurable weights for scoring (to address arbitrariness)
        self.regime_score_weights = self.config.get('regime_score_weights', {
            MarketRegime.TRENDING_BULL: {'ema_alignment': 0.3, 'trending': 0.2, 'price_above_ema': 0.2, 'macd': 0.2, 'rsi': 0.1},
            MarketRegime.TRENDING_BEAR: {'ema_alignment': 0.3, 'trending': 0.2, 'price_above_ema': 0.2, 'macd': 0.2, 'rsi': 0.1},
            MarketRegime.RANGING_HIGH_VOL: {'not_trending': 0.4, 'high_vol': 0.3, 'wide_range': 0.2, 'not_near_extremes': 0.1},
            MarketRegime.RANGING_LOW_VOL: {'not_trending': 0.4, 'not_high_vol': 0.3, 'tight_range': 0.2, 'low_adx': 0.1},
            MarketRegime.BREAKOUT_BULL: {'near_resistance': 0.3, 'expanding_vol': 0.2, 'high_rsi': 0.2, 'macd': 0.2, 'high_volume': 0.1},
            MarketRegime.BREAKOUT_BEAR: {'near_support': 0.3, 'expanding_vol': 0.2, 'low_rsi': 0.2, 'macd': 0.2, 'high_volume': 0.1},
            MarketRegime.CONSOLIDATION: {'tight_range': 0.4, 'not_high_vol': 0.3, 'low_adx': 0.2, 'neutral_rsi': 0.1}
        })
        
        # Track previous regime for smoothing
        self.previous_regime = None
    
    def _initialize_regime_configs(self) -> Dict[MarketRegime, RegimeConfig]:
        """Initialize parameter sets for different market regimes"""
        return {
            MarketRegime.TRENDING_BULL: RegimeConfig(
                regime=MarketRegime.TRENDING_BULL,
                ema_periods={
                    '1h': [8, 13, 21],    # Faster for trend following
                    '4h': [5, 8, 13],
                    '1d': [3, 5, 8]
                },
                supertrend_config={
                    'atr_len': 8,          # Shorter ATR for responsiveness
                    'atr_mult': 2.5,       # Lower multiplier for earlier signals
                    'span': 10
                },
                confidence_threshold=0.7,
                description="Strong uptrend with momentum"
            ),
            MarketRegime.TRENDING_BEAR: RegimeConfig(
                regime=MarketRegime.TRENDING_BEAR,
                ema_periods={
                    '1h': [8, 13, 21],
                    '4h': [5, 8, 13],
                    '1d': [3, 5, 8]
                },
                supertrend_config={
                    'atr_len': 8,
                    'atr_mult': 2.5,
                    'span': 10
                },
                confidence_threshold=0.7,
                description="Strong downtrend with momentum"
            ),
            MarketRegime.RANGING_HIGH_VOL: RegimeConfig(
                regime=MarketRegime.RANGING_HIGH_VOL,
                ema_periods={
                    '1h': [21, 34, 55],    # Slower to avoid whipsaws
                    '4h': [13, 21, 34],
                    '1d': [8, 13, 21]
                },
                supertrend_config={
                    'atr_len': 14,         # Standard ATR
                    'atr_mult': 3.5,       # Higher multiplier for volatility
                    'span': 20
                },
                confidence_threshold=0.6,
                description="Sideways market with high volatility"
            ),
            MarketRegime.RANGING_LOW_VOL: RegimeConfig(
                regime=MarketRegime.RANGING_LOW_VOL,
                ema_periods={
                    '1h': [14, 21, 50],    # Standard periods
                    '4h': [8, 14, 21],
                    '1d': [5, 8, 14]
                },
                supertrend_config={
                    'atr_len': 12,
                    'atr_mult': 3.0,
                    'span': 14
                },
                confidence_threshold=0.5,
                description="Low volatility consolidation"
            ),
            MarketRegime.BREAKOUT_BULL: RegimeConfig(
                regime=MarketRegime.BREAKOUT_BULL,
                ema_periods={
                    '1h': [5, 8, 13],      # Very fast for breakout capture
                    '4h': [3, 5, 8],
                    '1d': [2, 3, 5]
                },
                supertrend_config={
                    'atr_len': 6,          # Very responsive
                    'atr_mult': 2.0,       # Low multiplier for early entry
                    'span': 8
                },
                confidence_threshold=0.8,
                description="Bullish breakout from consolidation"
            ),
            MarketRegime.BREAKOUT_BEAR: RegimeConfig(
                regime=MarketRegime.BREAKOUT_BEAR,
                ema_periods={
                    '1h': [5, 8, 13],
                    '4h': [3, 5, 8],
                    '1d': [2, 3, 5]
                },
                supertrend_config={
                    'atr_len': 6,
                    'atr_mult': 2.0,
                    'span': 8
                },
                confidence_threshold=0.8,
                description="Bearish breakdown from consolidation"
            ),
            MarketRegime.CONSOLIDATION: RegimeConfig(
                regime=MarketRegime.CONSOLIDATION,
                ema_periods={
                    '1h': [34, 55, 89],    # Very slow to avoid false signals
                    '4h': [21, 34, 55],
                    '1d': [13, 21, 34]
                },
                supertrend_config={
                    'atr_len': 20,         # Long ATR for stability
                    'atr_mult': 4.0,       # High multiplier to avoid noise
                    'span': 25
                },
                confidence_threshold=0.4,
                description="Deep consolidation phase"
            )
        }
    
    def classify_regime(self, data: pd.DataFrame, atr_data: pd.DataFrame = None) -> Tuple[MarketRegime, float, Dict[str, Any]]:
        """
        Classify the current market regime based on multiple indicators.
        
        :param data: DataFrame with OHLC(V) data (requires 'high', 'low', 'close'; 'open' and 'volume' optional).
        :param atr_data: Optional pre-calculated ATR data.
        :return: (regime, confidence, analysis_details)
        """
        try:
            # Validate required columns
            required_cols = ['high', 'low', 'close']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            if len(data) < max(self.volatility_window, self.trend_window, self.atr_lookback):
                self.logger.warning("Insufficient data length; returning default regime")
                return MarketRegime.RANGING_LOW_VOL, 0.5, {}
            
            analysis = self._analyze_market_conditions(data, atr_data)
            regime, confidence = self._determine_regime(analysis)
            
            self.logger.info(f"Regime classified as {regime.value} with confidence {confidence:.2f}")
            return regime, confidence, analysis
            
        except Exception as e:
            self.logger.error(f"Error in regime classification: {e}")
            return MarketRegime.RANGING_LOW_VOL, 0.5, {}
    
    def _analyze_market_conditions(self, data: pd.DataFrame, atr_data: pd.DataFrame = None) -> Dict[str, Any]:
        """Analyze various market conditions with stability checks"""
        analysis = {}
        
        # 1. Volatility Analysis
        analysis['volatility'] = self._analyze_volatility(data, atr_data)
        
        # 2. Trend Analysis
        analysis['trend'] = self._analyze_trend(data)
        
        # 3. Range Analysis
        analysis['range'] = self._analyze_range(data)
        
        # 4. Momentum Analysis
        analysis['momentum'] = self._analyze_momentum(data)
        
        # 5. Volume Analysis (if available, with fallback)
        if 'volume' in data.columns and not data['volume'].dropna().empty:
            analysis['volume'] = self._analyze_volume(data)
        else:
            analysis['volume'] = {'volume_ratio': 1.0, 'volume_trend': 0, 'obv_trend': 0, 'is_high_volume': False, 'is_volume_increasing': False}
        
        return analysis
    
    def _analyze_volatility(self, data: pd.DataFrame, atr_data: pd.DataFrame = None) -> Dict[str, float]:
        """Analyze volatility characteristics with guards"""
        returns = data['close'].pct_change().dropna()
        if len(returns) < self.volatility_window:
            return {'current_vol': 0, 'vol_percentile': 50, 'atr_percentile': 50, 'atr_trend': 0, 'is_high_vol': False, 'is_expanding_vol': False}
        
        realized_vol = returns.rolling(self.volatility_window).std() * np.sqrt(24 * 365)
        current_vol = realized_vol.iloc[-1]
        
        vol_percentile = realized_vol.rank(pct=True).iloc[-1] * 100
        
        atr_percentile = 50
        atr_trend = 0
        if atr_data is not None and len(atr_data) > 0:
            atr_col = [col for col in atr_data.columns if 'atr_' in col and 'percentile' not in col]
            if atr_col:
                atr_values = atr_data[atr_col[0]].dropna()
                if len(atr_values) > self.atr_lookback:
                    atr_percentile = atr_values.rolling(self.atr_lookback).rank(pct=True).iloc[-1] * 100
                    atr_trend = (atr_values.iloc[-1] / atr_values.iloc[-5] - 1) * 100 if atr_values.iloc[-5] != 0 else 0
        
        return {
            'current_vol': current_vol,
            'vol_percentile': vol_percentile,
            'atr_percentile': atr_percentile,
            'atr_trend': atr_trend,
            'is_high_vol': vol_percentile > 70,
            'is_expanding_vol': atr_trend > 5
        }
    
    def _analyze_trend(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trend characteristics with vectorized operations"""
        close = data['close']
        
        if len(close) < 50:  # Minimum for EMAs
            return {'price_above_fast_ema': False, 'price_above_medium_ema': False, 'price_above_slow_ema': False,
                    'ema_bullish_alignment': False, 'ema_bearish_alignment': False, 'adx': 0, 'trend_strength': 0,
                    'trend_consistency': 0.5, 'is_trending': False, 'is_strong_trend': False}
        
        ema_fast = close.ewm(span=8, adjust=False).mean()
        ema_medium = close.ewm(span=21, adjust=False).mean()
        ema_slow = close.ewm(span=50, adjust=False).mean()
        
        price_above_fast = close.iloc[-1] > ema_fast.iloc[-1]
        price_above_medium = close.iloc[-1] > ema_medium.iloc[-1]
        price_above_slow = close.iloc[-1] > ema_slow.iloc[-1]
        
        ema_bullish_alignment = (ema_fast.iloc[-1] > ema_medium.iloc[-1] > ema_slow.iloc[-1])
        ema_bearish_alignment = (ema_fast.iloc[-1] < ema_medium.iloc[-1] < ema_slow.iloc[-1])
        
        # ADX calculation (vectorized)
        high_low = data['high'] - data['low']
        high_close_prev = np.abs(data['high'] - data['close'].shift(1))
        low_close_prev = np.abs(data['low'] - data['close'].shift(1))
        true_range = np.maximum.reduce([high_low, high_close_prev, low_close_prev])
        true_range = pd.Series(true_range, index=data.index).fillna(0)
        
        plus_dm = np.where((data['high'] - data['high'].shift(1)) > (data['low'].shift(1) - data['low']),
                           np.maximum(data['high'] - data['high'].shift(1), 0), 0)
        minus_dm = np.where((data['low'].shift(1) - data['low']) > (data['high'] - data['high'].shift(1)),
                            np.maximum(data['low'].shift(1) - data['low'], 0), 0)
        
        tr_smooth = pd.Series(true_range).ewm(span=14, adjust=False).mean()
        plus_di = (pd.Series(plus_dm).ewm(span=14, adjust=False).mean() / tr_smooth) * 100
        minus_di = (pd.Series(minus_dm).ewm(span=14, adjust=False).mean() / tr_smooth) * 100
        
        dx = np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan) * 100
        adx = dx.ewm(span=14, adjust=False).mean()
        current_adx = adx.iloc[-1] if len(adx) > 0 else 0
        
        trend_consistency = self._calculate_trend_consistency(close, self.trend_window)
        
        return {
            'price_above_fast_ema': price_above_fast,
            'price_above_medium_ema': price_above_medium,
            'price_above_slow_ema': price_above_slow,
            'ema_bullish_alignment': ema_bullish_alignment,
            'ema_bearish_alignment': ema_bearish_alignment,
            'adx': current_adx,
            'trend_strength': current_adx,
            'trend_consistency': trend_consistency,
            'is_trending': current_adx > 25,
            'is_strong_trend': current_adx > 40
        }
    
    def _analyze_range(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze ranging characteristics"""
        close = data['close']
        high = data['high']
        low = data['low']
        
        lookback = min(50, len(data))
        if lookback < 10:
            return {'range_size_pct': 0, 'position_in_range': 0.5, 'upper_breakout_distance': 0, 'lower_breakout_distance': 0,
                    'bb_width': 0, 'is_near_resistance': False, 'is_near_support': False, 'is_tight_range': False, 'is_wide_range': False}
        
        recent_high = high.tail(lookback).max()
        recent_low = low.tail(lookback).min()
        range_size = (recent_high - recent_low) / recent_low * 100 if recent_low != 0 else 0
        
        current_price = close.iloc[-1]
        position_in_range = (current_price - recent_low) / (recent_high - recent_low) if recent_high != recent_low else 0.5
        
        upper_breakout_distance = (recent_high - current_price) / current_price * 100 if current_price != 0 else 0
        lower_breakout_distance = (current_price - recent_low) / current_price * 100 if current_price != 0 else 0
        
        bb_period = 20
        bb_std = 2
        bb_ma = close.rolling(bb_period).mean()
        bb_std_dev = close.rolling(bb_period).std()
        bb_upper = bb_ma + (bb_std_dev * bb_std)
        bb_lower = bb_ma - (bb_std_dev * bb_std)
        bb_width = ((bb_upper - bb_lower) / bb_ma * 100).iloc[-1] if len(bb_ma) > 0 and bb_ma.iloc[-1] != 0 else 0
        
        return {
            'range_size_pct': range_size,
            'position_in_range': position_in_range,
            'upper_breakout_distance': upper_breakout_distance,
            'lower_breakout_distance': lower_breakout_distance,
            'bb_width': bb_width,
            'is_near_resistance': position_in_range > 0.8,
            'is_near_support': position_in_range < 0.2,
            'is_tight_range': bb_width < 5,
            'is_wide_range': bb_width > 15
        }
    
    def _analyze_momentum(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze momentum characteristics"""
        close = data['close']
        if len(close) < 26:  # Min for MACD
            return {'rsi': 50, 'macd': 0, 'macd_signal': 0, 'macd_histogram': 0, 'is_oversold': False, 'is_overbought': False, 'macd_bullish': False}
        
        rsi_period = 14
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1] if len(rsi) > 0 else 50
        
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_histogram = macd - macd_signal
        
        return {
            'rsi': current_rsi,
            'macd': macd.iloc[-1] if len(macd) > 0 else 0,
            'macd_signal': macd_signal.iloc[-1] if len(macd_signal) > 0 else 0,
            'macd_histogram': macd_histogram.iloc[-1] if len(macd_histogram) > 0 else 0,
            'is_oversold': current_rsi < 30,
            'is_overbought': current_rsi > 70,
            'macd_bullish': macd.iloc[-1] > macd_signal.iloc[-1] if len(macd) > 0 else False
        }
    
    def _analyze_volume(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume characteristics with fallback defaults"""
        volume = data['volume']
        close = data['close']
        
        if len(volume) < 20:
            return {'volume_ratio': 1.0, 'volume_trend': 0, 'obv_trend': 0, 'is_high_volume': False, 'is_volume_increasing': False}
        
        vol_ma = volume.rolling(20).mean()
        current_vol_ratio = volume.iloc[-1] / vol_ma.iloc[-1] if vol_ma.iloc[-1] != 0 else 1.0
        
        vol_trend = volume.rolling(10).mean().pct_change().iloc[-1] * 100 if len(volume) > 10 else 0
        
        obv = (volume * np.sign(close.diff()).fillna(0)).cumsum()
        obv_trend = obv.rolling(10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0).iloc[-1] if len(obv) > 10 else 0
        
        return {
            'volume_ratio': current_vol_ratio,
            'volume_trend': vol_trend,
            'obv_trend': obv_trend,
            'is_high_volume': current_vol_ratio > 1.5,
            'is_volume_increasing': vol_trend > 10
        }
    
    def _calculate_trend_consistency(self, close: pd.Series, window: int) -> float:
        """Calculate trend consistency over a window"""
        if len(close) < window:
            return 0.5
        
        recent_close = close.tail(window)
        direction = np.sign(recent_close.diff())
        consistency = (direction == direction.shift(1)).mean()
        return consistency if not np.isnan(consistency) else 0.5
    
    def _determine_regime(self, analysis: Dict[str, Any]) -> Tuple[MarketRegime, float]:
        """Determine market regime based on analysis with temporal smoothing and probabilistic output"""
        scores = {regime: 0.0 for regime in MarketRegime}
        
        for regime in MarketRegime:
            weights = self.regime_score_weights.get(regime, {})
            vol = analysis.get('volatility', {})
            trend = analysis.get('trend', {})
            range_analysis = analysis.get('range', {})
            momentum = analysis.get('momentum', {})
            volume = analysis.get('volume', {})
            
            if regime == MarketRegime.TRENDING_BULL:
                scores[regime] += weights.get('ema_alignment', 0.3) if trend.get('ema_bullish_alignment', False) else 0
                scores[regime] += weights.get('trending', 0.2) if trend.get('is_trending', False) else 0
                scores[regime] += weights.get('price_above_ema', 0.2) if trend.get('price_above_slow_ema', False) else 0
                scores[regime] += weights.get('macd', 0.2) if momentum.get('macd_bullish', False) else 0
                scores[regime] += weights.get('rsi', 0.1) if momentum.get('rsi', 50) > 50 else 0
                
            elif regime == MarketRegime.TRENDING_BEAR:
                scores[regime] += weights.get('ema_alignment', 0.3) if trend.get('ema_bearish_alignment', False) else 0
                scores[regime] += weights.get('trending', 0.2) if trend.get('is_trending', False) else 0
                scores[regime] += weights.get('price_above_ema', 0.2) if not trend.get('price_above_slow_ema', True) else 0
                scores[regime] += weights.get('macd', 0.2) if not momentum.get('macd_bullish', True) else 0
                scores[regime] += weights.get('rsi', 0.1) if momentum.get('rsi', 50) < 50 else 0
                
            elif regime == MarketRegime.RANGING_HIGH_VOL:
                scores[regime] += weights.get('not_trending', 0.4) if not trend.get('is_trending', True) else 0
                scores[regime] += weights.get('high_vol', 0.3) if vol.get('is_high_vol', False) else 0
                scores[regime] += weights.get('wide_range', 0.2) if range_analysis.get('is_wide_range', False) else 0
                scores[regime] += weights.get('not_near_extremes', 0.1) if not (range_analysis.get('is_near_resistance', False) or range_analysis.get('is_near_support', False)) else 0
                
            elif regime == MarketRegime.RANGING_LOW_VOL:
                scores[regime] += weights.get('not_trending', 0.4) if not trend.get('is_trending', True) else 0
                scores[regime] += weights.get('not_high_vol', 0.3) if not vol.get('is_high_vol', True) else 0
                scores[regime] += weights.get('tight_range', 0.2) if range_analysis.get('is_tight_range', False) else 0
                scores[regime] += weights.get('low_adx', 0.1) if trend.get('adx', 100) < 20 else 0
                
            elif regime == MarketRegime.BREAKOUT_BULL:
                scores[regime] += weights.get('near_resistance', 0.3) if range_analysis.get('is_near_resistance', False) else 0
                scores[regime] += weights.get('expanding_vol', 0.2) if vol.get('is_expanding_vol', False) else 0
                scores[regime] += weights.get('high_rsi', 0.2) if momentum.get('rsi', 50) > 60 else 0
                scores[regime] += weights.get('macd', 0.2) if momentum.get('macd_bullish', False) else 0
                scores[regime] += weights.get('high_volume', 0.1) if volume.get('is_high_volume', False) else 0
                
            elif regime == MarketRegime.BREAKOUT_BEAR:
                scores[regime] += weights.get('near_support', 0.3) if range_analysis.get('is_near_support', False) else 0
                scores[regime] += weights.get('expanding_vol', 0.2) if vol.get('is_expanding_vol', False) else 0
                scores[regime] += weights.get('low_rsi', 0.2) if momentum.get('rsi', 50) < 40 else 0
                scores[regime] += weights.get('macd', 0.2) if not momentum.get('macd_bullish', True) else 0
                scores[regime] += weights.get('high_volume', 0.1) if volume.get('is_high_volume', False) else 0
                
            elif regime == MarketRegime.CONSOLIDATION:
                scores[regime] += weights.get('tight_range', 0.4) if range_analysis.get('is_tight_range', False) else 0
                scores[regime] += weights.get('not_high_vol', 0.3) if not vol.get('is_high_vol', True) else 0
                scores[regime] += weights.get('low_adx', 0.2) if trend.get('adx', 100) < 15 else 0
                scores[regime] += weights.get('neutral_rsi', 0.1) if 40 < momentum.get('rsi', 50) < 60 else 0
            
        # Temporal smoothing with previous regime
        if self.previous_regime is not None:
            prev_scores = {r: 1.0 if r == self.previous_regime else 0.0 for r in MarketRegime}
            for r in scores:
                scores[r] = self.smoothing_factor * scores[r] + (1 - self.smoothing_factor) * prev_scores[r]
        
        # Probabilistic normalization with softmax
        score_values = np.array(list(scores.values()))
        probabilities = softmax(score_values)
        regime_probs = {regime: prob for regime, prob in zip(scores.keys(), probabilities)}
        
        best_regime = max(regime_probs, key=regime_probs.get)
        confidence = regime_probs[best_regime]
        
        # Update previous for next call
        self.previous_regime = best_regime
        
        return best_regime, confidence
    
    def get_regime_config(self, regime: MarketRegime) -> RegimeConfig:
        """Get configuration for a specific regime"""
        return self.regime_configs.get(regime, self.regime_configs[MarketRegime.RANGING_LOW_VOL])
    
    def get_adaptive_parameters(self, data: pd.DataFrame, atr_data: pd.DataFrame = None) -> Dict[str, Any]:
        """Get adaptive parameters based on current market regime"""
        regime, confidence, analysis = self.classify_regime(data, atr_data)
        config = self.get_regime_config(regime)
        
        return {
            'regime': regime,
            'confidence': confidence,
            'ema_periods': config.ema_periods,
            'supertrend_config': config.supertrend_config,
            'analysis': analysis,
            'regime_description': config.description
        }
    
    def validate_regimes(self, classifications: pd.DataFrame) -> Dict[str, Any]:
        """Validate a series of regime classifications (e.g., from backtest)"""
        if 'regime' not in classifications.columns:
            raise ValueError("Classifications DataFrame must have 'regime' column")
        
        # Regime distribution
        distribution = classifications['regime'].value_counts(normalize=True).to_dict()
        
        # Switch rate
        classifications['switch'] = classifications['regime'] != classifications['regime'].shift(1)
        switch_rate = classifications['switch'].mean()
        
        # Average confidence per regime
        if 'confidence' in classifications.columns:
            avg_confidence = classifications.groupby('regime')['confidence'].mean().to_dict()
        else:
            avg_confidence = {}
        
        # Stability: Average duration per regime
        classifications['group'] = (classifications['regime'] != classifications['regime'].shift(1)).cumsum()
        durations = classifications.groupby(['regime', 'group']).size()
        avg_duration = durations.groupby('regime').mean().to_dict()
        
        return {
            'distribution': distribution,
            'switch_rate': switch_rate,
            'avg_confidence': avg_confidence,
            'avg_duration': avg_duration,
            'total_regimes': len(classifications),
            'unique_regimes': len(set(classifications['regime']))
        }
