import pandas as pd
import numpy as np
from typing import Optional, Dict


class FairValueGap:
    """
    Sophisticated FVG (Fair Value Gap) Detector.
    Filters out noise using Volume, ATR, and Momentum confirmation.
    """
    
    def __init__(self, **kwargs):
        # Thresholds (Tunable per asset/timeframe)
        self.min_gap_pct = kwargs.get('min_gap_pct', 0.003)  # 0.3% minimum gap
        self.atr_multiplier = kwargs.get('atr_multiplier', 0.5)  # Gap must be > 0.5 * ATR
        self.vol_percentile = kwargs.get('vol_percentile', 60)  # Volume must be > 60th percentile
        self.momentum_threshold = kwargs.get('momentum_threshold', 0.6)  # Candle body/range ratio
        
    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Quick ATR calculation for the most recent bars."""
        if len(data) < period:
            period = len(data)
            
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift(1))
        low_close = np.abs(data['low'] - data['close'].shift(1))
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(period).mean().iloc[-1]
        
        return atr if not pd.isna(atr) else 0.0
    
    def detect(self, data: pd.DataFrame) -> Optional[Dict]:
        """
        Detects FVG based on the last 3 closed candles.
        
        Structure:
        [Candle 1 (Anchor)] -- [Candle 2 (Displacement)] -- [Candle 3 (Signal/Current)]
        
        Gap is between Candle 1 and Candle 3.
        Displacement (Volume/Momentum) is checked on Candle 2.
        """
        if len(data) < 20:  # Need history for ATR and Volume baseline
            return None
            
        # --- Extract Last 3 Candles ---
        c_current = data.iloc[-1]   # Candle 3 (Signal)
        c_prev = data.iloc[-2]      # Candle 2 (Displacement/Gap Candle)
        c_anchor = data.iloc[-3]    # Candle 1 (Anchor)
        
        # --- 1. Detect Gap Existence ---
        bullish_gap = c_current['low'] > c_anchor['high']
        bearish_gap = c_current['high'] < c_anchor['low']
        
        if not (bullish_gap or bearish_gap):
            return None
        
        gap_type = 'BULLISH' if bullish_gap else 'BEARISH'
        
        # --- FIX 1: Validate Displacement Direction ---
        displacement_direction = 'BULLISH' if c_prev['close'] > c_prev['open'] else 'BEARISH'
        
        if gap_type != displacement_direction:
            return None  # Contradiction: Gap direction doesn't match displacement candle
        
        # --- 2. Calculate Gap Metrics ---
        if gap_type == 'BULLISH':
            gap_size = c_current['low'] - c_anchor['high']
            gap_top = c_current['low']
            gap_bottom = c_anchor['high']
        else:
            gap_size = c_anchor['low'] - c_current['high']
            gap_top = c_anchor['low']
            gap_bottom = c_current['high']
        
        # FIX 2: Validate Gap Size
        if gap_size <= 0:
            return None
        
        gap_pct = gap_size / c_anchor['close']
        
        # --- 3. Filter: Minimum Gap Size ---
        atr = self.calculate_atr(data.iloc[:-1])  # ATR of the last N bars before signal
        min_gap_absolute = max(
            c_anchor['close'] * self.min_gap_pct,
            atr * self.atr_multiplier
        )
        
        if gap_size < min_gap_absolute:
            return None  # Gap too small (noise)
        
        # --- 4. Filter: Volume Confirmation (On Displacement Candle) ---
        # FIX 3: Safer volume lookback slicing
        start_idx = max(0, len(data) - 22)
        end_idx = len(data) - 2
        vol_lookback = data['volume'].iloc[start_idx:end_idx]
        
        if len(vol_lookback) < 5:  # Need minimum history
            return None
        
        avg_vol = vol_lookback.mean()
        vol_threshold = np.percentile(vol_lookback, self.vol_percentile)
        
        signal_vol = c_prev['volume']  # Check Displacement Candle
        volume_ratio = signal_vol / avg_vol if avg_vol > 0 else 0
        
        if signal_vol < vol_threshold:
            return None  # Volume too weak
        
        # --- 5. Filter: Momentum (On Displacement Candle) ---
        candle_range = c_prev['high'] - c_prev['low']
        candle_body = abs(c_prev['close'] - c_prev['open'])
        
        momentum_score = candle_body / candle_range if candle_range > 0 else 0
        
        if momentum_score < self.momentum_threshold:
            return None  # Weak candle (too much wick, indecision)
        
        # --- 6. Calculate Confidence Score (Composite) ---
        gap_score = min(gap_pct / 0.01, 1.0)  # Normalize to 1% gap = max
        vol_score = min((volume_ratio - 1.0) / 1.0, 1.0)  # 2x volume = max
        momentum_score_norm = momentum_score  # Already 0-1
        
        confidence = (gap_score * 0.4 + vol_score * 0.3 + momentum_score_norm * 0.3)
        
        return {
            'type': gap_type,
            'gap_size': gap_size,
            'gap_pct': gap_pct,
            'gap_top': gap_top,
            'gap_bottom': gap_bottom,
            'volume_ratio': volume_ratio,
            'momentum_score': momentum_score,
            'confidence': confidence,
            'atr': atr
        }

    def detect_historical(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Vectorized scan for FVG occurrences.
        Much faster than iterating row-by-row.
        """
        df = data.copy()
        
        # --- Pre-calculations ---
        # 1. ATR (Shifted by 1 because detect() uses ATR of previous bars)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().shift(1)
        
        # 2. Volume Threshold (Shifted by 2 because we look at 20 bars before displacement)
        # Displacement is at i-1. Lookback is i-21 to i-2.
        # So at index i, we want rolling stats of volume shifted by 2.
        vol_rolling = df['volume'].shift(2).rolling(20)
        vol_threshold = vol_rolling.quantile(self.vol_percentile / 100.0)
        avg_vol = vol_rolling.mean()
        
        # 3. Previous Candles
        prev_close = df['close'].shift(1)
        prev_open = df['open'].shift(1)
        prev_high = df['high'].shift(1)
        prev_low = df['low'].shift(1)
        prev_vol = df['volume'].shift(1)
        
        anchor_high = df['high'].shift(2)
        anchor_low = df['low'].shift(2)
        anchor_close = df['close'].shift(2)
        
        # --- Logic ---
        
        # 1. Gap Existence
        bullish_gap = df['low'] > anchor_high
        bearish_gap = df['high'] < anchor_low
        
        # 2. Displacement Direction
        prev_bullish = prev_close > prev_open
        prev_bearish = prev_close < prev_open
        
        valid_bullish = bullish_gap & prev_bullish
        valid_bearish = bearish_gap & prev_bearish
        
        # 3. Gap Size
        gap_size = pd.Series(0.0, index=df.index)
        gap_size[valid_bullish] = df['low'] - anchor_high
        gap_size[valid_bearish] = anchor_low - df['high']
        
        # 4. Min Gap Filter
        min_gap = np.maximum(
            anchor_close * self.min_gap_pct,
            atr * self.atr_multiplier
        )
        size_filter = gap_size > min_gap
        
        # 5. Volume Filter
        vol_filter = prev_vol > vol_threshold
        
        # 6. Momentum Filter
        prev_range = prev_high - prev_low
        prev_body = np.abs(prev_close - prev_open)
        momentum = prev_body / prev_range.replace(0, 1) # Avoid div/0
        mom_filter = momentum > self.momentum_threshold
        
        # --- Combine Filters ---
        final_mask = size_filter & vol_filter & mom_filter & (valid_bullish | valid_bearish)
        
        # --- Construct Result ---
        result = pd.DataFrame(index=df.index)
        result['fvg_type'] = None
        result.loc[final_mask & valid_bullish, 'fvg_type'] = 'BULLISH'
        result.loc[final_mask & valid_bearish, 'fvg_type'] = 'BEARISH'
        
        result['fvg_gap_top'] = np.nan
        result.loc[final_mask & valid_bullish, 'fvg_gap_top'] = df['low']
        result.loc[final_mask & valid_bearish, 'fvg_gap_top'] = anchor_low
        
        result['fvg_gap_bottom'] = np.nan
        result.loc[final_mask & valid_bullish, 'fvg_gap_bottom'] = anchor_high
        result.loc[final_mask & valid_bearish, 'fvg_gap_bottom'] = df['high']
        
        # Metrics
        result['fvg_confidence'] = np.nan 
        
        # Vectorized confidence
        gap_pct = gap_size / anchor_close
        gap_score = (gap_pct / 0.01).clip(upper=1.0)
        vol_ratio = (prev_vol / avg_vol).replace([np.inf, -np.inf], 0).fillna(0)
        vol_score = ((vol_ratio - 1.0) / 1.0).clip(upper=1.0)
        
        conf = (gap_score * 0.4 + vol_score * 0.3 + momentum * 0.3)
        result.loc[final_mask, 'fvg_confidence'] = conf
        
        return result

    def plot(self, data: pd.DataFrame, **kwargs) -> None:
        """Placeholder for plotting logic"""
        pass

