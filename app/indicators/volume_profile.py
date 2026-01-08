import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from app.indicators.BaseIndicatorInterface import BaseIndicatorInterface
import numpy as np
from numba import njit

@njit(cache=False)
def _calc_value_area(hist, bin_centers, total_vol, n_bins):
    # POC
    max_idx = np.argmax(hist)
    poc = bin_centers[max_idx]
    
    # Value Area (70%)
    target = total_vol * 0.7
    curr = hist[max_idx]
    l, r = max_idx, max_idx
    
    while curr < target:
        v_l = hist[l-1] if l > 0 else 0
        v_r = hist[r+1] if r < n_bins - 1 else 0
        
        if v_r > v_l:
            r += 1
            curr += v_r
            if r >= n_bins - 1: break
        else:
            l -= 1
            curr += v_l
            if l <= 0: break
            
    return poc, bin_centers[r], bin_centers[l]

@njit(cache=False)
def _calc_moments(hist, bin_centers, total_vol, n_bins):
    ent = 0.0
    mean_p = 0.0
    
    # 1st Pass: Entropy & Mean
    for k in range(n_bins):
        val = hist[k]
        if val > 0:
            p = val / total_vol
            ent -= p * np.log2(p)
            mean_p += bin_centers[k] * p
    
    norm_factor = np.log2(n_bins)
    entropy = ent / norm_factor if norm_factor > 0 else 0.0
        
    # 2nd Pass: Variance, Skew, Kurtosis
    var = 0.0
    m3 = 0.0
    m4 = 0.0
    
    for k in range(n_bins):
        val = hist[k]
        if val > 0:
            p = val / total_vol
            diff = bin_centers[k] - mean_p
            diff_sq = diff * diff
            var += diff_sq * p
            m3 += (diff_sq * diff) * p
            m4 += (diff_sq * diff_sq) * p
    
    std = np.sqrt(var)
    skew = 0.0
    kurt = 0.0
    
    if std > 0:
        skew = m3 / (std**3)
        kurt = (m4 / (std**4)) - 3
        
    return entropy, skew, kurt

@njit(cache=False)
def _rolling_profile_kernel(
    prices, volumes, lows, highs, 
    lookbacks, 
    min_p, bin_size, n_bins
):
    """
    Computes Rolling POC, VA, Entropy, Skew, Kurtosis.
    Optimized for O(N) performance using Sliding Window.
    """
    n = len(prices)
    pocs = np.full(n, np.nan)
    vahs = np.full(n, np.nan)
    vals = np.full(n, np.nan)
    entropy = np.full(n, np.nan)
    skew = np.full(n, np.nan)
    kurt = np.full(n, np.nan)
    
    bin_centers = min_p + (np.arange(n_bins) + 0.5) * bin_size
    
    # Persistent Histogram State for Sliding Window
    hist = np.zeros(n_bins)
    current_start = 0
    
    for i in range(n):
        L = lookbacks[i]
        # Safety: Ensure L is reasonable
        if L < 1: L = 1
        if L > n: L = n
        
        # Ensure target_start is non-negative
        target_start = i - L + 1
        if target_start < 0: target_start = 0
        
        # --- 1. Add New Bar (i) ---
        p_l, p_h, vol = lows[i], highs[i], volumes[i]
        
        # Handle NaNs in data by skipping
        if np.isnan(p_l) or np.isnan(p_h) or np.isnan(vol):
            # If data is bad, we just don't add it to hist, 
            # but we still need to advance window logic
            pass 
        else:
            idx_l = int((p_l - min_p) / bin_size)
            idx_h = int((p_h - min_p) / bin_size)
            
            # Clamp to grid
            idx_l = max(0, min(idx_l, n_bins - 1))
            idx_h = max(0, min(idx_h, n_bins - 1))
            
            if idx_h == idx_l:
                hist[idx_l] += vol
            else:
                v_share = vol / (idx_h - idx_l + 1)
                for b in range(idx_l, idx_h + 1):
                    hist[b] += v_share
                
        # --- 2. Remove Old Bars (Sliding Window) ---
        while current_start < target_start:
            p_l_old, p_h_old, vol_old = lows[current_start], highs[current_start], volumes[current_start]
            
            if not (np.isnan(p_l_old) or np.isnan(p_h_old) or np.isnan(vol_old)):
                idx_l_old = int((p_l_old - min_p) / bin_size)
                idx_h_old = int((p_h_old - min_p) / bin_size)
                idx_l_old = max(0, min(idx_l_old, n_bins - 1))
                idx_h_old = max(0, min(idx_h_old, n_bins - 1))
                
                if idx_h_old == idx_l_old:
                    hist[idx_l_old] -= vol_old
                else:
                    v_share_old = vol_old / (idx_h_old - idx_l_old + 1)
                    for b in range(idx_l_old, idx_h_old + 1):
                        hist[b] -= v_share_old
            
            current_start += 1
            
        # --- 3. Handle Lookback Expansion ---
        while current_start > target_start:
            current_start -= 1
            p_l_add, p_h_add, vol_add = lows[current_start], highs[current_start], volumes[current_start]
            
            if not (np.isnan(p_l_add) or np.isnan(p_h_add) or np.isnan(vol_add)):
                idx_l_add = int((p_l_add - min_p) / bin_size)
                idx_h_add = int((p_h_add - min_p) / bin_size)
                idx_l_add = max(0, min(idx_l_add, n_bins - 1))
                idx_h_add = max(0, min(idx_h_add, n_bins - 1))
                
                if idx_h_add == idx_l_add:
                    hist[idx_l_add] += vol_add
                else:
                    v_share_add = vol_add / (idx_h_add - idx_l_add + 1)
                    for b in range(idx_l_add, idx_h_add + 1):
                        hist[b] += v_share_add

        # --- 4. Statistics ---
        total_vol = np.sum(hist)
        if total_vol <= 0: continue
            
        # Call Helpers
        pocs[i], vahs[i], vals[i] = _calc_value_area(hist, bin_centers, total_vol, n_bins)
        entropy[i], skew[i], kurt[i] = _calc_moments(hist, bin_centers, total_vol, n_bins)
            
    return pocs, vahs, vals, entropy, skew, kurt


class VolumeProfile(BaseIndicatorInterface):
    """
    Volume Profile Module ("The Map").
    Calculates the volume distribution over price levels to identify
    High Volume Nodes (HVN) and Low Volume Nodes (LVN).
    Supports both Fixed Lookback (Composite) and Session-Based (Dynamic) modes.
    Includes Developing POC/VA generation for plotting.
    """


    def __init__(self, name: str = "VolumeProfile", **kwargs):
        super().__init__(name, **kwargs)
        self.bins = kwargs.get('bins', 100)
        self.lookback = kwargs.get('lookback', 200) # Fixed bars to look back (Composite Mode)
        self.session_mode = kwargs.get('session_mode', False) # If True, ignores lookback and uses session start
        self.hvn_percentile = kwargs.get('hvn_percentile', 70)
        self.lvn_percentile = kwargs.get('lvn_percentile', 30)
        
        # State to hold the calculated profile
        self.profile_df: Optional[pd.DataFrame] = None
        self.bin_height: float = 0.0
        self.min_price: float = 0.0
        self.poc_price: float = 0.0


    def _get_default_params(self):
        return {
            'bins': {'type': 'int', 'default': 100, 'min': 10, 'max': 500, 'description': 'Number of Price Bins'},
            'lookback': {'type': 'int', 'default': 200, 'min': 50, 'max': 1000, 'description': 'Lookback Period (Bars)'},
            'session_mode': {'type': 'boolean', 'default': False, 'description': 'Use Daily Session (Reset at Midnight)'},
            'hvn_percentile': {'type': 'int', 'default': 70, 'min': 50, 'max': 95, 'description': 'HVN Threshold %'},
            'lvn_percentile': {'type': 'int', 'default': 30, 'min': 5, 'max': 50, 'description': 'LVN Threshold %'}
        }

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculates the Volume Profile (Static Snapshot).
        Features:
        1. Range-Based Volume Distribution (Fixes 'Spiky' profiles).
        2. Session-Aware Slicing (Optional).
        """
        df = data.copy()
        
        # --- 1. Determine Data Subset ---
        if self.session_mode:
            # Logic: Find the last time the day changed (Session Start)
            # Assuming DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                # Fallback if no datetime index: use full data or error
                subset = df
            else:
                # Get start of current day (Normalize to Midnight)
                last_ts = df.index[-1]
                start_of_day = last_ts.normalize()
                subset = df[df.index >= start_of_day]
                
                # Safety check: if new session just started (empty), use last 10 bars
                if subset.empty:
                    subset = df.iloc[-10:]
        else:
            # Standard Fixed Lookback (Composite Profile)
            subset = df.iloc[-self.lookback:] if len(df) > self.lookback else df


        if subset.empty: return df


        # --- 2. Define Price Bins ---
        min_p = subset['low'].min()
        max_p = subset['high'].max()
        
        if min_p == max_p: return df
            
        self.min_price = min_p
        # Create bins
        bins = np.linspace(min_p, max_p, self.bins + 1)
        self.bin_height = bins[1] - bins[0]
        
        # --- 3. Distribute Volume (Range-Based Logic) ---
        # Instead of 1 bin per candle, we find Start Bin (Low) and End Bin (High)
        low_idxs = np.digitize(subset['low'].values, bins) - 1
        high_idxs = np.digitize(subset['high'].values, bins) - 1
        
        # Clip to valid range
        low_idxs = np.clip(low_idxs, 0, self.bins - 1)
        high_idxs = np.clip(high_idxs, 0, self.bins - 1)
        
        volume_by_bin = np.zeros(self.bins)
        vol_values = subset['volume'].values
        
        # Optimized Loop for Volume Spreading
        # Logic: If candle covers 5 bins, spread volume/5 to each bin.
        for l, h, vol in zip(low_idxs, high_idxs, vol_values):
            if h == l:
                volume_by_bin[l] += vol
            else:
                span = h - l + 1
                vol_per_bin = vol / span
                # Fast numpy slice assignment
                volume_by_bin[l : h+1] += vol_per_bin
            
        # --- 4. Create Profile DataFrame ---
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        self.profile_df = pd.DataFrame({
            'price': bin_centers,
            'volume': volume_by_bin
        })
        
        # --- 5. Calculate Statistics (POC, Thresholds) ---
        
        # A. POC (Point of Control) - Price with Max Volume
        max_vol_idx = self.profile_df['volume'].idxmax()
        self.poc_price = self.profile_df.iloc[max_vol_idx]['price']
        
        # B. Thresholds (Ignore zero volume bins for cleaner stats)
        nonzero_vols = self.profile_df[self.profile_df['volume'] > 0]['volume']
        
        if nonzero_vols.empty:
            hvn_thresh = 0
            lvn_thresh = 0
        else:
            hvn_thresh = np.percentile(nonzero_vols, self.hvn_percentile)
            lvn_thresh = np.percentile(nonzero_vols, self.lvn_percentile)
        
        # C. Assign Status
        def get_status(vol):
            if vol >= hvn_thresh: return 'HVN'
            if vol <= lvn_thresh: return 'LVN' # Includes Vacuum zones
            return 'NEUTRAL'
            
        self.profile_df['status'] = self.profile_df['volume'].apply(get_status)
        
        return df


    def calculate_developing_series(self, data: pd.DataFrame, dynamic_lookbacks: np.ndarray = None) -> pd.DataFrame:
        """
        Calculates Rolling POC/VAH/VAL using the optimized Numba kernel.
        Supports Dynamic Lookback (Adaptive Length).
        """
        # 1. Prepare Data for Numba
        # Ensure data is float64 and fill NaNs to avoid Numba issues
        prices = np.nan_to_num(data['close'].values, nan=0.0).astype(np.float64)
        volumes = np.nan_to_num(data['volume'].values, nan=0.0).astype(np.float64)
        lows = np.nan_to_num(data['low'].values, nan=0.0).astype(np.float64)
        highs = np.nan_to_num(data['high'].values, nan=0.0).astype(np.float64)
        
        n = len(prices)
        
        # 2. Handle Lookbacks
        if dynamic_lookbacks is None:
            # Static mode: Create array of constant lookback
            lookbacks = np.full(n, self.lookback, dtype=np.int64)
        else:
            # Dynamic mode: Ensure it's int array and matches length
            # Fill NaNs in lookbacks with default lookback
            lookbacks = np.nan_to_num(dynamic_lookbacks, nan=self.lookback).astype(np.int64)
            
            if len(lookbacks) != n:
                # Pad or trim if lengths mismatch (e.g. due to indicator warmup)
                # For safety, we default to self.lookback if mismatch
                lookbacks = np.full(n, self.lookback, dtype=np.int64)
            
            # Clip lookbacks to valid range [1, n] to prevent infinite loops
            lookbacks = np.clip(lookbacks, 1, n)

        # 3. Define Global Bins (based on entire dataset range for stability)
        # Filter out 0s if they were result of nan_to_num and look weird
        valid_mask = (lows > 0) & (highs > 0)
        if not np.any(valid_mask):
             return pd.DataFrame(index=data.index, columns=['developing_poc', 'developing_vah', 'developing_val'])
             
        min_p = lows[valid_mask].min()
        max_p = highs[valid_mask].max()
        
        if min_p >= max_p:
            return pd.DataFrame(index=data.index, columns=['developing_poc', 'developing_vah', 'developing_val'])

        bin_size = (max_p - min_p) / self.bins
        
        # 4. Run Optimized Kernel
        print(f"DEBUG: Running Optimized Numba Kernel with n={n}, bins={self.bins}, bin_size={bin_size:.4f}")
        pocs, vahs, vals, entropy, skew, kurt = _rolling_profile_kernel(
            prices, volumes, lows, highs,
            lookbacks,
            min_p, bin_size, self.bins
        )
        
        # 5. Return DataFrame
        return pd.DataFrame({
            'developing_poc': pocs,
            'developing_vah': vahs,
            'developing_val': vals,
            'entropy': entropy,
            'skew': skew,
            'kurt': kurt
        }, index=data.index)


    def get_volume_context(self, price: float) -> str:
        """
        Returns the context (HVN/LVN/NEUTRAL) for a given price level.
        """
        if self.profile_df is None or self.profile_df.empty:
            return "NEUTRAL"
            
        if price < self.min_price or self.bin_height == 0:
            return "LVN" # Assume vacuum outside range
            
        idx = int((price - self.min_price) / self.bin_height)
        
        # Clamp index
        if idx < 0: idx = 0
        if idx >= len(self.profile_df): idx = len(self.profile_df) - 1
        
        return self.profile_df.iloc[idx]['status']


    def get_poc(self) -> float:
        """Returns the Point of Control (Price with Max Volume)"""
        return self.poc_price


    def plot(self, data: pd.DataFrame, **kwargs) -> None:
        """
        Plotting is not yet implemented for Volume Profile.
        """
        pass

    
    def get_entropy(self) -> float:
        """
        Calculates Shannon Entropy of the current Volume Profile.
        Measures market "chaos" / efficiency.
        
        Returns: A normalized value between 0.0 and 1.0
        - 0.0 to 0.5: Organized (Bell Curve) - Mean Reversion Market
        - 0.5 to 0.7: Neutral
        - 0.7 to 1.0: Chaotic (Flat/Multi-Modal) - Breakout Market
        """
        if self.profile_df is None or self.profile_df.empty:
            return 0.5  # Neutral default
            
        # 1. Get Probability Distribution
        volumes = self.profile_df['volume'].values
        total_vol = np.sum(volumes)
        
        if total_vol == 0:
            return 0.5  # No volume = neutral
        
        probs = volumes / total_vol
        
        # 2. Filter out zeros to avoid log(0) error
        probs = probs[probs > 0]
        
        if len(probs) == 0:
            return 0.5
        
        # 3. Calculate Shannon Entropy
        # Formula: H = -Σ(p_i * log2(p_i))
        entropy = -np.sum(probs * np.log2(probs))
        
        # 4. Normalize to [0, 1]
        # Max entropy occurs when distribution is uniform (all bins equal)
        # Max = log2(N) where N is number of non-zero bins
        max_entropy = np.log2(len(probs))
        
        if max_entropy == 0:
            return 0.5
        
        normalized_entropy = entropy / max_entropy
        
        return normalized_entropy


    def get_skewness(self) -> float:
        """
        Calculates the Skewness of the Volume Profile.
        Detects "Trap" scenarios (Top-Heavy vs Bottom-Heavy distribution).
        
        Returns:
        > +0.5 : b-Shape (Bottom Heavy) - Bullish Accumulation / Trapped Shorts
        -0.5 to +0.5 : Balanced
        < -0.5 : P-Shape (Top Heavy) - Distribution / Trapped Longs
        """
        if self.profile_df is None or self.profile_df.empty:
            return 0.0
            
        # We treat 'price' as the random variable and 'volume' as weights
        prices = self.profile_df['price'].values
        volumes = self.profile_df['volume'].values
        
        # Filter out zero volume bins
        mask = volumes > 0
        prices = prices[mask]
        volumes = volumes[mask]
        
        if len(prices) == 0:
            return 0.0
        
        # Weighted Mean
        total_vol = np.sum(volumes)
        if total_vol == 0:
            return 0.0
        
        mean_price = np.average(prices, weights=volumes)
        
        # Weighted Variance
        variance = np.average((prices - mean_price)**2, weights=volumes)
        std_dev = np.sqrt(variance)
        
        if std_dev == 0:
            return 0.0  # No spread = symmetric
        
        # Weighted Skewness
        # Formula: Skew = E[(X - μ)³] / σ³
        skew = np.average((prices - mean_price)**3, weights=volumes) / (std_dev**3)
        
        return skew


    def get_regime_classification(self) -> Dict[str, Any]:
        entropy = self.get_entropy()
        skewness = self.get_skewness()
        kurtosis = self.get_kurtosis()
        
        # --- The "Margin" Logic ---
        # 1. Is it a Range? (Organized OR Strongly Peaked)
        # We allow higher entropy (0.65) IF kurtosis is high (> 1.0)
        is_organized = (entropy < 0.6) or ((entropy < 0.7) and (kurtosis > 1.0))
        
        if is_organized:
            # We are in a TRADABLE RANGE (Bell-Curve-ish)
            
            if abs(skewness) < 0.5:
                regime = 'BALANCED_RANGE'
                strategy = 'MEAN_REVERSION' # Fade edges to POC
            elif skewness > 0.5:
                # b-Shape (Bottom Heavy) -> Range with Bullish Bias
                regime = 'ACCUMULATION_RANGE'
                strategy = 'BUY_DIPS' # Don't short the top, only buy bottom
            else:
                # P-Shape (Top Heavy) -> Range with Bearish Bias
                regime = 'DISTRIBUTION_RANGE'
                strategy = 'SELL_RALLIES' # Don't buy bottom, only short top
                
        else:
            # High Entropy + Low Kurtosis = CHAOS / BREAKOUT MODE
            # The profile is Flat or Multi-Modal
            
            if kurtosis < -1.0:
                # Extremely flat profile (Thin trading everywhere)
                regime = 'THIN_PROFILE' 
                strategy = 'VOLATILITY_EXPANSION' # Any move will be fast
            else:
                regime = 'TREND_PENDING'
                strategy = 'WAIT_FOR_BREAKOUT'

        return {
            'entropy': entropy,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'regime': regime,
            'strategy': strategy
        }


    def get_kurtosis(self) -> float:
        """
        Calculates Fisher-Pearson Coefficient of Skewness (Weighted).
        Measures the "Peakedness" of the profile.
        
        Returns:
        > 1.0 : Leptokurtic (Sharp Peak / Fat Tails) -> STRONG POC (Good for Range Trading)
        ~ 0.0 : Mesokurtic (Normal Distribution)
        < -1.0 : Platykurtic (Flat / Square) -> WEAK POC (Risk of Breakout)
        """
        if self.profile_df is None or self.profile_df.empty:
            return 0.0
            
        prices = self.profile_df['price'].values
        volumes = self.profile_df['volume'].values
        
        # Filter zeros
        mask = volumes > 0
        prices = prices[mask]
        volumes = volumes[mask]
        
        if len(prices) < 2: return 0.0
        
        total_vol = np.sum(volumes)
        mean_price = np.average(prices, weights=volumes)
        
        variance = np.average((prices - mean_price)**2, weights=volumes)
        std_dev = np.sqrt(variance)
        
        if std_dev == 0: return 0.0
        
        # Weighted Kurtosis Formula: E[(x-u)^4] / sigma^4  - 3 (Excess)
        fourth_moment = np.average((prices - mean_price)**4, weights=volumes)
        kurtosis = (fourth_moment / (std_dev**4)) - 3
        
        return kurtosis

    def get_liquidity_features(self, smooth_window: int = 3) -> Dict[str, np.ndarray]:
        """
        Computes Liquidity Walls and Holes from Volume Profile curvature.
        Returns arrays aligned with profile_df['price'].
        """
        if self.profile_df is None or self.profile_df.empty:
            return {}

        vol = self.profile_df['volume'].values.astype(float)

        # Optional: light smoothing to reduce single-bin noise
        if smooth_window > 1:
            kernel = np.ones(smooth_window) / smooth_window
            vol_smooth = np.convolve(vol, kernel, mode='same')
        else:
            vol_smooth = vol

        # First derivative: change in volume between neighboring price levels
        dV = np.gradient(vol_smooth)

        # Second derivative: curvature (how fast the change itself is changing)
        d2V = np.gradient(dV)

        # Heuristics:
        # Liquidity Wall (HVN) = Peak = Negative Curvature (Concave Down)
        # Liquidity Hole (LVN) = Valley = Positive Curvature (Concave Up)
        
        # Walls: We look for strong NEGATIVE curvature (Peaks)
        wall_threshold = np.percentile(d2V, 10)   # Bottom 10% (Most Negative)
        
        # Holes: We look for strong POSITIVE curvature (Valleys)
        hole_threshold = np.percentile(d2V, 90)   # Top 10% (Most Positive)

        walls = (d2V <= wall_threshold).astype(int)
        holes = (d2V >= hole_threshold).astype(int)

        self.profile_df['liq_wall'] = walls
        self.profile_df['liq_hole'] = holes

        return {
            'curvature': d2V,
            'walls': walls,
            'holes': holes
        }

    def get_features(self, df: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """
        Extracts normalized features for the Bot Brain.
        """
        if self.profile_df is None or kwargs.get('force_recalc', False):
            self.calculate(df)
            
        current_price = df['close'].iloc[-1]
        
        # 1. Location Metrics (Where are we?)
        # We need the *Latest* Developing VA if available, or Static VA
        # For consistency with the "Senses" architecture, we use the Static Profile logic
        # but mapped to the current price.
        
        # Get Key Levels
        poc = self.get_poc()
        
        # Calculate VAH/VAL from profile_df (Snapshot)
        # Note: Your calculate() method computes these but doesn't store them as public props yet
        # We can quickly derive them from profile_df
        total_vol = self.profile_df['volume'].sum()
        target = total_vol * 0.7
        
        # Sort by volume desc to find value area bins
        sorted_df = self.profile_df.sort_values('volume', ascending=False)
        sorted_df['cum_vol'] = sorted_df['volume'].cumsum()
        
        # Handle empty profile case
        if sorted_df.empty:
             return {}

        value_area_df = sorted_df[sorted_df['cum_vol'] <= target]
        if not value_area_df.empty:
            vah = value_area_df['price'].max()
            val = value_area_df['price'].min()
        else:
            vah = poc
            val = poc
            
        # Feature: Normalized Position in Value Area
        # 0.0 = VAL, 1.0 = VAH, 0.5 = POC
        va_range = vah - val
        # Use a small epsilon to avoid division by zero
        if abs(va_range) < 1e-9:
            vp_pos = 0.5 
        else:
            vp_pos = (current_price - val) / va_range
            
        # 2. Regime Metrics (From Histogram Shape)
        skew = self.get_skewness()
        kurt = self.get_kurtosis()
        entropy = self.get_entropy()
        
        # 3. Micro-Structure (Wall vs Hole)
        # Check if current price is inside a Liquidity Wall or Hole
        liq_context = 0.0 # 0 = Neutral
        
        # Locate bin
        if self.bin_height > 1e-9: # Safe check
            idx = int((current_price - self.min_price) / self.bin_height)
            # Clip index to be safe
            idx = max(0, min(idx, len(self.profile_df) - 1))
            
            # Check previously calculated walls/holes
            if 'liq_wall' not in self.profile_df.columns:
                self.get_liquidity_features() # lazy calc
            
            # Additional safety: ensure columns exist after call
            if 'liq_wall' in self.profile_df.columns:
                is_wall = self.profile_df.iloc[idx]['liq_wall']
                is_hole = self.profile_df.iloc[idx]['liq_hole']
                
                if is_wall > 0: liq_context = 1.0   # Support/Resistance friction
                if is_hole > 0: liq_context = -1.0  # Slip zone (Acceleration)

        return {
            'vp_pos': float(vp_pos),       # <0 (Oversold), 0-1 (Range), >1 (Overbought)
            'vp_skew': float(skew),        # >0.5 (Bullish Accumulation), <-0.5 (Bearish Dist)
            'vp_kurt': float(kurt),        # >1 (Strong Reversion), <0 (Breakout Prone)
            'vp_entropy': float(entropy),  # >0.8 (Chaos/Trend), <0.6 (Range)
            'vp_liq_type': float(liq_context), # 1 (Wall), -1 (Hole)
            'vp_poc': float(poc),
            'vp_vah': float(vah),
            'vp_val': float(val)
        }


    def _get_plot_trace(self, data: pd.DataFrame, **kwargs) -> List[Dict[str, Any]]:
        """
        Generates Plotly traces for the frontend.
        Uses Developing POC/VAH/VAL for time-series plotting.
        """
        # Calculate developing series
        dev_series = self.calculate_developing_series(data)
        
        x_dates = dev_series.index.tolist()
        
        traces = []
        
        # POC
        traces.append({
            'x': x_dates,
            'y': dev_series['developing_poc'].tolist(),
            'type': 'scatter',
            'mode': 'lines',
            'name': 'Developing POC',
            'line': {'color': 'red', 'width': 2}
        })
        
        # VAH
        traces.append({
            'x': x_dates,
            'y': dev_series['developing_vah'].tolist(),
            'type': 'scatter',
            'mode': 'lines',
            'name': 'Developing VAH',
            'line': {'color': 'green', 'width': 1, 'dash': 'dot'}
        })
        
        # VAL
        traces.append({
            'x': x_dates,
            'y': dev_series['developing_val'].tolist(),
            'type': 'scatter',
            'mode': 'lines',
            'name': 'Developing VAL',
            'line': {'color': 'green', 'width': 1, 'dash': 'dot'}
        })
        
        return traces




