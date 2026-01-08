import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

class PathfindingTrendline:
    """
    Finds the 'Optimal' Trendline using Graph Theory / Dynamic Programming.
    Unlike Linear Regression, this guarantees the line respects price structure (no body cuts).
    """

    def __init__(self, pivot_window: int = 3):
        self.pivot_window = pivot_window

    def find_resistance_path(self, df: pd.DataFrame) -> List[Tuple[int, float]]:
        """
        Finds the optimal Resistance Line (connecting Highs).
        Returns list of points [(index, price), (index, price)...]
        """
        highs = df['high'].values
        opens = df['open'].values
        closes = df['close'].values
        n = len(df)

        # 1. Identify Pivot Highs
        pivots = []
        # Vectorized pivot check is possible, but loop is readable for this logic
        for i in range(self.pivot_window, n - self.pivot_window):
            window = highs[i-self.pivot_window : i+self.pivot_window+1]
            if highs[i] == np.max(window):
                pivots.append(i)

        if len(pivots) < 2:
            return []

        # 2. Dynamic Programming State
        # best_path[i] = {score: float, prev_node: int_idx}
        # Score = Total duration (index distance) covered by valid lines ending at i
        dp = {p: {'score': 0, 'prev': -1} for p in pivots}

        # 3. Build the Graph (Look backwards)
        for i in range(len(pivots)):
            curr_idx = pivots[i]
            curr_price = highs[curr_idx]
            
            # Try to connect to all previous pivots
            # Optimization: Don't look back infinitely. Maybe last 50 candles?
            # For now, we look at all previous pivots for max accuracy.
            for j in range(i):
                prev_idx = pivots[j]
                prev_price = highs[prev_idx]
                
                # Check Validity: Does line cut bodies?
                slope = (curr_price - prev_price) / (curr_idx - prev_idx)
                intercept = prev_price - (slope * prev_idx)
                
                is_valid = True
                # Scan all bars between prev and curr
                # This makes it O(N^3) technically, but N=pivots is small.
                for k in range(prev_idx + 1, curr_idx):
                    line_y = slope * k + intercept
                    body_top = max(opens[k], closes[k])
                    
                    if line_y < body_top: # CUT DETECTED
                        is_valid = False
                        break
                
                if is_valid:
                    # Score calculation: Length of this segment
                    segment_len = curr_idx - prev_idx
                    new_score = dp[prev_idx]['score'] + segment_len
                    
                    if new_score > dp[curr_idx]['score']:
                        dp[curr_idx]['score'] = new_score
                        dp[curr_idx]['prev'] = prev_idx

        # 4. Reconstruct the Winner
        # Find the pivot that has the highest total score (Longest valid trend structure)
        best_end_node = max(dp, key=lambda k: dp[k]['score'])
        
        # If score is 0, no valid line exists > 2 points
        if dp[best_end_node]['score'] == 0:
            return []
            
        path = []
        curr = best_end_node
        while curr != -1:
            path.append((curr, highs[curr]))
            curr = dp[curr]['prev']
            
        path.reverse() # Sort by time
        return path

    def find_support_path(self, df: pd.DataFrame) -> List[Tuple[int, float]]:
        """
        Finds the optimal Support Line (connecting Lows).
        Same logic, inverted.
        """
        lows = df['low'].values
        opens = df['open'].values
        closes = df['close'].values
        n = len(df)

        pivots = []
        for i in range(self.pivot_window, n - self.pivot_window):
            window = lows[i-self.pivot_window : i+self.pivot_window+1]
            if lows[i] == np.min(window):
                pivots.append(i)

        if len(pivots) < 2: return []

        dp = {p: {'score': 0, 'prev': -1} for p in pivots}

        for i in range(len(pivots)):
            curr_idx = pivots[i]
            curr_price = lows[curr_idx]
            
            for j in range(i):
                prev_idx = pivots[j]
                prev_price = lows[prev_idx]
                
                slope = (curr_price - prev_price) / (curr_idx - prev_idx)
                intercept = prev_price - (slope * prev_idx)
                
                is_valid = True
                for k in range(prev_idx + 1, curr_idx):
                    line_y = slope * k + intercept
                    body_bottom = min(opens[k], closes[k])
                    
                    if line_y > body_bottom: # CUT DETECTED (Line went above body bottom)
                        is_valid = False
                        break
                
                if is_valid:
                    segment_len = curr_idx - prev_idx
                    new_score = dp[prev_idx]['score'] + segment_len
                    if new_score > dp[curr_idx]['score']:
                        dp[curr_idx]['score'] = new_score
                        dp[curr_idx]['prev'] = prev_idx

        best_end_node = max(dp, key=lambda k: dp[k]['score'])
        if dp[best_end_node]['score'] == 0: return []
            
        path = []
        curr = best_end_node
        while curr != -1:
            path.append((curr, lows[curr]))
            curr = dp[curr]['prev']
            
        path.reverse()
        return path

    def get_projected_line(self, path: List[Tuple[int, float]], last_bar_index: int) -> Optional[dict]:
        """
        Takes the best path and projects the final segment to the current bar.
        Returns the slope, intercept, and projected price at the last bar.
        """
        if len(path) < 2:
            return None
            
        # Get the last two points in the optimal path
        (idx_last, price_last) = path[-1]
        (idx_prev, price_prev) = path[-2]
        
        # Calculate Slope of the final winning trendline
        slope = (price_last - price_prev) / (idx_last - idx_prev)
        intercept = price_last - (slope * idx_last)
        
        # Project to current live bar
        projected_price = slope * last_bar_index + intercept
        
        return {
            'slope': slope,
            'intercept': intercept,
            'last_pivot_idx': idx_last,
            'current_price_projection': projected_price
        }

