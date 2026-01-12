import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from typing import Dict, List, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.covariance import LedoitWolf
import logging

logger = logging.getLogger(__name__)

class ClusterManager:
    """
    Manages asset correlations using Hierarchical Risk Parity (HRP) logic.
    Performs 'BTC-Detrending' to cluster assets based on idiosyncratic behavior
    rather than market beta.
    """
    
    def __init__(self, benchmark_symbol: str = 'BTCUSDT', lookback_days: int = 30):
        self.benchmark_symbol = benchmark_symbol
        self.lookback_days = lookback_days
        self.cluster_map: Dict[str, int] = {} # {Symbol: ClusterID}
        self.last_update_time = None
        
        # Risk Limits
        self.max_pos_per_cluster = 2 # Max active trades per cluster

    def update_clusters(self, price_df: pd.DataFrame, method: str = 'spearman', use_denoising: bool = False) -> Dict[str, int]:
        """
        Re-calculates clusters based on recent price history.
        
        Args:
            price_df: DataFrame with datetime index and columns = symbols.
            method: Correlation method ('pearson', 'kendall', 'spearman'). Default 'spearman' (Upgraded).
            use_denoising: If True, applies Ledoit-Wolf shrinkage (Upgraded).
        """
        if price_df.empty or self.benchmark_symbol not in price_df.columns:
            logger.warning("Invalid data for clustering. Missing benchmark or empty.")
            return {}

        # 1. Calculate Log Returns
        returns = np.log(price_df / price_df.shift(1)).dropna()
        
        if len(returns) < 50: # Need sufficient samples for correlation
            logger.warning("Not enough data points for robust clustering.")
            return {}

        # 2. Detrending (Remove BTC Beta)
        # We want to cluster based on "Altcoin Behavior", not "Following BTC"
        benchmark_ret = returns[self.benchmark_symbol].values.reshape(-1, 1)
        residual_returns = pd.DataFrame(index=returns.index)
        
        for col in returns.columns:
            if col == self.benchmark_symbol:
                continue
                
            asset_ret = returns[col].values
            # Run fast Linear Regression
            # y = beta * x + alpha + residual
            # we want the residual
            reg = LinearRegression().fit(benchmark_ret, asset_ret)
            pred = reg.predict(benchmark_ret)
            residuals = asset_ret - pred
            
            residual_returns[col] = residuals

        # 3. Correlation Distance Matrix
        # Distance = sqrt(2 * (1 - Correlation))
        # Range: 0 (Identical) to 2 (Inverse)
        
        # Drop columns with zero variance (constant residuals) to avoid NaNs
        valid_cols = [c for c in residual_returns.columns if residual_returns[c].std() > 1e-8]
        if len(valid_cols) < 2:
            logger.warning("Not enough assets with variance for clustering.")
            return {}
            
        residual_active = residual_returns[valid_cols]
        
        # 3. Correlation Matrix Calculation (with Upgrades)
        if use_denoising:
            # Upgrade 2: Ledoit-Wolf Shrinkage (Noise Reduction)
            # Fits a shrunk covariance matrix and converts it to correlation
            try:
                lw = LedoitWolf()
                # LedoitWolf expects array-like, returns covariance matrix
                cov_matrix = lw.fit(residual_active).covariance_
                
                # Convert Covariance to Correlation: Corr_ij = Cov_ij / (std_i * std_j)
                d = np.sqrt(np.diag(cov_matrix))
                corr_matrix_vals = cov_matrix / np.outer(d, d)
                
                corr_matrix = pd.DataFrame(
                    corr_matrix_vals, 
                    index=residual_active.columns, 
                    columns=residual_active.columns
                )
                logger.debug("Applied Ledoit-Wolf Denoising")
            except Exception as e:
                logger.error(f"Ledoit-Wolf failed ({e}), falling back to {method}")
                corr_matrix = residual_active.corr(method=method)
        else:
            # Upgrade 1: Spearman Rank Correlation (Robust to Outliers)
            # Defaulted via method='spearman'
            corr_matrix = residual_active.corr(method=method)
        
        # distance = sqrt(2(1-corr))
        # Clip correlation to [-1, 1] to avoid floating point errors slightly outside range
        dist_matrix = np.sqrt(2 * (1 - corr_matrix.clip(-1.0, 1.0)))
        
        # SAFETY: Ensure diagonal is exactly 0 and fill NaNs with max distance (2.0)
        np.fill_diagonal(dist_matrix.values, 0.0)
        dist_matrix = dist_matrix.fillna(2.0)
        
        # 4. Hierarchical Clustering (Ward's Method)
        # Convert to condensed distance matrix for scipy
        condensed_dist = squareform(dist_matrix.values, checks=False)
        linkage_matrix = linkage(condensed_dist, method='ward')
        
        # 5. Form Clusters (Cut the Tree)
        # We use a distance threshold. 
        # t=1.5 is a heuristic; can be tuned or use max_clust
        # Alternatively, we can force K clusters relative to universe size
        n_assets = len(residual_active.columns)
        n_clusters = max(3, int(n_assets / 8)) # Avg 8 coins per cluster
        
        labels = fcluster(linkage_matrix, t=n_clusters, criterion='maxclust')
        
        # 6. Build Map
        self.cluster_map = {
            col: int(label) 
            for col, label in zip(residual_active.columns, labels)
        }
        
        # Add BTC to its own unique cluster (0)
        self.cluster_map[self.benchmark_symbol] = 0
        
        # Logging
        cluster_counts = pd.Series(list(self.cluster_map.values())).value_counts()
        logger.info(f"Updated Clusters: Formed {len(cluster_counts)} groups.")
        logger.debug(f"Cluster Sizes: \n{cluster_counts}")
        
        return self.cluster_map

    def check_risk_allowance(self, symbol: str, current_positions: List[str]) -> bool:
        """
        Checks if opening a trade for 'symbol' violates cluster risk limits.
        
        Args:
            symbol: The candidate coin.
            current_positions: List of symbols currently held.
        """
        if symbol not in self.cluster_map:
            # If unknown, allow it but log warning (or map to 'Misc' cluster)
            logger.warning(f"Symbol {symbol} not in cluster map. Treating as neutral.")
            return True
            
        target_cluster = self.cluster_map[symbol]
        
        # Count how many current positions are in this cluster
        count = 0
        for pos in current_positions:
            if self.cluster_map.get(pos) == target_cluster:
                count += 1
                
        if count >= self.max_pos_per_cluster:
            logger.info(f"Risk Reject {symbol}: Cluster {target_cluster} full ({count}/{self.max_pos_per_cluster})")
            return False
            
        return True

    def get_cluster_mates(self, symbol: str) -> List[str]:
        """Returns all symbols in the same cluster"""
        if symbol not in self.cluster_map: return []
        target = self.cluster_map[symbol]
        return [s for s, c in self.cluster_map.items() if c == target and s != symbol]
