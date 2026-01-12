import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, Optional

from app.db.db_handler import DBHandler
from app.db.mtf_data_manager import MTFDataManager  # Import MTFDataManager
from app.indicators.regime_metrices import RegimeMetrics
from app.indicators.hilbert_cycle import HilbertCycle

logger = logging.getLogger(__name__)

# Constants for analysis
ANALYSIS_LOOKBACK_LIMIT = 500

def get_lookback_delta(timeframe: str) -> timedelta:
    """Determine lookback period based on timeframe to ensure enough bars."""
    if timeframe == '15m': 
        return timedelta(days=7)
    elif timeframe == '1h':
        return timedelta(days=30)
    elif timeframe == '4h': 
        return timedelta(days=90)
    # Default fallback
    return timedelta(days=60) 

async def calculate_market_status_on_demand(
    symbol: str, 
    timeframe: str, 
    db_handler: DBHandler
) -> Dict[str, Any]:
    """
    Perform on-demand calculation of market status (Regime + Hilbert).
    Used when cached data is missing.
    """
    status = {}
    
    end_time = datetime.now()
    delta = get_lookback_delta(timeframe)
    start_time = end_time - delta
    
    # improved logging
    logger.info(f"Using lookback {delta} for {symbol} {timeframe}")
    
    # ALWAYS fetch 1m candles and resample, because DB likely only has 1m data
    # Increase limit to ensuring enough 1m bars for resampling
    # 500 bars * ratio (e.g. 60 for 1h) -> 30000 bars
    # But read_candles limit is hard count.
    # Set a high limit or rely on time range.
    
    candles = await db_handler.read_candles(
        symbol=symbol,
        interval='1m',  # Force 1m for source data
        start_time=start_time,
        end_time=end_time,
        limit=100000 # Fetch enough raw data
    )
    
    if candles and len(candles) > 100:
        # Convert asyncpg Records to dicts and enforce float types
        data_list = []
        for c in candles:
            d = dict(c)
            # Handle potential bucket alias from DB
            if 'bucket' in d:
                d['timestamp'] = d.pop('bucket')
            # Ensure floats
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in d:
                    d[col] = float(d[col])
            data_list.append(d)

        df = pd.DataFrame(data_list)
        
        if 'timestamp' in df.columns:
            df.set_index('timestamp', inplace=True)
            # Check if index is already tz-aware datetime (from DB)
            if not isinstance(df.index, pd.DatetimeIndex):
                # Only use unit='ms' if it looks like a number
                # But since DB returns datetime objects, simple to_datetime is usually enough
                # or just leave it if it's already datetime objects
                df.index = pd.to_datetime(df.index)
        
        # Resample if needed
        if timeframe != '1m':
             df = MTFDataManager.resample_ohlcv(df, timeframe)

        # Sort index just in case
        df.sort_index(inplace=True)
        
        # Check if we have enough data AFTER resampling
        if len(df) < 50:
            logger.warning(f"❌ Not enough data after resampling to {timeframe}: {len(df)} bars")
            return {}

        try:
            # A. Run Regime Metrics
            regime = RegimeMetrics()
            regime_feats = regime.get_features(df)
            regime_meta = regime.metrics_df.iloc[-1]
            
            # B. Run Hilbert Cycle
            hilbert = HilbertCycle()
            hilbert_feats = hilbert.get_features(df)
            
            # Merge into status dict (Normalized keys for Frontend)
            status = {
                'timestamp': str(df.index[-1]),
                'regime': str(regime_meta.get('regime', 'UNCERTAIN')),
                
                # Regime Features
                'hurst': float(regime_feats.get('regime_hurst', 0.5)),
                'trend_strength': float(regime_feats.get('regime_trend_score', 0.0)),
                'volatility': float(regime_feats.get('regime_vol_stress', 0.0)),
                'skew': float(regime_feats.get('regime_skew', 0.0)),
                'kurtosis': float(regime_feats.get('regime_tail_risk', 0.0)),
                
                # Hilbert Features
                'cycle_period': float(hilbert_feats.get('hilbert_period', 20.0)),
                'cycle_phase': float(hilbert_feats.get('hilbert_phase', 0.0)),
                'cycle_state': float(hilbert_feats.get('hilbert_state', 0.0))
            }
        except Exception as e:
            logger.error(f"Error computing metrics for {symbol}: {e}")
            
    else:
        logger.warning(f"❌ Not enough data for on-demand calc: {len(candles) if candles else 0} bars")
        
    return status

async def get_market_status_with_fallback(
    db_handler: DBHandler,
    symbol: str, 
    timeframe: str
) -> Dict[str, Any]:
    """
    Fetch market status from DB, fallback to on-demand calculation if missing.
    Aggregates the logic of checking cache and triggering fallback calculation.
    """
    # 1. Try to get cached regime metrics from DB first
    regime_data = await db_handler.get_latest_regime_metrics(symbol, timeframe)
    
    status = {}
    
    # 2. Logic: If found and recent (e.g. within 2x interval), use it.
    if regime_data:
        logger.debug(f"✅ Found cached regime data for {symbol} {timeframe}")
        status = regime_data
        
    # 3. On-Demand Calculation (Fallback)
    if not regime_data or 'cycle_period' not in regime_data:
        logger.info(f"⚠️ Metrics missing or incomplete for {symbol} {timeframe}, calculating on-demand...")
        status = await calculate_market_status_on_demand(symbol, timeframe, db_handler)
    
    return status
