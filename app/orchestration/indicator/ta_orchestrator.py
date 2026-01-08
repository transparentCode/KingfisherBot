from typing import Dict, Any, Optional
import pandas as pd

from app.models.indicator_context import IndicatorExecutionContext
from app.orchestration.indicator.indicator_orchestrator import BaseIndicatorOrchestrator
from config.asset_indicator_config import ConfigurationManager

class TechnicalAnalysisOrchestrator(BaseIndicatorOrchestrator):
    """
    Unified orchestrator for standard technical analysis indicators 
    (Moving Averages, Oscillators, Volatility, etc.)
    """

    def __init__(self, indicator_registry, min_data_points: int = 30, config_manager: Optional[ConfigurationManager] = None):
        super().__init__(indicator_registry, config_manager=config_manager)
        self.min_data_points = min_data_points

    async def execute(self, context: IndicatorExecutionContext) -> Dict[str, Any]:
        if not self._validate_context(context):
            return {}

        results = {}

        # 1. Process Moving Averages (Fast, usually no caching needed)
        ma_results = await self._process_moving_averages(context)
        results.update(ma_results)

        # 2. Process Oscillators (RSI, etc. - might need caching for divergence)
        osc_results = await self._process_oscillators(context)
        results.update(osc_results)

        # 3. Future: Process Volatility (BB, ATR)
        # vol_results = await self._process_volatility(context)
        # results.update(vol_results)

        return results

    # ==========================================
    # SECTION: Moving Averages
    # ==========================================
    async def _process_moving_averages(self, context: IndicatorExecutionContext) -> Dict[str, Any]:
        results = {}
        indicators = self._get_indicators_by_category("Moving Averages")
        
        for indicator_id, metadata in indicators.items():
            # Get configs (Asset specific -> Default)
            configs = self._get_ma_configs(context, indicator_id, metadata.parameter_schema)
            
            for config_name, params in configs.items():
                # Iterate available timeframes
                for timeframe, df in context.data_cache.items():
                    if len(df) < self.min_data_points: continue

                    # Create & Calculate
                    instance = self._create_indicator_instance(indicator_id, **params)
                    if not instance: continue
                    
                    # Use a copy to prevent polluting the cached dataframe with indicator columns
                    result_df = instance.calculate(df.copy())
                    
                    if result_df is not None and not result_df.empty:
                        key = f"{indicator_id}_{config_name}_{timeframe}"
                        results[key] = {
                            'data': result_df,
                            'metadata': {
                                'type': 'MA',
                                'category': 'Moving Averages',
                                'indicator_id': indicator_id,
                                'timeframe': timeframe,
                                'latest': self._get_latest_value(result_df, params.get('col_name')),
                                'params': params
                            }
                        }
        return results

    # ==========================================
    # SECTION: Oscillators
    # ==========================================
    async def _process_oscillators(self, context: IndicatorExecutionContext) -> Dict[str, Any]:
        results = {}
        indicators = self._get_indicators_by_category("Oscillators")
        self.logger.info(f"Processing Oscillators: Found {len(indicators)} indicators: {list(indicators.keys())}")

        for indicator_id, metadata in indicators.items():
            configs = self._get_oscillator_configs(context, indicator_id, metadata.parameter_schema)
            self.logger.info(f"Configs for {indicator_id}: {list(configs.keys())}")

            for config_name, params in configs.items():
                for timeframe, df in context.data_cache.items():
                    if len(df) < self.min_data_points: continue

                    # Check Cache for expensive divergence logic
                    cache_key = f"{indicator_id}_{config_name}_{timeframe}"
                    cached = await self._get_cached_results(context.asset, indicator_id, timeframe)
                    if cached:
                        results[cache_key] = cached
                        continue

                    # Create & Calculate
                    instance = self._create_indicator_instance(indicator_id, **params)
                    if not instance: 
                        self.logger.warning(f"Failed to create instance for {indicator_id}")
                        continue

                    # Use a copy to prevent polluting the cached dataframe
                    try:
                        result_df = instance.calculate(df.copy())
                    except Exception as e:
                        self.logger.error(f"Error calculating {indicator_id}: {e}")
                        continue

                    # Divergence Logic
                    if params.get('detect_divergence'):
                        try:
                            result_df = instance.detect_divergence(result_df, **params)
                        except Exception as e:
                            self.logger.error(f"Divergence error {indicator_id}: {e}")

                    if result_df is not None and not result_df.empty:
                        metadata = {
                            'type': 'Oscillator',
                            'category': 'Oscillators',
                            'indicator_id': indicator_id,
                            'timeframe': timeframe,
                            'latest': self._get_latest_value(result_df, params.get('col_name')),
                            'overbought_oversold': self._check_overbought_oversold(result_df, params)
                        }
                        
                        # Add divergence metadata if applicable
                        if params.get('detect_divergence'):
                            div_meta = self._check_current_bar_divergence(result_df)
                            metadata['divergence'] = div_meta
                            if div_meta.get('has_new_divergence'):
                                self.logger.info(f"Divergence detected for {indicator_id} on {timeframe}: {div_meta}")

                        result_entry = {'data': result_df, 'metadata': metadata}
                        results[cache_key] = result_entry

                        # Cache if expensive
                        if params.get('detect_divergence'):
                            await self._cache_results(context.asset, indicator_id, timeframe, result_entry)
                    else:
                        self.logger.warning(f"Result DF empty for {indicator_id} on {timeframe}")

        return results

    # ==========================================
    # SECTION: Helpers (Shared across types)
    # ==========================================
    def _get_latest_value(self, df: pd.DataFrame, col_name: str = None) -> Optional[float]:
        """Generic helper to get the last value of the main column"""
        if df.empty: return None
        try:
            # If col_name not provided, try to guess or take last column
            if col_name and col_name in df.columns:
                return float(df[col_name].iloc[-1])
            return float(df.iloc[-1, -1]) # Fallback to last column
        except:
            return None

    def _check_overbought_oversold(self, df: pd.DataFrame, params: Dict[str, Any] = None) -> Dict[str, bool]:
        """Check if oscillator is in overbought/oversold territory"""

        overbought_level = params.get('overbought', 70) if params else 70
        oversold_level = params.get('oversold', 30) if params else 30

        # Use generic helper instead of missing specific one
        col_name = params.get('col_name') if params else None
        latest_value = self._get_latest_value(df, col_name)

        if latest_value is None:
            return {'overbought': False, 'oversold': False, 'neutral': True}

        return {
            'overbought': latest_value > overbought_level,
            'oversold': latest_value < oversold_level,
            'neutral': oversold_level <= latest_value <= overbought_level,
            'reading': latest_value
        }

    def _check_current_bar_divergence(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check if the current (latest) bar has a new divergence signal"""

        if df.empty:
            return {'has_new_divergence': False}

        # Check if divergence columns exist
        if 'bullish_divergence' not in df.columns or 'bearish_divergence' not in df.columns:
            return {'has_new_divergence': False}

        # Get the last row (current bar)
        current_bar = df.iloc[-1]

        # Check if current bar has new divergence signals
        has_bullish = bool(current_bar.get('bullish_divergence', False))
        has_bearish = bool(current_bar.get('bearish_divergence', False))
        divergence_signal = current_bar.get('divergence_signal', 0)

        # Only return if there's an actual new divergence on current bar
        if has_bullish or has_bearish:
            return {
                'has_new_divergence': True,
                'divergence_type': 'bullish' if has_bullish else 'bearish',
                'signal_strength': abs(divergence_signal) if divergence_signal != 0 else 1,
                'timestamp': current_bar.name,
                'is_bullish': has_bullish,
                'is_bearish': has_bearish
            }

        return {'has_new_divergence': False}

    def _get_ma_configs(self, context: IndicatorExecutionContext, indicator_id: str, param_schema: dict) -> Dict[str, Dict]:
        """Get indicator configs from asset configuration if available, otherwise use defaults"""
        
        # 1. Try to get from ConfigManager first
        if self.config_manager:
            try:
                asset_config = self.config_manager.get_effective_asset_config(context.asset)
                if asset_config and hasattr(asset_config, 'ma_configs') and asset_config.ma_configs:
                    return asset_config.ma_configs
            except Exception as e:
                self.logger.warning(f"Failed to get asset config for {context.asset}: {e}")

        # 2. Fallback: Use default logic based on parameter schema
        params = param_schema.get('parameters', {})
        period_config = params.get('period', {})

        if indicator_id in ['SMA', 'EMA']:
            min_period = period_config.get('min', 5)
            max_period = period_config.get('max', 100)

            return {
                'fast': {'period': max(min_period, 9), 'source': 'close', 'col_name': f"sma_{max(min_period, 9)}_close" if indicator_id == 'SMA' else f"ema_{max(min_period, 9)}_close"},
                'medium': {'period': max(min_period, 21), 'source': 'close', 'col_name': f"sma_{max(min_period, 21)}_close" if indicator_id == 'SMA' else f"ema_{max(min_period, 21)}_close"},
                'slow': {'period': min(max_period, 50), 'source': 'close', 'col_name': f"sma_{min(max_period, 50)}_close" if indicator_id == 'SMA' else f"ema_{min(max_period, 50)}_close"},
                'trend': {'period': min(max_period, 200), 'source': 'close', 'col_name': f"sma_{min(max_period, 200)}_close" if indicator_id == 'SMA' else f"ema_{min(max_period, 200)}_close"}
            }

        return {'default': {'period': 20, 'source': 'close', 'col_name': 'sma_20_close' if indicator_id == 'SMA' else 'ema_20_close'}}

    def _get_oscillator_configs(self, context: IndicatorExecutionContext, indicator_id: str, param_schema: dict) -> Dict[str, Dict]:
        """Define different parameter configurations for oscillators"""

        if indicator_id == 'RSI':
            return {
                'default': {'length': 14, 'source': 'close', 'col_name': 'rsi_14_close'},
                'short': {'length': 9, 'source': 'close', 'col_name': 'rsi_9_close'},
                'long': {'length': 21, 'source': 'close', 'col_name': 'rsi_21_close'},
                # Add divergence detection configuration
                'divergence_detection': {
                    'length': 14,
                    'source': 'close',
                    'col_name': 'rsi_14_close',
                    'detect_divergence': True,
                    'lookback_left': 5,
                    'lookback_right': 5,
                    'range_lower': 5,
                    'range_upper': 60
                }
            }

        return {'default': {}}