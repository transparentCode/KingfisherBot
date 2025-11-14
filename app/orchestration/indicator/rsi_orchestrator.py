from typing import Dict, Any, Optional
import pandas as pd

from app.models.indicator_context import IndicatorExecutionContext
from app.orchestration.indicator.indicator_orchestrator import BaseIndicatorOrchestrator
from config.asset_indicator_config import ConfigurationManager


class OscillatorOrchestrator(BaseIndicatorOrchestrator):

    def __init__(self, indicator_registry, min_data_points: int = 30, config_manager: ConfigurationManager = None):
        super().__init__(indicator_registry)
        self.min_data_points = min_data_points
        self.config_manager = config_manager

    async def execute(self, context: IndicatorExecutionContext) -> Dict[str, Any]:
        if not self._validate_context(context):
            return {}

        results = {}
        oscillator_indicators = self._get_indicators_by_category("Oscillators")

        for indicator_id, indicator_metadata in oscillator_indicators.items():
            try:
                indicator_class = indicator_metadata.class_ref
                param_schema = indicator_metadata.parameter_schema
                configs = self._get_oscillator_configs(indicator_id, param_schema)

                for config_name, params in configs.items():
                    required_timeframes = getattr(indicator_class, 'required_timeframes', [context.primary_timeframe])

                    for timeframe in required_timeframes:
                        if timeframe not in context.data_cache:
                            continue

                        df = context.data_cache[timeframe]
                        if df.empty or len(df) < self.min_data_points:
                            continue

                        indicator_instance = self._create_indicator_instance(indicator_id, **params)
                        if not indicator_instance:
                            continue

                        # Calculate basic indicator
                        result_df = indicator_instance.calculate(df)

                        # Apply divergence detection if requested
                        if params.get('detect_divergence', False):
                            try:
                                result_df = indicator_instance.detect_divergence(result_df, **params)
                            except Exception as e:
                                self.logger.error(f"Failed to detect divergence for {indicator_id}: {e}")

                        if result_df is not None and not result_df.empty:
                            result_key = f"{indicator_id}_{config_name}_{timeframe}"

                            # Build metadata
                            metadata = {
                                'indicator_id': indicator_id,
                                'config': config_name,
                                'params': params,
                                'timeframe': timeframe,
                                'category': 'Oscillators',
                                'latest_reading': self._get_latest_oscillator_reading(result_df, params),
                                'overbought_oversold': self._check_overbought_oversold(result_df, params),
                                'calculation_time': pd.Timestamp.now()
                            }

                            # Add current bar divergence detection if requested
                            if params.get('detect_divergence', False):
                                current_divergence = self._check_current_bar_divergence(result_df)
                                metadata.update({
                                    'has_divergence_analysis': True,
                                    'current_divergence': current_divergence
                                })

                            results[result_key] = {
                                'data': result_df,
                                'metadata': metadata
                            }

            except Exception as e:
                self.logger.error(f"Error calculating oscillator {indicator_id}: {e}")
                continue

        return results

    def _get_oscillator_configs(self, indicator_id: str, param_schema: dict) -> Dict[str, Dict]:
        """Define different parameter configurations for oscillators"""

        if indicator_id == 'RSI':
            return {
                'default': {'length': 14, 'source': 'close'},
                'short': {'length': 9, 'source': 'close'},
                'long': {'length': 21, 'source': 'close'},
                # Add divergence detection configuration
                'divergence_detection': {
                    'length': 14,
                    'source': 'close',
                    'detect_divergence': True,
                    'lookback_left': 5,
                    'lookback_right': 5,
                    'range_lower': 5,
                    'range_upper': 60
                }
            }

        return {'default': {}}

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

    def _get_latest_oscillator_reading(self, df: pd.DataFrame, params: Dict[str, Any] = None) -> Optional[float]:
        """Get latest oscillator value using dynamic column names"""

        if params:
            length = params.get('length', 14)
            source = params.get('source', 'close')
            expected_column = f"rsi_{length}_{source}"

            if expected_column in df.columns:
                try:
                    latest_value = df[expected_column].iloc[-1]
                    if pd.notna(latest_value):
                        return float(latest_value)
                except (IndexError, ValueError):
                    pass

        return None

    def _check_overbought_oversold(self, df: pd.DataFrame, params: Dict[str, Any] = None) -> Dict[str, bool]:
        """Check if oscillator is in overbought/oversold territory"""

        overbought_level = params.get('overbought', 70) if params else 70
        oversold_level = params.get('oversold', 30) if params else 30

        latest_value = self._get_latest_oscillator_reading(df, params)

        if latest_value is None:
            return {'overbought': False, 'oversold': False, 'neutral': True}

        return {
            'overbought': latest_value > overbought_level,
            'oversold': latest_value < oversold_level,
            'neutral': oversold_level <= latest_value <= overbought_level,
            'reading': latest_value
        }
