from typing import Dict, Any, Optional

import pandas as pd

from app.models.indicator_context import IndicatorExecutionContext
from app.orchestration.indicator.indicator_orchestrator import BaseIndicatorOrchestrator


class MovingAverageOrchestrator(BaseIndicatorOrchestrator):

    def __init__(self, indicator_registry, min_data_points: int = 20):
        super().__init__(indicator_registry)
        self.min_data_points = min_data_points

    def _get_indicator_configs(self, indicator_id: str, param_schema: dict) -> Dict[str, Dict]:
        """Define different parameter configurations for indicators"""
        # Extract parameter constraints from schema
        params = param_schema.get('parameters', {})
        period_config = params.get('period', {})

        if indicator_id in ['SMA', 'EMA']:
            # Use schema constraints if available
            min_period = period_config.get('min', 5)
            max_period = period_config.get('max', 100)

            return {
                'fast': {'period': max(min_period, 14)},
                'medium': {'period': max(min_period, 21)},
                'slow': {'period': min(max_period, 50)}
            }

        return {'default': {}}

    def _get_latest_ma_value(self, df: pd.DataFrame, params: Dict[str, Any] = None) -> Optional[float]:
        """Get latest moving average value using dynamic column names"""

        # Try to find columns with expected pattern first
        if params:
            period = params.get('period', 20)
            source = params.get('source', 'close')
            expected_column = f"sma_{period}_{source}"

            if expected_column in df.columns:
                try:
                    latest_value = df[expected_column].iloc[-1]
                    if pd.notna(latest_value):
                        return float(latest_value)
                except (IndexError, ValueError):
                    pass

        # Fallback to pattern matching for MA columns
        ma_patterns = ['sma_', 'ema_', 'ma_']
        for pattern in ma_patterns:
            matching_cols = [col for col in df.columns if col.startswith(pattern)]
            for col in matching_cols:
                try:
                    latest_value = df[col].iloc[-1]
                    if pd.notna(latest_value):
                        return float(latest_value)
                except (IndexError, ValueError):
                    continue

        return None

    async def execute(self, context: IndicatorExecutionContext) -> Dict[str, Any]:
        if not self._validate_context(context):
            return {}

        results = {}
        ma_indicators = self._get_indicators_by_category("Moving Averages")

        for indicator_id, indicator_metadata in ma_indicators.items():
            try:
                indicator_class = indicator_metadata.class_ref

                # Use metadata parameter schema instead of creating temp instance
                param_schema = indicator_metadata.parameter_schema
                configs = self._get_indicator_configs(indicator_id, param_schema)

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

                        result_df = indicator_instance.calculate(df)

                        if result_df is not None and not result_df.empty:
                            result_key = f"{indicator_id}_{config_name}_{timeframe}"
                            results[result_key] = {
                                'data': result_df,
                                'metadata': {
                                    'indicator_id': indicator_id,
                                    'config': config_name,
                                    'params': params,
                                    'timeframe': timeframe,
                                    'category': 'Moving Averages',
                                    'latest_value': self._get_latest_ma_value(result_df, params),
                                    'calculation_time': pd.Timestamp.now()
                                }
                            }

            except Exception as e:
                self.logger.error(f"Error calculating MA {indicator_id}: {e}")
                continue

        return results
