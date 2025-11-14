from typing import Dict, Any, Optional

import pandas as pd

from app.models.indicator_context import IndicatorExecutionContext
from app.orchestration.indicator.indicator_orchestrator import BaseIndicatorOrchestrator
from config.asset_indicator_config import ConfigurationManager


class TrendAnalysisOrchestrator(BaseIndicatorOrchestrator):
    def __init__(self, indicator_registry, min_data_points: int = 20, config_manager: ConfigurationManager = None):
        super().__init__(indicator_registry)
        self.min_data_points = min_data_points
        self.config_manager = config_manager

    async def execute(self, context: IndicatorExecutionContext) -> Dict[str, Any]:
        results = {}
        # Look for both categories to handle your TrendLineWithBreaks
        trend_indicators = {
            **self._get_indicators_by_category("Trendlines")
        }

        for timeframe, df in context.data_cache.items():
            if df.empty or len(df) < self.min_data_points:
                continue

            for indicator_id, indicator_data in trend_indicators.items():
                try:
                    # Use attribute access instead of dictionary syntax
                    indicator_class = indicator_data.class_ref
                    required_timeframes = getattr(indicator_class, 'required_timeframes', [context.primary_timeframe])

                    for timeframe in required_timeframes:
                        if timeframe not in context.data_cache:
                            self.logger.warning(f"Missing data for {timeframe} - skipping {indicator_id}")
                            continue

                        df = context.data_cache[timeframe]
                        if df.empty or len(df) < 20:
                            self.logger.warning(f"Insufficient data for {indicator_id} on {timeframe}")
                            continue

                        indicator_instance = indicator_class()
                        result_df = indicator_instance.calculate(df)

                        if result_df is not None and not result_df.empty:
                            result_key = f"{indicator_id}_{timeframe}"
                            results[result_key] = {
                                'data': result_df,
                                'metadata': {
                                    'indicator_id': indicator_id,
                                    'timeframe': timeframe,
                                    'category': getattr(indicator_instance, 'category', 'Trend Analysis'),
                                    'latest_signal': self._extract_latest_signal(result_df),
                                    'has_new_signals': self._check_new_signals(result_df),
                                    'calculation_time': pd.Timestamp.now()
                                }
                            }

                            self.logger.debug(f"Calculated {indicator_id} for {timeframe}")

                except Exception as e:
                    self.logger.error(f"Error calculating {indicator_id}: {e}")
                    continue

        return results

    def _extract_latest_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract latest signal from trend indicator result"""
        latest_signal = {}

        # TrendLineWithBreaks specific columns
        signal_columns = [
            'upbreak_signal', 'downbreak_signal', 'upos', 'dnos',
            'trend_direction', 'signal'
        ]

        for col in signal_columns:
            if col in df.columns:
                latest_value = df[col].iloc[-1]
                if not pd.isna(latest_value):
                    latest_signal[col] = latest_value

        return latest_signal

    def _check_new_signals(self, df: pd.DataFrame) -> bool:
        """Check if current bar generated new breakout signals"""
        if 'upbreak_signal' in df.columns and 'downbreak_signal' in df.columns:
            latest_upbreak = df['upbreak_signal'].iloc[-1] == 1
            latest_downbreak = df['downbreak_signal'].iloc[-1] == 1
            return latest_upbreak or latest_downbreak
        return False
