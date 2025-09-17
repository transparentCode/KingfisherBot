import logging
from typing import Dict, Any, List, Optional
import pandas as pd


class SignalAggregationProcessor:
    def __init__(self, config=None):
        self.config = config
        self.logger = logging.getLogger("app")
        self.aggregated_signals = {}

    async def process_results(self, asset: str, all_results: Dict[str, Any]):
        """Aggregate all signals and make composite decisions"""
        try:
            # Collect all signal types
            signals = {
                'trendline_signals': self._extract_trendline_signals(all_results),
                'ma_signals': self._extract_ma_signals(all_results),
                'confluence_signals': self._analyze_ma_confluence(all_results),
                'price_action': self._analyze_price_action(all_results)
            }

            # Create composite signals
            composite_signals = self._create_composite_signals(asset, signals)

            # Store for notification processor
            self.aggregated_signals[asset] = composite_signals

            if composite_signals:
                self.logger.info(f"Composite signals for {asset}: {len(composite_signals)} signals detected")

        except Exception as e:
            self.logger.error(f"Error in signal aggregation for {asset}: {e}")

    def _extract_trendline_signals(self, results: Dict[str, Any]) -> Dict[str, List]:
        """Extract trendline breakout signals"""
        trendline_signals = {'bullish': [], 'bearish': []}

        for result_key, result_data in results.items():
            metadata = result_data.get('metadata', {})
            if metadata.get('category') in ['Trend Analysis', 'Trendlines']:
                latest_signal = metadata.get('latest_signal', {})
                timeframe = metadata.get('timeframe')

                if latest_signal.get('upbreak_signal') == 1:
                    trendline_signals['bullish'].append({
                        'timeframe': timeframe,
                        'indicator': metadata.get('indicator_id'),
                        'signal_type': 'upbreak',
                        'strength': 1.0
                    })

                if latest_signal.get('downbreak_signal') == 1:
                    trendline_signals['bearish'].append({
                        'timeframe': timeframe,
                        'indicator': metadata.get('indicator_id'),
                        'signal_type': 'downbreak',
                        'strength': 1.0
                    })

        return trendline_signals

    def _extract_ma_signals(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract MA signals and price vs MA analysis"""
        ma_signals = {'above_ma': [], 'below_ma': [], 'current_price': None}

        current_price = None
        for result_key, result_data in results.items():
            metadata = result_data.get('metadata', {})

            # Get current price
            if not current_price and 'data' in result_data:
                df = result_data['data']
                if not df.empty and 'close' in df.columns:
                    current_price = float(df['close'].iloc[-1])
                    ma_signals['current_price'] = current_price

            # Check MA signals
            if metadata.get('category') == 'Moving Averages':
                ma_value = metadata.get('latest_value')
                if ma_value and current_price:
                    signal_data = {
                        'timeframe': metadata.get('timeframe'),
                        'indicator': f"{metadata.get('indicator_id')}_{metadata.get('config')}",
                        'ma_value': ma_value,
                        'price': current_price
                    }

                    if current_price > ma_value:
                        ma_signals['above_ma'].append(signal_data)
                    else:
                        ma_signals['below_ma'].append(signal_data)

        return ma_signals

    def _analyze_ma_confluence(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze MA confluence like your MTF processor"""
        confluence_by_tf = {}

        timeframe_data = self._group_results_by_timeframe(results)

        for timeframe, tf_results in timeframe_data.items():
            current_price = self._get_current_price(tf_results)
            if not current_price:
                continue

            ma_confluence = self._check_ma_confluence(tf_results, current_price)
            if ma_confluence['signal']:
                confluence_by_tf[timeframe] = ma_confluence

        return confluence_by_tf

    def _create_composite_signals(self, asset: str, signals: Dict) -> List[Dict]:
        """Create composite signals combining all indicators"""
        composite_signals = []

        # Group by timeframe for analysis
        timeframes = set()
        if signals['trendline_signals']['bullish']:
            timeframes.update([s['timeframe'] for s in signals['trendline_signals']['bullish']])
        if signals['trendline_signals']['bearish']:
            timeframes.update([s['timeframe'] for s in signals['trendline_signals']['bearish']])

        for timeframe in timeframes:
            # Check for confluence of signals in this timeframe
            composite_signal = self._evaluate_timeframe_confluence(
                asset, timeframe, signals
            )

            if composite_signal:
                composite_signals.append(composite_signal)

        return composite_signals

    def _evaluate_timeframe_confluence(self, asset: str, timeframe: str, signals: Dict) -> Optional[Dict]:
        """Evaluate confluence for specific timeframe"""
        score = 0
        signal_details = []
        direction = None

        # Check trendline signals
        tl_bullish = [s for s in signals['trendline_signals']['bullish'] if s['timeframe'] == timeframe]
        tl_bearish = [s for s in signals['trendline_signals']['bearish'] if s['timeframe'] == timeframe]

        if tl_bullish:
            score += 3  # Trendline breakout is strong signal
            signal_details.extend(tl_bullish)
            direction = 'bullish'

        if tl_bearish:
            score += 3
            signal_details.extend(tl_bearish)
            direction = 'bearish'

        # Check MA signals for this timeframe
        current_price = signals['ma_signals']['current_price']
        if current_price:
            tf_ma_above = [s for s in signals['ma_signals']['above_ma'] if s['timeframe'] == timeframe]
            tf_ma_below = [s for s in signals['ma_signals']['below_ma'] if s['timeframe'] == timeframe]

            if tf_ma_above and direction == 'bullish':
                score += len(tf_ma_above)  # Each MA adds to confluence
                signal_details.append({
                    'type': 'price_above_ma',
                    'count': len(tf_ma_above),
                    'timeframe': timeframe
                })

            if tf_ma_below and direction == 'bearish':
                score += len(tf_ma_below)
                signal_details.append({
                    'type': 'price_below_ma',
                    'count': len(tf_ma_below),
                    'timeframe': timeframe
                })

        # Check confluence signals
        if timeframe in signals['confluence_signals']:
            conf = signals['confluence_signals'][timeframe]
            if conf['direction'] == direction:
                score += 2  # MA confluence adds strength
                signal_details.append({
                    'type': 'ma_confluence',
                    'strength': conf['strength'],
                    'timeframe': timeframe
                })

        # Return composite signal if score is high enough
        if score >= 4:  # Minimum threshold for notification
            return {
                'asset': asset,
                'timeframe': timeframe,
                'direction': direction,
                'composite_score': score,
                'signal_details': signal_details,
                'current_price': current_price,
                'timestamp': pd.Timestamp.now()
            }

        return None

    # Helper methods from your MTF processor
    def _group_results_by_timeframe(self, results: Dict[str, Any]) -> Dict[str, Dict]:
        timeframe_data = {}
        for result_key, result_data in results.items():
            if 'metadata' in result_data and 'timeframe' in result_data['metadata']:
                timeframe = result_data['metadata']['timeframe']
                if timeframe not in timeframe_data:
                    timeframe_data[timeframe] = {}
                timeframe_data[timeframe][result_key] = result_data
        return timeframe_data

    def _get_current_price(self, tf_results: Dict) -> Optional[float]:
        for result_key, result_data in tf_results.items():
            if 'data' in result_data:
                df = result_data['data']
                if not df.empty and 'close' in df.columns:
                    try:
                        return float(df['close'].iloc[-1])
                    except (IndexError, ValueError, TypeError):
                        continue
        return None

    def _check_ma_confluence(self, tf_results: Dict, current_price: float) -> Dict:
        # Your existing confluence logic
        ma_values = {}
        for result_key, result_data in tf_results.items():
            metadata = result_data.get('metadata', {})
            if metadata.get('category') == 'Moving Averages' and metadata.get('latest_value'):
                ma_name = f"{metadata['indicator_id']}_{metadata['config']}"
                ma_values[ma_name] = float(metadata['latest_value'])

        if not ma_values:
            return {'signal': False}

        return self._evaluate_confluence_conditions(current_price, ma_values)

    def _evaluate_confluence_conditions(self, current_price: float, ma_values: Dict) -> Dict:
        bullish_count = sum(1 for ma_val in ma_values.values() if current_price >= ma_val)
        total_mas = len(ma_values)
        confluence_strength = bullish_count / total_mas

        if confluence_strength >= 0.8 or confluence_strength <= 0.2:
            direction = 'bullish' if confluence_strength >= 0.8 else 'bearish'
            return {
                'signal': True,
                'direction': direction,
                'strength': confluence_strength,
                'ma_values': ma_values
            }
        return {'signal': False}

    def _analyze_price_action(self, results: Dict[str, Any]) -> Dict:
        # Add price action analysis if needed
        return {}

    def get_aggregated_signals(self, asset: str) -> Optional[List[Dict]]:
        """Allow other processors to access aggregated signals"""
        return self.aggregated_signals.get(asset)
