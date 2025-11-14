from app.orchestration.processor.telegram_notification_processor import TelegramNotificationProcessor
from app.telegram.telegram_client import TelegramClient
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.special import softmax
from config.asset_indicator_config import ConfigurationManager  # Assuming this is your config class

class SignalAggregationProcessor:
    def __init__(self, config_manager: ConfigurationManager = None):
        self.logger = logging.getLogger("app")
        self.aggregated_signals = {}
        self.config_manager = config_manager
        
        # Configuration settings with defaults
        self.confluence_line_method = self._get_config_value('confluence_line_method', 'mean')
        self.slope_calculation_method = self._get_config_value('slope_calculation_method', 'linear_regression')
        self.htf_priority_weights = self._get_htf_weights()
        self.smoothing_factor = self._get_config_value('smoothing_factor', 0.7)  # For temporal smoothing
        
        # Track previous conviction for smoothing
        self.previous_conviction = {}

        self.telegram_processor = None
        self._initialize_telegram_processor()

    def _initialize_telegram_processor(self):
        """Initialize Telegram processor if enabled"""
        try:
            # Get telegram configuration
            telegram_config = self._get_telegram_config()
            
            if not telegram_config.get('enabled', False):
                self.logger.info("Telegram notifications disabled in configuration")
                return

            # Try to create Telegram client
            telegram_client = TelegramClient()
            
            if telegram_client and hasattr(telegram_client, 'send_message'):
                self.telegram_processor = TelegramNotificationProcessor(
                    config=telegram_config,
                    telegram_client=telegram_client,
                    telegram_enabled=True
                )
                self.logger.info("Telegram processor initialized successfully")
            else:
                self.logger.warning("Failed to initialize Telegram client")
                
        except Exception as e:
            self.logger.error(f"Error initializing Telegram processor: {e}")
            self.telegram_processor = None
    
    def _get_telegram_config(self) -> Dict:
        """Get Telegram configuration from config manager"""
        if self.config_manager and hasattr(self.config_manager, 'global_config'):
            telegram_config = self.config_manager.global_config.get('telegram_notifications', {})
            
            # Add required fields for compatibility
            telegram_config.update({
                'timeframes_to_monitor': ['15m', '30m', '1h', '4h'],
                'save_charts': True,
                'charts_dir': './charts'
            })
            return telegram_config
        
        # Fallback configuration
        return {
            'enabled': False,
            'timeframes_to_monitor': ['15m', '30m'],
            'save_charts': False,
            'charts_dir': './charts'
        }

    async def initialize(self):
        """Initialize Redis and Telegram processor"""
        await self.redis_handler.initialize()
        
        if self.telegram_processor:
            await self.telegram_processor.initialize()
            self.logger.info("SignalAggregationProcessor initialized with Redis and Telegram")
        else:
            self.logger.info("SignalAggregationProcessor initialized with Redis only")
    
    def _get_config_value(self, key: str, default: Any) -> Any:
        if self.config_manager and hasattr(self.config_manager, 'global_config'):
            return self.config_manager.global_config.get('signal_aggregation', {}).get(key, default)
        return default
    
    def _get_htf_weights(self) -> Dict[str, int]:
        default_weights = {
            '1d': 5, '4h': 4, '1h': 3, '30m': 2, '15m': 1, '5m': 1
        }
        if self.config_manager and hasattr(self.config_manager, 'global_config'):
            custom_weights = self.config_manager.global_config.get('signal_aggregation', {}).get('htf_weights', {})
            default_weights.update(custom_weights)
        return default_weights
    
    async def process_results(self, asset: str, all_results: Dict[str, Any]):
        try:
            self.logger.info(f"Starting enhanced MTF signal aggregation for {asset}")
            
            # Validate input
            if not all_results:
                self.logger.warning("No results provided for aggregation")
                self.aggregated_signals[asset] = []
                return
            
            # Log MTF data summary
            timeframe_summary = {}
            for result_key in all_results.keys():
                parts = result_key.split('_')
                if parts:
                    timeframe = parts[-1]
                    timeframe_summary[timeframe] = timeframe_summary.get(timeframe, 0) + 1
            
            # chnage logging to debug
            self.logger.info(f"MTF Results Summary - {asset}: {timeframe_summary}")
            self.logger.info(f"Total result keys: {list(all_results.keys())}")
            
            # Analyze MTF EMA direction agreement
            mtf_analysis = self._analyze_ema_direction_agreement(all_results)
            
            # Collect all signal types with enhanced confluence
            signals = {
                'trendline_signals': self._extract_trendline_signals(all_results),
                'ma_signals': self._extract_ma_signals(all_results),
                'confluence_signals': self._analyze_enhanced_ma_confluence(all_results),
                'price_action': self._analyze_price_action(all_results),
                'mtf_analysis': mtf_analysis
            }
            self.logger.info(f"MTF Direction Analysis: {mtf_analysis}")
            
            # Create composite signals with MTF enhancements
            composite_signals = self._create_enhanced_composite_signals(asset, signals)
            
            # Store for notification processor
            self.aggregated_signals[asset] = composite_signals
            
            if composite_signals:
                self.logger.info(f"Enhanced composite signals for {asset}: {len(composite_signals)} signals detected")
                for signal in composite_signals:
                    self.logger.info(f"Signal: {signal['direction']} on {signal['timeframe']} "
                                     f"(score: {signal['composite_score']}, conviction: {signal.get('conviction_level', 'normal')})")
            
        except Exception as e:
            self.logger.error(f"Error in enhanced signal aggregation for {asset}: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            self.aggregated_signals[asset] = []
    
    def _analyze_ema_direction_agreement(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze EMA direction agreement across timeframes"""
        timeframe_directions = {}
        
        # Group by timeframes (ordered by priority)
        ordered_timeframes = ['1d', '4h', '1h', '30m', '15m', '5m']
        
        for timeframe in ordered_timeframes:
            tf_results = {k: v for k, v in all_results.items() if k.endswith(f'_{timeframe}')}
            
            if tf_results:
                direction_data = self._calculate_tf_ema_direction(tf_results, timeframe)
                if direction_data['slope_count'] > 0:
                    timeframe_directions[timeframe] = direction_data
        
        # Calculate agreement strength
        agreement = self._evaluate_direction_agreement(timeframe_directions)
        
        return {
            'timeframe_directions': timeframe_directions,
            'agreement': agreement,
            'conviction_multiplier': self._get_conviction_multiplier(agreement)
        }
    
    def _calculate_tf_ema_direction(self, tf_results: Dict, timeframe: str) -> Dict:
        """Calculate EMA direction for a specific timeframe using configurable method"""
        ema_slopes = []
        
        for result_key, result_data in tf_results.items():
            if 'EMA' in result_key and 'data' in result_data:
                df = result_data['data']
                if len(df) >= 5:  # Need enough points for regression
                    ema_cols = [col for col in df.columns if 'ema' in col.lower()]
                    for ema_col in ema_cols:
                        slope = self._calculate_slope(df[ema_col])
                        if slope is not None:
                            ema_slopes.append(slope)
        
        if ema_slopes:
            avg_slope = np.mean(ema_slopes)
            slope_std = np.std(ema_slopes) if len(ema_slopes) > 1 else 0
            
            # Determine direction with confidence
            if avg_slope > slope_std * 0.1:  # Threshold based on std deviation
                direction = 'bullish'
            elif avg_slope < -slope_std * 0.1:
                direction = 'bearish'
            else:
                direction = 'neutral'
            
            # Calculate strength (normalized)
            max_abs_slope = max(abs(s) for s in ema_slopes) if ema_slopes else 0
            strength = min(abs(avg_slope) / max_abs_slope, 1.0) if max_abs_slope > 0 else 0
            
            return {
                'direction': direction,
                'strength': strength,
                'slope_count': len(ema_slopes),
                'avg_slope': avg_slope,
                'timeframe_weight': self.htf_priority_weights.get(timeframe, 1)
            }
        
        return {'direction': 'neutral', 'strength': 0, 'slope_count': 0, 'timeframe_weight': 1}
    
    def _calculate_slope(self, series: pd.Series) -> Optional[float]:
        """Calculate slope using configurable method with error handling"""
        try:
            series_clean = series.dropna()
            if len(series_clean) < 3:
                return 0.0  # Neutral slope
            
            if self.slope_calculation_method == 'linear_regression':
                X = np.arange(len(series_clean)).reshape(-1, 1)
                y = series_clean.values
                reg = LinearRegression()
                reg.fit(X, y)
                return float(reg.coef_[0])
                
            elif self.slope_calculation_method == 'simple':
                recent_values = series_clean.tail(3).values
                return float((recent_values[-1] - recent_values[0]) / 2)
                
            else:
                recent_values = series_clean.tail(3).values
                return float((recent_values[-1] - recent_values[0]) / 2)
                
        except Exception as e:
            self.logger.error(f"Error calculating slope: {e}")
            return 0.0  # Neutral fallback
    
    def _evaluate_direction_agreement(self, timeframe_directions: Dict) -> Dict:
        """Evaluate how well timeframes agree on direction with improved scoring"""
        if not timeframe_directions:
            return {'same_direction_count': 0, 'agreeing_timeframes': [], 'conviction_score': 0}
        
        # Count directions with weights
        direction_scores = {'bullish': 0, 'bearish': 0, 'neutral': 0}
        
        for tf, data in timeframe_directions.items():
            direction = data['direction']
            weight = data['timeframe_weight']
            strength = data['strength']
            direction_scores[direction] += weight * strength
        
        # Find dominant direction
        dominant_direction = max(direction_scores, key=direction_scores.get)
        agreeing_timeframes = [tf for tf, data in timeframe_directions.items() 
                             if data['direction'] == dominant_direction]
        
        # Calculate conviction score
        conviction_score = sum(timeframe_directions[tf]['timeframe_weight'] * timeframe_directions[tf]['strength'] for tf in agreeing_timeframes)
        
        return {
            'dominant_direction': dominant_direction,
            'same_direction_count': len(agreeing_timeframes),
            'agreeing_timeframes': agreeing_timeframes,
            'conviction_score': min(conviction_score, 10),  # Cap at 10
            'direction_scores': direction_scores
        }
    
    def _get_conviction_multiplier(self, agreement: Dict) -> float:
        """Get conviction multiplier based on MTF agreement with smoothing"""
        agreeing_count = agreement['same_direction_count']
        conviction_score = agreement['conviction_score']
        
        raw_multiplier = 1.0
        if agreeing_count >= 3 and conviction_score >= 6:
            raw_multiplier = 1.5
        elif agreeing_count >= 2 and conviction_score >= 4:
            raw_multiplier = 1.2
        elif agreeing_count >= 2:
            raw_multiplier = 1.1
        
        # Smooth with previous (reduce flipping)
        prev_multiplier = self.previous_conviction.get('multiplier', 1.0)
        smoothed = self.smoothing_factor * raw_multiplier + (1 - self.smoothing_factor) * prev_multiplier
        self.previous_conviction['multiplier'] = smoothed
        
        return smoothed
    
    def _calculate_confluence_line(self, ma_values: Dict[str, float]) -> Dict:
        """Calculate explicit confluence line from EMA values with validation"""
        # Validate and parse EMA values
        ema_data = {}
        for indicator_id, value in ma_values.items():
            if 'ema' in indicator_id.lower() and pd.notna(value) and value > 0:
                period = self._extract_ema_period(indicator_id)
                if period:
                    ema_data[period] = float(value)
        
        if len(ema_data) < 2:
            return {'line': None, 'method': self.confluence_line_method, 'ema_count': len(ema_data)}
        
        values = np.array(list(ema_data.values()))
        
        if self.confluence_line_method == 'weighted':
            weights = np.array(list(ema_data.keys()))
            total_weight = np.sum(weights)
            confluence_line = np.sum(weights * values) / total_weight if total_weight > 0 else np.mean(values)
            
        elif self.confluence_line_method == 'median':
            confluence_line = np.median(values)
            
        else:  # 'mean' (default)
            confluence_line = np.mean(values)
        
        return {
            'line': confluence_line,
            'method': self.confluence_line_method,
            'ema_count': len(ema_data),
            'ema_data': ema_data,
            'ema_periods': list(ema_data.keys())
        }
    
    def _extract_ema_period(self, indicator_id: str) -> Optional[int]:
        """Extract EMA period from indicator ID or config with fallback"""
        try:
            if 'fast' in indicator_id.lower():
                return 14
            elif 'medium' in indicator_id.lower():
                return 21
            elif 'slow' in indicator_id.lower():
                return 50
            else:
                import re
                numbers = re.findall(r'\d+', indicator_id)
                if numbers:
                    return int(numbers[0])
            return None
        except:
            return None
    
    def _analyze_enhanced_ma_confluence(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced MA confluence analysis with explicit confluence line"""
        confluence_by_tf = {}
        timeframe_data = self._group_results_by_timeframe(results)
        
        for timeframe, tf_results in timeframe_data.items():
            current_price = self._get_current_price(tf_results)
            if not current_price:
                continue
            
            enhanced_confluence = self._check_enhanced_ma_confluence(tf_results, current_price)
            if enhanced_confluence['signal'] or enhanced_confluence.get('confluence_line'):
                confluence_by_tf[timeframe] = enhanced_confluence
        
        return confluence_by_tf
    
    def _check_enhanced_ma_confluence(self, tf_results: Dict, current_price: float) -> Dict:
        """Enhanced MA confluence check with explicit confluence line"""
        ma_values = {}
        for result_key, result_data in tf_results.items():
            try:
                metadata = result_data.get('metadata', {})
                if not isinstance(metadata, dict):
                    continue
                
                category = metadata.get('category')
                if category != 'Moving Averages':
                    continue
                
                indicator_id = metadata.get('indicator_id')
                latest_value = metadata.get('latest_value')
                
                if latest_value is None and 'data' in result_data:
                    df = result_data['data']
                    latest_value = self._get_latest_ma_value_from_df(df)
                
                if latest_value is not None:
                    ma_values[indicator_id] = float(latest_value)
            except Exception as e:
                self.logger.error(f"Error processing enhanced MA confluence for {result_key}: {e}")
                continue
        
        if not ma_values:
            return {'signal': False, 'confluence_line': None}
        
        # Calculate confluence line
        confluence_line_data = self._calculate_confluence_line(ma_values)
        
        # Evaluate traditional confluence
        traditional_confluence = self._evaluate_confluence_conditions(current_price, ma_values)
        
        # Enhanced result
        enhanced_result = traditional_confluence.copy()
        enhanced_result['confluence_line'] = confluence_line_data
        
        # Add confluence line analysis if line exists
        if confluence_line_data['line']:
            line_value = confluence_line_data['line']
            distance_pct = ((current_price - line_value) / line_value) * 100 if line_value != 0 else 0
            
            enhanced_result['line_analysis'] = {
                'distance_pct': round(distance_pct, 2),
                'position': 'above' if distance_pct > 0 else 'below',
                'significant': abs(distance_pct) > 0.5
            }
            
            self.logger.debug(f"Confluence line: {line_value:.2f}, Price: {current_price:.2f}, Distance: {distance_pct:.2f}%")
        
        return enhanced_result
    
    def _create_enhanced_composite_signals(self, asset: str, signals: Dict) -> List[Dict]:
        """Create enhanced composite signals with MTF analysis"""
        composite_signals = []
        
        # Extract timeframes with validation
        timeframes = set()
        for category_signals in [signals['trendline_signals'], signals['ma_signals'], signals['confluence_signals']]:
            if isinstance(category_signals, dict):
                for sub_signals in category_signals.values():
                    if isinstance(sub_signals, list):
                        timeframes.update(s.get('timeframe') for s in sub_signals if s.get('timeframe'))
            elif isinstance(category_signals, list):
                timeframes.update(s.get('timeframe') for s in category_signals if s.get('timeframe'))
        
        timeframes = {tf for tf in timeframes if tf and tf != 'unknown'}
        
        if not timeframes:
            self.logger.warning(f"No valid timeframes found in signals for {asset}")
            return []
        
        self.logger.info(f"Processing {len(timeframes)} timeframes for {asset}: {timeframes}")
        
        # Get MTF analysis
        mtf_analysis = signals.get('mtf_analysis', {})
        
        for timeframe in timeframes:
            composite_signal = self._evaluate_enhanced_timeframe_confluence(
                asset, timeframe, signals, mtf_analysis
            )
            if composite_signal:
                composite_signals.append(composite_signal)
        
        return composite_signals
    
    def _evaluate_enhanced_timeframe_confluence(self, asset: str, timeframe: str, 
                                                signals: Dict, mtf_analysis: Dict) -> Optional[Dict]:
        """Enhanced confluence evaluation with explicit line and MTF agreement"""
        score = 0
        signal_details = []
        direction = None
        conviction_level = 'normal'
        
        # Check trendline signals
        tl_bullish = [s for s in signals.get('trendline_signals', {}).get('bullish', []) if s['timeframe'] == timeframe]
        tl_bearish = [s for s in signals.get('trendline_signals', {}).get('bearish', []) if s['timeframe'] == timeframe]
        
        if tl_bullish:
            score += 3 * len(tl_bullish)
            signal_details.extend(tl_bullish)
            direction = 'bullish'
        if tl_bearish:
            score += 3 * len(tl_bearish)
            signal_details.extend(tl_bearish)
            direction = 'bearish'
        
        # Enhanced confluence signals analysis
        confluence_signals = signals.get('confluence_signals', {})
        if timeframe in confluence_signals:
            conf = confluence_signals[timeframe]
            
            # Traditional confluence
            if conf['signal']:
                if not direction:
                    direction = conf['direction']
                elif direction != conf['direction']:
                    # Conflict - reduce score
                    score -= 1
                score += 2
                signal_details.append({
                    'type': 'ma_confluence',
                    'direction': conf['direction'],
                    'strength': conf['strength'],
                    'timeframe': timeframe
                })
            
            # Confluence line analysis
            if conf.get('confluence_line') and conf['confluence_line']['line']:
                line_data = conf['confluence_line']
                line_analysis = conf.get('line_analysis', {})
                if line_analysis.get('significant'):
                    position = line_analysis['position']
                    distance_pct = line_analysis['distance_pct']
                    
                    line_direction = 'bullish' if position == 'above' else 'bearish'
                    if not direction:
                        direction = line_direction
                    elif direction != line_direction:
                        score -= 1  # Conflict penalty
                    
                    # Score based on distance
                    if abs(distance_pct) > 1.0:
                        score += 3
                    elif abs(distance_pct) > 0.5:
                        score += 2
                    else:
                        score += 1
                    
                    signal_details.append({
                        'type': 'confluence_line',
                        'position': position,
                        'distance_pct': distance_pct,
                        'line_value': line_data['line'],
                        'method': line_data['method'],
                        'ema_count': line_data['ema_count'],
                        'timeframe': timeframe
                    })
        
        # MTF direction agreement bonus
        if mtf_analysis and 'agreement' in mtf_analysis:
            agreement = mtf_analysis['agreement']
            conviction_multiplier = mtf_analysis['conviction_multiplier']
            
            tf_directions = mtf_analysis.get('timeframe_directions', {})
            if timeframe in tf_directions:
                tf_direction = tf_directions[timeframe]['direction']
                dominant_direction = agreement.get('dominant_direction')
                
                if tf_direction == dominant_direction and tf_direction != 'neutral':
                    if not direction:
                        direction = tf_direction
                    elif direction != tf_direction:
                        score -= 2  # Strong conflict with MTF
                    
                    # Apply MTF agreement bonus with HTF priority
                    agreeing_tfs = agreement['agreeing_timeframes']
                    htf_bonus = sum(self.htf_priority_weights.get(tf, 1) * tf_directions[tf]['strength']
                                  for tf in agreeing_tfs if tf != timeframe)
                    score += min(htf_bonus, 6)
                    
                    signal_details.append({
                        'type': 'mtf_agreement',
                        'dominant_direction': dominant_direction,
                        'agreeing_timeframes': agreeing_tfs,
                        'htf_bonus': htf_bonus,
                        'timeframe': timeframe
                    })
        
        # Check MA signals for this timeframe
        ma_signals = signals.get('ma_signals', {})
        current_price = ma_signals.get('current_price')
        if current_price and direction:
            tf_ma_above = [s for s in ma_signals.get('above_ma', []) if s['timeframe'] == timeframe]
            tf_ma_below = [s for s in ma_signals.get('below_ma', []) if s['timeframe'] == timeframe]
            
            if tf_ma_above and direction == 'bullish':
                score += len(tf_ma_above)
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
        
        # Apply conviction multiplier
        score *= conviction_multiplier
        
        # Dynamic threshold
        min_score = 3 if conviction_level == 'high' else 4 if conviction_level == 'medium' else 5
        
        if score >= min_score and direction:
            return {
                'asset': asset,
                'timeframe': timeframe,
                'direction': direction,
                'composite_score': int(score),
                'conviction_level': conviction_level,
                'signal_details': signal_details,
                'current_price': current_price,
                'mtf_agreement': mtf_analysis.get('agreement', {}),
                'timestamp': pd.Timestamp.now()
            }
        return None
    
    def _extract_trendline_signals(self, results: Dict[str, Any]) -> Dict[str, List]:
        """Extract trendline breakout signals (unchanged from original)"""
        trendline_signals = {'bullish': [], 'bearish': []}
        for result_key, result_data in results.items():
            metadata = result_data.get('metadata', {})
            if not isinstance(metadata, dict):
                continue
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
        """Extract MA signals and price vs MA analysis with enhanced logging"""
        ma_signals = {'above_ma': [], 'below_ma': [], 'current_price': None}
        current_price = None
        processed_count = 0
        
        for result_key, result_data in results.items():
            try:
                metadata = result_data.get('metadata', {})
                if not isinstance(metadata, dict):
                    continue
                
                # Get current price
                if not current_price and 'data' in result_data:
                    df = result_data['data']
                    if not df.empty and 'close' in df.columns:
                        current_price = float(df['close'].iloc[-1])
                        ma_signals['current_price'] = current_price
                        self.logger.debug(f"Current price detected: {current_price}")
                
                category = metadata.get('category')
                timeframe = metadata.get('timeframe', 'unknown')
                indicator_id = metadata.get('indicator_id')
                latest_value = metadata.get('latest_value')
                
                if category == 'Moving Averages' and current_price:
                    ma_value = latest_value
                    if ma_value is None and 'data' in result_data:
                        df = result_data['data']
                        ma_value = self._get_latest_ma_value_from_df(df)
                    
                    if ma_value is not None:
                        signal_data = {
                            'timeframe': timeframe,
                            'indicator': indicator_id,
                            'ma_value': float(ma_value),
                            'price': current_price
                        }
                        if current_price > float(ma_value):
                            ma_signals['above_ma'].append(signal_data)
                        else:
                            ma_signals['below_ma'].append(signal_data)
                        
                        processed_count += 1
                        self.logger.debug(f"Processed MA signal: {indicator_id} on {timeframe} = {ma_value}")
            except Exception as e:
                self.logger.error(f"Error processing MA signals for {result_key}: {e}")
                continue
        
        self.logger.info(f"Processed {processed_count} MA signals. Above MA: {len(ma_signals['above_ma'])}, Below MA: {len(ma_signals['below_ma'])}")
        return ma_signals
    
    def _get_latest_ma_value_from_df(self, df: pd.DataFrame) -> Optional[float]:
        """Extract latest MA value from dataframe when not in metadata"""
        try:
            if df.empty:
                return None
            ma_columns = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume'] and 'ema' in col.lower()]
            if ma_columns:
                latest_value = df[ma_columns[0]].iloc[-1]
                if pd.notna(latest_value):
                    return float(latest_value)
            return None
        except Exception as e:
            self.logger.error(f"Error extracting MA value from dataframe: {e}")
            return None
    
    def _group_results_by_timeframe(self, results: Dict[str, Any]) -> Dict[str, Dict]:
        """Group results by timeframe with validation"""
        timeframe_data = {}
        for result_key, result_data in results.items():
            metadata = result_data.get('metadata', {})
            if isinstance(metadata, dict) and 'timeframe' in metadata:
                timeframe = metadata['timeframe']
                if timeframe not in timeframe_data:
                    timeframe_data[timeframe] = {}
                timeframe_data[timeframe][result_key] = result_data
        return timeframe_data
    
    def _get_current_price(self, tf_results: Dict) -> Optional[float]:
        """Get current price with fallback"""
        for result_data in tf_results.values():
            if 'data' in result_data:
                df = result_data['data']
                if not df.empty and 'close' in df.columns:
                    try:
                        return float(df['close'].iloc[-1])
                    except:
                        continue
        return None
    
    def _evaluate_confluence_conditions(self, current_price: float, ma_values: Dict) -> Dict:
        """Evaluate confluence conditions"""
        if not ma_values:
            return {'signal': False}
        
        bullish_count = sum(1 for ma_val in ma_values.values() if current_price >= ma_val)
        total_mas = len(ma_values)
        confluence_strength = bullish_count / total_mas
        
        if confluence_strength >= 0.8:
            return {'signal': True, 'direction': 'bullish', 'strength': confluence_strength}
        elif confluence_strength <= 0.2:
            return {'signal': True, 'direction': 'bearish', 'strength': 1 - confluence_strength}
        return {'signal': False}
    
    def _analyze_price_action(self, results: Dict[str, Any]) -> Dict:
        """Placeholder for price action analysis (expand as needed)"""
        return {}
    
    def get_aggregated_signals(self, asset: str) -> Optional[List[Dict]]:
        """Allow other processors to access aggregated signals"""
        return self.aggregated_signals.get(asset)
    
    def validate_signals(self, aggregated_signals: List[Dict]) -> Dict[str, Any]:
        """Validate aggregated signals (e.g., from multiple runs)"""
        if not aggregated_signals:
            return {'total_signals': 0, 'distribution': {}, 'avg_score': 0, 'switch_rate': 0}
        
        df = pd.DataFrame(aggregated_signals)
        if 'timestamp' not in df.columns or 'direction' not in df.columns:
            raise ValueError("Aggregated signals must have 'timestamp' and 'direction'")
        
        df = df.sort_values('timestamp')
        
        # Signal distribution
        distribution = df['direction'].value_counts(normalize=True).to_dict()
        
        # Average score per direction
        avg_score = df.groupby('direction')['composite_score'].mean().to_dict()
        
        # Switch rate
        df['switch'] = df['direction'] != df['direction'].shift(1)
        switch_rate = df['switch'].mean()
        
        return {
            'total_signals': len(df),
            'distribution': distribution,
            'avg_score': avg_score,
            'switch_rate': switch_rate
        }
