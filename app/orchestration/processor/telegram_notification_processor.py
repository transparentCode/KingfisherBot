from datetime import datetime
import logging
import json
import os
from typing import Dict, Any, List, Optional
import pandas as pd

from app.db.redis_handler import RedisHandler
from app.telegram.telegram_client import TelegramClient
from app.orchestration.processor.notification_policy import NotificationPolicy
from app.orchestration.processor.notification_formatter import NotificationFormatter

class TelegramNotificationProcessor:
    """Enhanced processor with Redis-based throttling and configurable per-asset settings"""

    def __init__(
        self, config, telegram_client: TelegramClient, telegram_enabled: bool = True
    ):
        self.config = config
        self.telegram_client = telegram_client
        self.logger = logging.getLogger("app")
        self.telegram_enabled = telegram_enabled and telegram_client is not None
        self.redis_handler = RedisHandler()
        
        # Initialize delegates
        self.policy = NotificationPolicy(config, self.redis_handler)
        self.formatter = NotificationFormatter(config)

    async def initialize(self):
        """Initialize Redis connection"""
        await self.redis_handler.initialize()
        self.logger.info("TelegramNotificationProcessor initialized with Redis")

    async def process_signals(self, asset: str, composite_signals: List[Dict], raw_results: Dict[str, Any]):
        """Entry point called by SignalAggregationProcessor"""
        if not self.telegram_enabled:
            return

        try:
            if not await self.policy.check_global_throttle():
                self.logger.info("Global throttle active - skipping notifications")
                return

            if composite_signals:
                await self._process_composite_signals(asset, composite_signals, raw_results)
                return 

            allow_individual = self.config.get('allow_individual_signals', False)
            if allow_individual:
                await self._process_individual_signals(asset, raw_results)

        except Exception as e:
            self.logger.error(f"Error processing telegram signals for {asset}: {e}", exc_info=True)

    async def _process_composite_signals(self, asset: str, composite_signals: List[Dict], raw_results: Dict[str, Any]):
        for signal in composite_signals:
            try:
                conviction_level = signal.get("conviction_level", "normal")
                composite_score = signal.get("composite_score", 0)
                priority = "high" if (conviction_level == "high" or composite_score >= 8) else \
                           "medium" if (conviction_level == "medium" or composite_score >= 6) else "normal"

                message = self.formatter.build_composite_message(asset, signal)
                
                chart_path = None
                if self.config.get('save_charts', False):
                    timeframe = signal.get('timeframe')
                    df = self._find_dataframe_for_timeframe(raw_results, timeframe)
                    if df is not None:
                        chart_path = await self.formatter.generate_chart(
                            asset, timeframe, df, 
                            f"composite_{signal['direction']}", pd.Timestamp.now()
                        )

                metadata = {
                    "signal_type": "composite",
                    "direction": signal["direction"],
                    "score": composite_score,
                    "conviction": conviction_level,
                    "timeframe": signal["timeframe"],
                    "components": len(signal.get("signal_details", [])),
                }

                await self.send_notification(asset, message, "composite", priority, chart_path, metadata)

            except Exception as e:
                self.logger.error(f"Error processing composite signal for {asset}: {e}")

    async def _process_individual_signals(self, asset: str, results: Dict[str, Any]):
        for result_key, result_data in results.items():
            try:
                metadata = result_data.get("metadata", {})
                timeframe = metadata.get("timeframe")
                if timeframe not in self.config.get('timeframes_to_monitor', []): continue

                category = metadata.get("category", "")
                if category == "Trend Analysis":
                    await self._handle_trendline_signals(asset, timeframe, result_data)
                elif category == "Oscillators":
                    await self._handle_oscillator_signals(asset, timeframe, result_data)
            except Exception as e:
                self.logger.error(f"Error processing individual signal {result_key}: {e}")

    async def _handle_trendline_signals(self, asset: str, timeframe: str, result_data: Dict):
        metadata = result_data.get("metadata", {})
        latest_signal = metadata.get("latest_signal", {})

        if latest_signal.get("upbreak_signal") == 1 or latest_signal.get("downbreak_signal") == 1:
            breakout_type = "upward" if latest_signal.get("upbreak_signal") == 1 else "downward"
            df = result_data["data"]
            latest_bar = df.iloc[-1]
            
            # Get session info for message
            session_info = self.policy.check_time_rules()
            
            message = self.formatter.build_breakout_message(
                asset, timeframe, breakout_type, df.index[-1], latest_bar["close"], session_info
            )

            chart_path = await self.formatter.generate_chart(
                asset, timeframe, df, breakout_type, df.index[-1]
            )

            metadata = {
                "signal_type": "breakout",
                "direction": breakout_type,
                "timeframe": timeframe,
                "price": latest_bar["close"],
            }

            await self.send_notification(asset, message, "breakout", "medium", chart_path, metadata)

    async def _handle_oscillator_signals(self, asset: str, timeframe: str, result_data: Dict):
        metadata = result_data.get("metadata", {})
        if metadata.get("overbought") or metadata.get("oversold"):
            signal_type = "overbought" if metadata.get("overbought") else "oversold"
            rsi_value = metadata.get("latest_reading", "N/A")

            message = self.formatter.build_rsi_message(asset, timeframe, signal_type, rsi_value)
            metadata = {"signal_type": "rsi", "condition": signal_type, "value": rsi_value, "timeframe": timeframe}

            await self.send_notification(asset, message, "rsi", "normal", None, metadata)

    async def send_notification(self, asset: str, message: str, signal_type: str = "default", 
                              priority: str = "normal", chart_path: str = None, metadata: Dict = None) -> bool:
        try:
            time_check = self.policy.check_time_rules(priority)
            if not time_check["allowed"]:
                self.logger.info(f"Notification blocked for {asset}: {time_check['reason']}")
                return False

            if not await self.policy.check_throttle(asset, signal_type, priority):
                return False

            success = await self._send_to_telegram(message, chart_path)

            if success:
                await self.policy.update_throttle(asset, signal_type, priority)
                self.logger.info(f"Notification sent: {asset} | {signal_type} | {priority}")
                
                if metadata:
                    metadata.update({"time_check": time_check})
                    await self._store_notification_metadata(asset, signal_type, metadata)
                return True
            else:
                await self._handle_send_failure(asset)
                return False

        except Exception as e:
            self.logger.error(f"Error sending notification for {asset}: {e}")
            return False

    async def _send_to_telegram(self, message: str, chart_path: str = None) -> bool:
        try:
            if chart_path and os.path.exists(chart_path):
                return self.telegram_client.send_photo(photo_path=chart_path, caption=message)
            else:
                return self.telegram_client.send_message(message=message)
        except Exception as e:
            self.logger.error(f"Telegram send failed: {e}")
            return False

    async def _store_notification_metadata(self, asset: str, signal_type: str, metadata: Dict):
        try:
            key = f"notification:metadata:{asset}:{datetime.now().strftime('%Y%m%d')}"
            await self.redis_handler.set(key, json.dumps(metadata), ttl=604800)
        except Exception as e:
            self.logger.error(f"Error storing metadata: {e}")

    async def _handle_send_failure(self, asset: str):
        try:
            failure_key = f"throttle:failures:{asset}"
            count = int(await self.redis_handler.get(failure_key) or 0) + 1
            await self.redis_handler.set(failure_key, str(count), ttl=3600)

            if count >= 3:
                await self.redis_handler.set(f"throttle:emergency:{asset}", "1", ttl=300)
                self.logger.warning(f"Emergency throttle activated for {asset}")
        except Exception as e:
            self.logger.error(f"Error handling failure: {e}")

    def _find_dataframe_for_timeframe(self, raw_results: Dict, timeframe: str) -> Optional[pd.DataFrame]:
        for data in raw_results.values():
            if data.get('metadata', {}).get('timeframe') == timeframe and 'data' in data:
                return data['data']
        return None
