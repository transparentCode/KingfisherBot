from typing import Dict, Any, List
import pandas as pd
from datetime import datetime
import os
import logging

from app.telegram.telegram_client import TelegramClient


class TelegramNotificationProcessor:
    def __init__(self, config, telegram_client: TelegramClient, telegram_enabled: bool = True):
        self.config = config
        self.telegram_client = telegram_client
        self.logger = logging.getLogger("app")
        self.telegram_enabled = telegram_enabled and telegram_client is not None
        self.aggregation_processor = None  # To be set externally if needed

        # Create charts directory if it doesn't exist
        if self.config.save_charts and not os.path.exists(self.config.charts_dir):
            os.makedirs(self.config.charts_dir)

    async def process_results(self, asset: str, results: Dict[str, Any]):
        """Process indicator results and send telegram notifications"""
        if not self.config.telegram_enabled or not self.telegram_client:
            return

        try:
            # First check if we have an aggregation processor with composite signals
            if hasattr(self, 'aggregation_processor') and self.aggregation_processor:
                composite_signals = self.aggregation_processor.get_aggregated_signals(asset)

                if composite_signals:
                    # Send composite signal notifications
                    await self._send_composite_signal_notifications(asset, composite_signals)
                    return  # Skip individual signal processing if we have composite signals

            # Fallback to original individual signal processing
            for result_key, result_data in results.items():
                metadata = result_data.get('metadata', {})
                timeframe = metadata.get('timeframe')

                if timeframe not in self.config.timeframes_to_monitor:
                    continue

                await self._check_and_send_notifications(asset, timeframe, result_data)

        except Exception as e:
            self.logger.error(f"Error processing telegram notifications for {asset}: {e}")

    async def _send_composite_signal_notifications(self, asset: str, composite_signals: List[Dict]):
        """Send notifications for composite signals"""
        for signal in composite_signals:
            try:
                direction = signal['direction']
                score = signal['composite_score']
                timeframe = signal['timeframe']
                current_price = signal['current_price']

                # Create enhanced message with composite signal info
                message = f"🎯 COMPOSITE SIGNAL ALERT 🎯\n\n" \
                          f"💰 Asset: {asset}\n" \
                          f"📈 Direction: {direction.upper()}\n" \
                          f"⏱️ Timeframe: {timeframe}\n" \
                          f"💵 Current Price: {current_price:.2f}\n" \
                          f"🔥 Signal Strength: {score}/10\n\n"

                # Add signal details
                message += "📊 Signal Components:\n"
                for detail in signal['signal_details']:
                    if detail.get('type') == 'upbreak' or detail.get('type') == 'downbreak':
                        message += f"• Trendline {detail['type']}\n"
                    elif detail.get('type') == 'price_above_ma':
                        message += f"• Price above {detail['count']} MAs\n"
                    elif detail.get('type') == 'price_below_ma':
                        message += f"• Price below {detail['count']} MAs\n"
                    elif detail.get('type') == 'ma_confluence':
                        message += f"• MA Confluence ({detail['strength']:.1%})\n"

                # Send the composite signal notification
                success = self.telegram_client.send_message(message=message)
                if success:
                    self.logger.info(f"Sent composite signal notification for {asset} {timeframe}")

            except Exception as e:
                self.logger.error(f"Error sending composite signal notification: {e}")

    async def _check_and_send_notifications(self, asset: str, timeframe: str, result_data: Dict):
        """Check for various signal types and send appropriate notifications"""
        try:
            metadata = result_data.get('metadata', {})
            latest_signal = metadata.get('latest_signal', {})

            # Check for trendline breakouts
            if self._has_breakout_signal(latest_signal):
                await self._send_breakout_notification(asset, timeframe, result_data, latest_signal)

            # Check for RSI overbought/oversold (if you have RSI indicators)
            elif metadata.get('category') == 'Oscillators' and 'rsi' in result_data:
                await self._check_rsi_notifications(asset, timeframe, result_data)

            # Add more notification types as needed

        except Exception as e:
            self.logger.error(f"Error checking notifications for {asset} {timeframe}: {e}")

    def _has_breakout_signal(self, signal: Dict[str, Any]) -> bool:
        """Check if signal contains breakout information"""
        return (signal.get('upbreak_signal') == 1 or
                signal.get('downbreak_signal') == 1)

    async def _send_breakout_notification(self, asset: str, timeframe: str, result_data: Dict, signal: Dict):
        """Send breakout notification - adapted from your original code"""
        try:
            df = result_data['data']
            latest_timestamp = df.index[-1]
            latest_bar = df.iloc[-1]

            # Determine breakout type
            breakout_type = "UPWARD" if signal.get('upbreak_signal') == 1 else "DOWNWARD"

            # Check if we should send notification (avoid duplicates)
            should_notify = await self._should_send_notification(asset, timeframe, latest_timestamp, breakout_type)

            if not should_notify:
                return

            # Get bar data
            bar_data = {
                'open': latest_bar['open'],
                'high': latest_bar['high'],
                'low': latest_bar['low'],
                'close': latest_bar['close'],
                'volume': latest_bar['volume']
            }

            # Save chart for notification if enabled
            chart_path = None
            if self.config.save_charts:
                chart_path = await self._save_chart(asset, timeframe, df, breakout_type, latest_timestamp)

            # Send the notification
            await self._send_telegram_message(
                asset=asset,
                timeframe=timeframe,
                breakout_type=breakout_type,
                timestamp=latest_timestamp,
                price=latest_bar['close'],
                bar_data=bar_data,
                chart_path=chart_path
            )

        except Exception as e:
            self.logger.error(f"Error sending breakout notification for {asset}: {e}")

    async def _should_send_notification(self, asset: str, timeframe: str, timestamp, signal_type: str) -> bool:
        """Check if we should send notification to avoid spam"""
        # You can implement logic to track sent notifications
        # For now, we'll assume it's handled by the database check in breakout processor
        return True

    async def _save_chart(self, asset: str, timeframe: str, df: pd.DataFrame,
                          breakout_type: str, breakout_timestamp) -> str:
        """Save chart of the breakout for notification - from your original code"""
        try:
            # For now, we'll create a simple chart filename
            # You'll need to integrate with your trendline indicator plotting
            timestamp_str = breakout_timestamp.strftime("%Y%m%d_%H%M%S")
            chart_filename = f"{asset.replace('/', '_')}_{timeframe}_{breakout_type.lower()}_{timestamp_str}.png"
            chart_path = os.path.join(self.config.charts_dir, chart_filename)

            # TODO: Integrate with your trendline indicator's plot method
            # This would require access to the specific indicator instance
            # For now, return the expected path

            self.logger.info(f"Chart path prepared: {chart_path}")
            return chart_path

        except Exception as e:
            self.logger.error(f"Error preparing chart path: {e}")
            return None

    async def _send_telegram_message(self, asset: str, timeframe: str, breakout_type: str, timestamp,
                                     price: float, bar_data: dict = None, chart_path: str = None):
        """Send breakout notification to Telegram - from your original code"""
        try:
            # Format timestamp for message
            formatted_time = timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')

            # Create message text
            message = f"📊 {breakout_type.upper()} BREAKOUT DETECTED 📊\n\n" \
                      f"💰 Asset: {asset}\n" \
                      f"⏱️ Timeframe: {timeframe}\n" \
                      f"🕒 Time: {formatted_time}\n" \
                      f"💵 Price: {price:.2f}"

            # Add OHLC data if provided
            if bar_data:
                message += f"\n\n📈 OHLC Data:\n" \
                           f"Open: {bar_data.get('open', 0):.2f}\n" \
                           f"High: {bar_data.get('high', 0):.2f}\n" \
                           f"Low: {bar_data.get('low', 0):.2f}\n" \
                           f"Close: {bar_data.get('close', 0):.2f}\n" \
                           f"Volume: {bar_data.get('volume', 0):.2f}"

            # Send message with photo if chart exists
            if chart_path and os.path.exists(chart_path):
                success = self.telegram_client.send_photo(
                    photo_path=chart_path,
                    caption=message
                )
                if success:
                    self.logger.info(f"Sent breakout notification with chart to Telegram for {asset} {timeframe}")
                else:
                    self.logger.warning(f"Failed to send chart for {asset} {timeframe}")
            else:
                # Fallback to text-only message
                success = self.telegram_client.send_message(message=message)
                if success:
                    self.logger.info(f"Sent text notification for {asset} {timeframe}")
                else:
                    self.logger.warning(f"Failed to send text notification for {asset} {timeframe}")

        except Exception as e:
            self.logger.error(f"Failed to send telegram notification for {asset} at {timeframe}: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())

    async def _check_rsi_notifications(self, asset: str, timeframe: str, result_data: Dict):
        """Check and send RSI-related notifications"""
        try:
            metadata = result_data.get('metadata', {})

            if metadata.get('overbought'):
                message = f"⚠️ RSI OVERBOUGHT ALERT ⚠️\n\n" \
                          f"💰 Asset: {asset}\n" \
                          f"⏱️ Timeframe: {timeframe}\n" \
                          f"📊 RSI: {metadata.get('latest_reading', 'N/A')}"

                self.telegram_client.send_message(message=message)

            elif metadata.get('oversold'):
                message = f"⚠️ RSI OVERSOLD ALERT ⚠️\n\n" \
                          f"💰 Asset: {asset}\n" \
                          f"⏱️ Timeframe: {timeframe}\n" \
                          f"📊 RSI: {metadata.get('latest_reading', 'N/A')}"

                self.telegram_client.send_message(message=message)

        except Exception as e:
            self.logger.error(f"Error checking RSI notifications: {e}")
