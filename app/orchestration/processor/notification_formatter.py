import os
import logging
import pandas as pd
from typing import Dict, Optional

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

class NotificationFormatter:
    """Handles message formatting and chart generation"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("app")
        
        charts_dir = self.config.get('charts_dir', './charts')
        if self.config.get('save_charts', False) and not os.path.exists(charts_dir):
            os.makedirs(charts_dir)

    def build_composite_message(self, asset: str, signal: Dict) -> str:
        direction = signal["direction"]
        score = signal["composite_score"]
        timeframe = signal["timeframe"]
        current_price = signal["current_price"]
        conviction = signal.get("conviction_level", "normal")

        conviction_emoji = {"high": "🔥", "medium": "⚡", "normal": "📊"}.get(conviction, "📊")
        direction_emoji = "🚀" if direction == "bullish" else "📉"

        message = (
            f"{conviction_emoji} COMPOSITE SIGNAL ALERT {conviction_emoji}\n\n"
            f"💰 Asset: {asset}\n"
            f"{direction_emoji} Direction: {direction.upper()}\n"
            f"⏱️ Timeframe: {timeframe}\n"
            f"💵 Current Price: ${current_price:.2f}\n"
            f"🎯 Signal Strength: {score}/10\n"
            f"🔥 Conviction: {conviction.upper()}\n\n"
        )

        if signal.get("signal_details"):
            message += "📊 Signal Components:\n"
            for detail in signal["signal_details"]:
                type_ = detail.get("type", "unknown")
                if type_ == "confluence_line":
                    message += f"• Confluence Line: {detail.get('distance_pct', 0):+.2f}% ({detail.get('position')})\n"
                elif type_ == "mtf_agreement":
                    message += f"• MTF Agreement: {len(detail.get('agreeing_timeframes', []))} timeframes\n"
                elif "breakout" in type_:
                    message += f"• Trendline {type_.title()}\n"
                elif "ma" in type_:
                    message += f"• MA Signals: {detail.get('count', 0)} confirmations\n"

        return message

    def build_breakout_message(self, asset: str, timeframe: str, type_: str, timestamp, price: float, session_info: Dict) -> str:
        formatted_time = timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")
        session_emoji = {"asia": "🌅", "europe": "🌞", "us": "🌙", "none": "⏰"}.get(session_info.get("session", "none"), "📊")

        return (
            f"{session_emoji} {type_.upper()} BREAKOUT DETECTED {session_emoji}\n\n"
            f"💰 Asset: {asset}\n"
            f"⏱️ Timeframe: {timeframe}\n"
            f"🕒 Time: {formatted_time}\n"
            f"💵 Price: ${price:.2f}\n"
            f"📈 Session: {session_info.get('session', 'Unknown').title()}"
        )

    def build_rsi_message(self, asset: str, timeframe: str, type_: str, value: float) -> str:
        emoji = "🔴" if type_ == "overbought" else "🟢"
        return (
            f"{emoji} RSI {type_.upper()} ALERT {emoji}\n\n"
            f"💰 Asset: {asset}\n"
            f"⏱️ Timeframe: {timeframe}\n"
            f"📊 RSI Value: {value}\n"
            f"⚠️ Condition: {type_.title()}"
        )

    async def generate_chart(self, asset: str, timeframe: str, df: pd.DataFrame, signal_type: str, timestamp) -> Optional[str]:
        if not HAS_PLOTLY or not self.config.get('save_charts', False):
            return None

        try:
            charts_dir = self.config.get('charts_dir', './charts')
            filename = f"{asset.replace('/', '_')}_{timeframe}_{signal_type}_{timestamp.strftime('%Y%m%d_%H%M%S')}.png"
            filepath = os.path.join(charts_dir, filename)

            plot_df = df.tail(50).copy()
            fig = go.Figure(data=[go.Candlestick(
                x=plot_df.index, open=plot_df['open'], high=plot_df['high'],
                low=plot_df['low'], close=plot_df['close'], name=asset
            )])

            fig.update_layout(
                title=f"{asset} - {timeframe} - {signal_type}",
                template="plotly_dark",
                xaxis_rangeslider_visible=False,
                height=600, width=1000,
                margin=dict(l=20, r=20, t=40, b=20)
            )

            fig.write_image(filepath)
            return filepath

        except Exception as e:
            self.logger.error(f"Error generating chart: {e}")
            if "kaleido" in str(e).lower():
                self.logger.warning("Install 'kaleido' for static image export")
            return None
