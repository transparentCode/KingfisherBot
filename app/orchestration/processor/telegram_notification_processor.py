from datetime import datetime
from app.db.redis_handler import RedisHandler
import logging
import os
from typing import Dict, Any, List
import pytz
import json

import pandas as pd

from app.telegram.telegram_client import TelegramClient


class TelegramNotificationProcessor:
    """Enhanced processor with Redis-based throttling and configurable per-asset settings"""

    def __init__(
        self, config, telegram_client: TelegramClient, telegram_enabled: bool = True
    ):
        self.config = config
        self.telegram_client = telegram_client
        self.logger = logging.getLogger("app")
        self.telegram_enabled = telegram_enabled and telegram_client is not None
        self.aggregation_processor = None
        self.redis_handler = RedisHandler()

        # Load throttling configurations
        self._load_throttling_configs()

        # Load time-aware configurations
        self._load_time_configs()

        # Create charts directory if it doesn't exist
        if self.config.save_charts and not os.path.exists(self.config.charts_dir):
            os.makedirs(self.config.charts_dir)

    def _load_throttling_configs(self):
        """Load configurable throttling settings"""
        # Global throttling defaults
        self.global_throttle = self.config.get(
            "global_throttle",
            {
                "max_per_minute": 15,
                "max_per_hour": 50,
                "emergency_cooldown": 300,  # 5 minutes emergency brake
            },
        )

        # Per-asset throttling defaults
        self.default_asset_throttle = self.config.get(
            "default_asset_throttle",
            {
                "cooldown_seconds": 300,  # 5 minutes between notifications
                "max_per_hour": 6,  # Max 6 notifications per hour per asset
                "priority_override": True,  # High priority can override cooldown
                "signal_type_specific": {
                    "breakout": {
                        "cooldown": 600,
                        "max_per_hour": 3,
                    },  # 10 min for breakouts
                    "composite": {
                        "cooldown": 300,
                        "max_per_hour": 4,
                    },  # 5 min for composite
                    "rsi": {"cooldown": 900, "max_per_hour": 2},  # 15 min for RSI
                    "ma_confluence": {
                        "cooldown": 450,
                        "max_per_hour": 3,
                    },  # 7.5 min for MA
                },
            },
        )

        # Per-asset specific overrides
        self.asset_throttle_overrides = self.config.get("asset_throttle_overrides", {})

    def _load_time_configs(self):
        """Load time-aware configuration"""
        self.time_config = self.config.get(
            "time_awareness",
            {
                "enabled": True,  # Master switch for all time-based checks
                "timezone": "Asia/Kolkata",
                
                # Individual feature toggles
                "features": {
                    "active_hours_check": True,      # Enable/disable active hours check
                    "quiet_hours_check": True,       # Enable/disable quiet hours check
                    "weekend_schedule": True,        # Enable/disable weekend vs weekday logic
                    "session_multipliers": True,     # Enable/disable session-based multipliers
                    "holiday_check": False,          # Enable/disable holiday checking
                    "emergency_override": True       # Allow emergency priority to bypass all checks
                },
                
                "active_hours": {
                    "enabled": True,  # Specific toggle for active hours
                    "weekdays": {"start": "09:00", "end": "17:30"},
                    "weekends": {"start": "10:00", "end": "16:00"},
                },
                
                "quiet_hours": {
                    "enabled": True,  # Specific toggle for quiet hours
                    "start": "22:00",
                    "end": "08:00",
                    "emergency_only": True,
                    "block_all": False,  # If true, blocks ALL notifications during quiet hours
                },
                
                "session_specific": {
                    "enabled": True,  # Toggle for session multipliers
                    "asia": {"start": "09:00", "end": "17:00", "multiplier": 1.0},
                    "europe": {"start": "13:30", "end": "22:00", "multiplier": 1.2},
                    "us": {"start": "19:30", "end": "02:00", "multiplier": 1.5},
                },
                
                "market_holidays": {
                    "enabled": False,  # Toggle for holiday checking
                    "dates": [],  # List of holiday dates
                    "behavior": "quiet_hours"  # 'block_all', 'quiet_hours', 'normal'
                }
            },
        )

    async def initialize(self):
        """Initialize Redis connection"""
        await self.redis_handler.initialize()
        self.logger.info("TelegramNotificationProcessor initialized with Redis")

    def _get_asset_throttle_config(
        self, asset: str, signal_type: str = "default"
    ) -> Dict:
        """Get throttle configuration for specific asset and signal type"""
        # Start with default
        config = self.default_asset_throttle.copy()

        # Apply asset-specific overrides
        if asset in self.asset_throttle_overrides:
            asset_config = self.asset_throttle_overrides[asset]
            config.update(asset_config)

        # Apply signal-type specific settings
        if signal_type in config.get("signal_type_specific", {}):
            signal_config = config["signal_type_specific"][signal_type]
            config.update(signal_config)

        return config

    async def _check_throttle_redis(
        self, asset: str, signal_type: str, priority: str = "normal"
    ) -> bool:
        """Advanced Redis-based throttling check"""
        try:
            current_time = datetime.now()
            throttle_config = self._get_asset_throttle_config(asset, signal_type)

            # High priority override check
            if priority == "high" and throttle_config.get("priority_override", True):
                # Still check for emergency cooldown
                emergency_key = f"throttle:emergency:{asset}"
                emergency_block = await self.redis_handler.get(emergency_key)
                if emergency_block:
                    self.logger.warning(f"Emergency throttle active for {asset}")
                    return False
                return True

            # Check cooldown period
            cooldown_key = f"throttle:cooldown:{asset}:{signal_type}"
            last_notification = await self.redis_handler.get(cooldown_key)

            if last_notification:
                last_time = datetime.fromisoformat(last_notification)
                time_diff = (current_time - last_time).total_seconds()
                required_cooldown = throttle_config.get("cooldown_seconds", 300)

                if time_diff < required_cooldown:
                    self.logger.debug(
                        f"Cooldown active for {asset}:{signal_type} - {required_cooldown - time_diff:.0f}s remaining"
                    )
                    return False

            # Check hourly limits
            hourly_key = f"throttle:hourly:{asset}:{current_time.strftime('%Y%m%d%H')}"
            hourly_count = await self.redis_handler.get(hourly_key)
            hourly_count = int(hourly_count) if hourly_count else 0

            max_hourly = throttle_config.get("max_per_hour", 6)
            if hourly_count >= max_hourly:
                self.logger.debug(
                    f"Hourly limit reached for {asset} ({hourly_count}/{max_hourly})"
                )
                return False

            # Check global limits
            global_key = f"throttle:global:{current_time.strftime('%Y%m%d%H%M')}"
            global_count = await self.redis_handler.get(global_key)
            global_count = int(global_count) if global_count else 0

            if global_count >= self.global_throttle["max_per_minute"]:
                self.logger.debug(f"Global per-minute limit reached ({global_count})")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error checking throttle for {asset}: {e}")
            return False  # Fail-safe: don't send if check fails

    async def _update_throttle_redis(
        self, asset: str, signal_type: str, priority: str = "normal"
    ):
        """Update Redis throttling counters"""
        try:
            current_time = datetime.now()
            throttle_config = self._get_asset_throttle_config(asset, signal_type)

            # Update cooldown timestamp
            cooldown_key = f"throttle:cooldown:{asset}:{signal_type}"
            cooldown_ttl = throttle_config.get("cooldown_seconds", 300)
            await self.redis_handler.set(
                cooldown_key, current_time.isoformat(), ttl=cooldown_ttl
            )

            # Update hourly counter
            hourly_key = f"throttle:hourly:{asset}:{current_time.strftime('%Y%m%d%H')}"
            await self.redis_handler.set(hourly_key, "1", ttl=3600)  # 1 hour TTL
            hourly_count = await self.redis_handler.get(hourly_key)
            if hourly_count:
                await self.redis_handler.set(
                    hourly_key, str(int(hourly_count) + 1), ttl=3600
                )

            # Update global counter
            global_key = f"throttle:global:{current_time.strftime('%Y%m%d%H%M')}"
            global_count = await self.redis_handler.get(global_key)
            if global_count:
                await self.redis_handler.set(
                    global_key, str(int(global_count) + 1), ttl=60
                )
            else:
                await self.redis_handler.set(global_key, "1", ttl=60)

            # Track notification history for analytics
            history_key = f"notifications:history:{asset}"
            notification_data = {
                "timestamp": current_time.isoformat(),
                "signal_type": signal_type,
                "priority": priority,
            }
            await self.redis_handler.set(
                f"{history_key}:{current_time.strftime('%Y%m%d%H%M%S')}",
                json.dumps(notification_data),
                ttl=86400,  # 24 hours
            )

        except Exception as e:
            self.logger.error(f"Error updating throttle for {asset}: {e}")

    def _is_within_active_hours(self, current_time: datetime = None, priority: str = "normal") -> Dict[str, Any]:
        """Enhanced configurable time-awareness with granular enable/disable controls"""
        if current_time is None:
            current_time = datetime.now()

        # Master switch check
        if not self.time_config.get("enabled", True):
            return {
                "allowed": True,
                "session": "none",
                "multiplier": 1.0,
                "reason": "time_awareness_disabled",
                "checks_performed": []
            }

        tz = pytz.timezone(self.time_config["timezone"])
        local_time = current_time.astimezone(tz)
        current_hour_min = local_time.strftime("%H:%M")
        is_weekend = local_time.weekday() >= 5

        result = {
            "allowed": True,
            "session": "none",
            "multiplier": 1.0,
            "reason": "active_hours",
            "checks_performed": []
        }

        features = self.time_config.get("features", {})

        # Emergency override check
        if priority == "emergency" and features.get("emergency_override", True):
            result.update({
                "allowed": True,
                "reason": "emergency_override",
                "checks_performed": ["emergency_override"]
            })
            return result

        # Holiday check
        if features.get("holiday_check", False):
            holiday_result = self._check_market_holidays(local_time, priority)
            result["checks_performed"].append("holiday_check")
            if not holiday_result["allowed"]:
                result.update(holiday_result)
                return result

        # Quiet hours check
        if features.get("quiet_hours_check", True):
            quiet_result = self._check_quiet_hours(current_hour_min, priority)
            result["checks_performed"].append("quiet_hours_check")
            if not quiet_result["allowed"]:
                result.update(quiet_result)
                return result

        # Active hours check
        if features.get("active_hours_check", True):
            active_result = self._check_active_hours(current_hour_min, is_weekend, priority)
            result["checks_performed"].append("active_hours_check")
            if not active_result["allowed"]:
                result.update(active_result)
                return result

        # Session multiplier calculation
        if features.get("session_multipliers", True):
            session_result = self._calculate_session_multiplier(current_hour_min)
            result["checks_performed"].append("session_multipliers")
            result.update(session_result)

        return result

    def _check_market_holidays(self, local_time: datetime, priority: str) -> Dict[str, Any]:
        """Check if current date is a market holiday"""
        holiday_config = self.time_config.get("market_holidays", {})
        
        if not holiday_config.get("enabled", False):
            return {"allowed": True}
        
        current_date = local_time.strftime("%Y-%m-%d")
        holiday_dates = holiday_config.get("dates", [])
        
        if current_date in holiday_dates:
            behavior = holiday_config.get("behavior", "quiet_hours")
            
            if behavior == "block_all":
                return {"allowed": False, "reason": "market_holiday_blocked"}
            elif behavior == "quiet_hours":
                # Treat as quiet hours - only emergency allowed
                return {"allowed": priority == "emergency", "reason": "market_holiday_quiet"}

    def _check_quiet_hours(self, current_hour_min: str, priority: str) -> Dict[str, Any]:
        """Check quiet hours with configurable behavior"""
        quiet_hours = self.time_config.get("quiet_hours", {})
        
        if not quiet_hours.get("enabled", True):
            return {"allowed": True}
        
        quiet_start = quiet_hours["start"]
        quiet_end = quiet_hours["end"]
        
        is_quiet_time = False
        if quiet_start > quiet_end:  # Overnight quiet hours
            is_quiet_time = current_hour_min >= quiet_start or current_hour_min <= quiet_end
        else:  # Same day quiet hours
            is_quiet_time = quiet_start <= current_hour_min <= quiet_end
        
        if is_quiet_time:
            if quiet_hours.get("block_all", False):
                # Block everything except emergency
                return {"allowed": priority == "emergency", "reason": "quiet_hours_block_all"}
            elif quiet_hours.get("emergency_only", True):
                # Only emergency allowed
                return {"allowed": priority == "emergency", "reason": "quiet_hours_emergency_only"}
            else:
                # Allow all (quiet hours disabled effectively)
                return {"allowed": True, "reason": "quiet_hours_allow_all"}
    
        return {"allowed": True}

    def _check_active_hours(self, current_hour_min: str, is_weekend: bool, priority: str) -> Dict[str, Any]:
        """Check active hours with weekend/weekday logic"""
        active_hours = self.time_config.get("active_hours", {})
        
        if not active_hours.get("enabled", True):
            return {"allowed": True}
        
        features = self.time_config.get("features", {})
        
        # Determine schedule based on weekend toggle
        if features.get("weekend_schedule", True) and is_weekend:
            schedule = active_hours.get("weekends", active_hours.get("weekdays", {}))
        else:
            schedule = active_hours.get("weekdays", {})
        
        if not schedule:
            return {"allowed": True}  # No schedule defined
        
        start_time = schedule.get("start")
        end_time = schedule.get("end")
        
        if not (start_time and end_time):
            return {"allowed": True}  # Invalid schedule
        
        if start_time <= current_hour_min <= end_time:
            return {"allowed": True}
        else:
            return {"allowed": False, "reason": "outside_active_hours"}

    def _calculate_session_multiplier(self, current_hour_min: str) -> Dict[str, Any]:
        """Calculate session multiplier if enabled"""
        session_config = self.time_config.get("session_specific", {})
        
        if not session_config.get("enabled", True):
            return {"session": "none", "multiplier": 1.0}
        
        for session, config in session_config.items():
            if session == "enabled":  # Skip the enabled flag
                continue
                
            session_start = config.get("start")
            session_end = config.get("end")
            
            if not (session_start and session_end):
                continue
            
            if session_start > session_end:  # Overnight session
                if current_hour_min >= session_start or current_hour_min <= session_end:
                    return {
                        "session": session,
                        "multiplier": config.get("multiplier", 1.0)
                    }
            else:  # Same day session
                if session_start <= current_hour_min <= session_end:
                    return {
                        "session": session,
                        "multiplier": config.get("multiplier", 1.0)
                    }
        
        return {"session": "none", "multiplier": 1.0}

    async def send_notification(
        self,
        asset: str,
        message: str,
        signal_type: str = "default",
        priority: str = "normal",
        chart_path: str = None,
        metadata: Dict = None,
    ) -> bool:
        """Enhanced unified notification sending with configurable time awareness"""
        try:
            # Configurable time-awareness check
            time_check = self._is_within_active_hours(priority=priority)
            
            if not time_check["allowed"]:
                self.logger.info(
                    f"Notification blocked for {asset}: {time_check['reason']} "
                    f"(checks: {', '.join(time_check.get('checks_performed', []))})"
                )
                return False

            # Apply session multiplier to throttling if enabled
            session_multiplier = time_check.get("multiplier", 1.0)
            
            # Throttling check
            if not await self._check_throttle_redis(asset, signal_type, priority):
                return False

            # Send the notification
            success = await self._send_to_telegram(message, chart_path)

            if success:
                # Update throttling counters
                await self._update_throttle_redis(asset, signal_type, priority)

                # Enhanced logging with time check details
                self.logger.info(
                    f"Notification sent: {asset} | {signal_type} | {priority} | "
                    f"Session: {time_check.get('session', 'none')} | "
                    f"Multiplier: {session_multiplier} | "
                    f"Time checks: {', '.join(time_check.get('checks_performed', []))}"
                )

                # Store notification metadata for analytics
                if metadata:
                    # Add time check results to metadata
                    metadata.update({
                        "time_check_results": time_check,
                        "session_multiplier": session_multiplier
                    })
                    await self._store_notification_metadata(asset, signal_type, metadata)

                return True
            else:
                await self._handle_send_failure(asset)
                return False

        except Exception as e:
            self.logger.error(f"Error sending notification for {asset}: {e}")
            return False

    async def _store_notification_metadata(
        self, asset: str, signal_type: str, metadata: Dict
    ):
        """Store notification metadata for analytics"""
        try:
            key = f"notification:metadata:{asset}:{datetime.now().strftime('%Y%m%d')}"
            await self.redis_handler.set(
                key, json.dumps(metadata), ttl=604800
            )  # 7 days
        except Exception as e:
            self.logger.error(f"Error storing notification metadata: {e}")

    async def _handle_send_failure(self, asset: str):
        """Handle repeated send failures with emergency throttling"""
        try:
            failure_key = f"throttle:failures:{asset}"
            failure_count = await self.redis_handler.get(failure_key)
            failure_count = int(failure_count) if failure_count else 0
            failure_count += 1

            await self.redis_handler.set(
                failure_key, str(failure_count), ttl=3600
            )  # 1 hour

            if failure_count >= 3:  # 3 failures trigger emergency throttle
                emergency_key = f"throttle:emergency:{asset}"
                await self.redis_handler.set(
                    emergency_key, "1", ttl=self.global_throttle["emergency_cooldown"]
                )
                self.logger.warning(
                    f"Emergency throttle activated for {asset} after {failure_count} failures"
                )

        except Exception as e:
            self.logger.error(f"Error handling send failure: {e}")

    async def process_results(self, asset: str, results: Dict[str, Any]):
        """Enhanced result processing with Redis-based efficiency"""
        if not self.telegram_enabled:
            return

        try:
            self.logger.info(f"Processing results for asset {asset}")

            # Check if notifications are globally throttled
            if not await self._check_global_throttle():
                self.logger.info("Global throttle active - skipping all notifications")
                return

            # Process composite signals first (higher priority)
            if hasattr(self, "aggregation_processor") and self.aggregation_processor:
                composite_signals = self.aggregation_processor.get_aggregated_signals(
                    asset
                )
                if composite_signals:
                    await self._process_composite_signals(asset, composite_signals)
                    return  # Skip individual signals if composite exists

            # Process individual signals
            await self._process_individual_signals(asset, results)

        except Exception as e:
            self.logger.error(
                f"Error processing telegram notifications for {asset}: {e}"
            )
            import traceback

            self.logger.error(traceback.format_exc())

    async def _check_global_throttle(self) -> bool:
        """Check if global throttling is active"""
        try:
            global_emergency = await self.redis_handler.get("throttle:global:emergency")
            return not bool(global_emergency)
        except:
            return True  # Allow if check fails

    async def _process_composite_signals(
        self, asset: str, composite_signals: List[Dict]
    ):
        """Process composite signals with enhanced metadata"""
        for signal in composite_signals:
            try:
                # Determine priority based on conviction and score
                conviction_level = signal.get("conviction_level", "normal")
                composite_score = signal.get("composite_score", 0)

                if conviction_level == "high" or composite_score >= 8:
                    priority = "high"
                elif conviction_level == "medium" or composite_score >= 6:
                    priority = "medium"
                else:
                    priority = "normal"

                # Build enhanced message
                message = self._build_composite_message(asset, signal)

                # Prepare metadata for analytics
                metadata = {
                    "signal_type": "composite",
                    "direction": signal["direction"],
                    "score": composite_score,
                    "conviction": conviction_level,
                    "timeframe": signal["timeframe"],
                    "components": len(signal.get("signal_details", [])),
                }

                # Send notification
                await self.send_notification(
                    asset=asset,
                    message=message,
                    signal_type="composite",
                    priority=priority,
                    metadata=metadata,
                )

            except Exception as e:
                self.logger.error(f"Error processing composite signal for {asset}: {e}")

    async def _process_individual_signals(self, asset: str, results: Dict[str, Any]):
        """Process individual signals with type-specific handling"""
        for result_key, result_data in results.items():
            try:
                metadata = result_data.get("metadata", {})
                timeframe = metadata.get("timeframe")

                if timeframe not in self.config.timeframes_to_monitor:
                    continue

                # Route to specific signal handlers
                category = metadata.get("category", "")
                if category == "Trend Analysis":
                    await self._handle_trendline_signals(asset, timeframe, result_data)
                elif category == "Oscillators":
                    await self._handle_oscillator_signals(asset, timeframe, result_data)
                elif category == "Moving Averages":
                    await self._handle_ma_signals(asset, timeframe, result_data)

            except Exception as e:
                self.logger.error(
                    f"Error processing individual signal {result_key}: {e}"
                )

    async def _handle_trendline_signals(
        self, asset: str, timeframe: str, result_data: Dict
    ):
        """Handle trendline breakout signals"""
        metadata = result_data.get("metadata", {})
        latest_signal = metadata.get("latest_signal", {})

        if self._has_breakout_signal(latest_signal):
            breakout_type = (
                "upward" if latest_signal.get("upbreak_signal") == 1 else "downward"
            )

            # Build message and prepare chart
            df = result_data["data"]
            latest_bar = df.iloc[-1]
            message = self._build_breakout_message(
                asset,
                timeframe,
                breakout_type,
                df.index[-1],
                latest_bar["close"],
                latest_bar.to_dict(),
            )

            chart_path = None
            if self.config.save_charts:
                chart_path = await self._save_chart(
                    asset, timeframe, df, breakout_type, df.index[-1]
                )

            metadata = {
                "signal_type": "breakout",
                "direction": breakout_type,
                "timeframe": timeframe,
                "price": latest_bar["close"],
            }

            await self.send_notification(
                asset=asset,
                message=message,
                signal_type="breakout",
                priority="medium",
                chart_path=chart_path,
                metadata=metadata,
            )

    async def _handle_oscillator_signals(
        self, asset: str, timeframe: str, result_data: Dict
    ):
        """Handle RSI and other oscillator signals"""
        metadata = result_data.get("metadata", {})

        if metadata.get("overbought") or metadata.get("oversold"):
            signal_type = "overbought" if metadata.get("overbought") else "oversold"
            rsi_value = metadata.get("latest_reading", "N/A")

            message = self._build_rsi_message(asset, timeframe, signal_type, rsi_value)

            metadata_dict = {
                "signal_type": "rsi",
                "condition": signal_type,
                "value": rsi_value,
                "timeframe": timeframe,
            }

            await self.send_notification(
                asset=asset,
                message=message,
                signal_type="rsi",
                priority="normal",
                metadata=metadata_dict,
            )

    async def _handle_ma_signals(self, asset: str, timeframe: str, result_data: Dict):
        """Handle moving average confluence signals"""
        # Implementation for MA-specific signals
        # You can add logic for MA crossovers, confluence changes, etc.
        pass

    # Keep all your existing utility methods for building messages, saving charts, etc.
    def _build_breakout_message(
        self,
        asset: str,
        timeframe: str,
        breakout_type: str,
        timestamp,
        price: float,
        bar_data: dict = None,
    ) -> str:
        """Build breakout message with enhanced formatting"""
        formatted_time = timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")

        # Session-aware emoji selection
        time_check = self._is_within_active_hours()
        session_emoji = {"asia": "🌅", "europe": "🌞", "us": "🌙", "none": "⏰"}.get(
            time_check.get("session", "none"), "📊"
        )

        message = (
            f"{session_emoji} {breakout_type.upper()} BREAKOUT DETECTED {session_emoji}\n\n"
            f"💰 Asset: {asset}\n"
            f"⏱️ Timeframe: {timeframe}\n"
            f"🕒 Time: {formatted_time}\n"
            f"💵 Price: ${price:.2f}\n"
            f"📈 Session: {time_check.get('session', 'Unknown').title()}"
        )

        if bar_data:
            message += (
                f"\n\n📊 OHLC Data:\n"
                f"🟢 Open: ${bar_data['open']:.2f}\n"
                f"⬆️ High: ${bar_data['high']:.2f}\n"
                f"⬇️ Low: ${bar_data['low']:.2f}\n"
                f"🔴 Close: ${bar_data['close']:.2f}\n"
                f"📈 Volume: {bar_data['volume']:,.0f}"
            )

        return message

    def _build_composite_message(self, asset: str, signal: Dict) -> str:
        """Build enhanced composite signal message"""
        direction = signal["direction"]
        score = signal["composite_score"]
        timeframe = signal["timeframe"]
        current_price = signal["current_price"]
        conviction = signal.get("conviction_level", "normal")

        # Conviction emoji mapping
        conviction_emoji = {"high": "🔥", "medium": "⚡", "normal": "📊"}.get(
            conviction, "📊"
        )

        # Direction emoji mapping
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

        # Add signal components
        signal_details = signal.get("signal_details", [])
        if signal_details:
            message += "📊 Signal Components:\n"
            for detail in signal_details:
                component_type = detail.get("type", "unknown")
                if component_type == "confluence_line":
                    distance = detail.get("distance_pct", 0)
                    position = detail.get("position", "unknown")
                    message += f"• Confluence Line: {distance:+.2f}% ({position})\n"
                elif component_type == "mtf_agreement":
                    agreeing_tfs = detail.get("agreeing_timeframes", [])
                    message += f"• MTF Agreement: {len(agreeing_tfs)} timeframes\n"
                elif "breakout" in component_type:
                    message += f"• Trendline {component_type.title()}\n"
                elif "ma" in component_type:
                    count = detail.get("count", 0)
                    message += f"• MA Signals: {count} confirmations\n"

        return message

    def _build_rsi_message(
        self, asset: str, timeframe: str, rsi_type: str, rsi_value: float
    ) -> str:
        """Build RSI alert message"""
        condition_emoji = "🔴" if rsi_type == "overbought" else "🟢"

        return (
            f"{condition_emoji} RSI {rsi_type.upper()} ALERT {condition_emoji}\n\n"
            f"💰 Asset: {asset}\n"
            f"⏱️ Timeframe: {timeframe}\n"
            f"📊 RSI Value: {rsi_value}\n"
            f"⚠️ Condition: {rsi_type.title()}"
        )

    # Keep your existing helper methods for chart saving, signal detection, etc.
    def _has_breakout_signal(self, signal: Dict[str, Any]) -> bool:
        """Check if signal contains breakout information"""
        return signal.get("upbreak_signal") == 1 or signal.get("downbreak_signal") == 1

    async def _save_chart(
        self,
        asset: str,
        timeframe: str,
        df: pd.DataFrame,
        breakout_type: str,
        breakout_timestamp,
    ) -> str:
        """Save chart for notification"""
        try:
            timestamp_str = breakout_timestamp.strftime("%Y%m%d_%H%M%S")
            chart_filename = f"{asset.replace('/', '_')}_{timeframe}_{breakout_type.lower()}_{timestamp_str}.png"
            chart_path = os.path.join(self.config.charts_dir, chart_filename)

            # TODO: Integrate with your plotting logic
            self.logger.info(f"Chart path prepared: {chart_path}")
            return chart_path
        except Exception as e:
            self.logger.error(f"Error preparing chart path: {e}")
            return None

    async def _send_to_telegram(self, message: str, chart_path: str = None) -> bool:
        """Core Telegram sending function"""
        try:
            if chart_path and os.path.exists(chart_path):
                return self.telegram_client.send_photo(
                    photo_path=chart_path, caption=message
                )
            else:
                return self.telegram_client.send_message(message=message)
        except Exception as e:
            self.logger.error(f"Telegram send failed: {e}")
            return False

    # Analytics and monitoring methods
    async def get_throttle_stats(self, asset: str = None) -> Dict[str, Any]:
        """Get throttling statistics for monitoring"""
        try:
            stats = {"global": {}, "assets": {}, "system": {}}

            current_time = datetime.now()

            # Global stats
            global_key = f"throttle:global:{current_time.strftime('%Y%m%d%H%M')}"
            global_count = await self.redis_handler.get(global_key)
            stats["global"]["current_minute"] = int(global_count) if global_count else 0

            # Asset-specific stats
            if asset:
                for signal_type in ["breakout", "composite", "rsi", "ma_confluence"]:
                    cooldown_key = f"throttle:cooldown:{asset}:{signal_type}"
                    last_notification = await self.redis_handler.get(cooldown_key)

                    if last_notification:
                        last_time = datetime.fromisoformat(last_notification)
                        time_since = (current_time - last_time).total_seconds()
                        stats["assets"][f"{asset}_{signal_type}"] = {
                            "last_notification": last_notification,
                            "seconds_since": time_since,
                        }

            return stats

        except Exception as e:
            self.logger.error(f"Error getting throttle stats: {e}")
            return {}

    async def reset_throttle(self, asset: str = None, signal_type: str = None):
        """Reset throttling for debugging/emergency (admin function)"""
        try:
            if asset and signal_type:
                # Reset specific asset/signal throttle
                cooldown_key = f"throttle:cooldown:{asset}:{signal_type}"
                await self.redis_handler.delete(cooldown_key)
                self.logger.info(f"Reset throttle for {asset}:{signal_type}")
            elif asset:
                # Reset all throttles for asset
                pattern = f"throttle:*:{asset}:*"
                deleted = await self.redis_handler.clear_pattern(pattern)
                self.logger.info(f"Reset {deleted} throttle keys for {asset}")
            else:
                # Reset global emergency throttle
                await self.redis_handler.delete("throttle:global:emergency")
                self.logger.info("Reset global emergency throttle")

        except Exception as e:
            self.logger.error(f"Error resetting throttle: {e}")

    def _validate_time_config(self) -> bool:
        """Validate time configuration settings"""
        try:
            config = self.time_config
            
            # Check timezone validity
            try:
                pytz.timezone(config.get("timezone", "UTC"))
            except pytz.exceptions.UnknownTimeZoneError:
                self.logger.warning(f"Invalid timezone: {config.get('timezone')}, using UTC")
                config["timezone"] = "UTC"
            
            # Validate time formats
            for schedule_type in ["active_hours", "quiet_hours"]:
                if schedule_type in config:
                    schedule = config[schedule_type]
                    for time_key in ["start", "end"]:
                        if time_key in schedule:
                            time_str = schedule[time_key]
                            try:
                                datetime.strptime(time_str, "%H:%M")
                            except ValueError:
                                self.logger.warning(f"Invalid time format in {schedule_type}.{time_key}: {time_str}")
                                return False
            
            # Validate session times
            session_config = config.get("session_specific", {})
            for session, session_data in session_config.items():
                if session == "enabled":
                    continue
                for time_key in ["start", "end"]:
                    if time_key in session_data:
                        try:
                            datetime.strptime(session_data[time_key], "%H:%M")
                        except ValueError:
                            self.logger.warning(f"Invalid time format in session {session}.{time_key}")
                            return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating time config: {e}")
            return False

    def get_time_config_status(self) -> Dict[str, Any]:
        """Get current time configuration status for monitoring"""
        return {
            "master_enabled": self.time_config.get("enabled", True),
            "features": self.time_config.get("features", {}),
            "timezone": self.time_config.get("timezone", "UTC"),
            "current_session": self._calculate_session_multiplier(datetime.now().strftime("%H:%M")),
            "validation_passed": self._validate_time_config()
        }
