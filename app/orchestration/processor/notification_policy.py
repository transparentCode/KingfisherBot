from datetime import datetime
import logging
import pytz
import json
from typing import Dict, Any, Optional
from app.db.redis_handler import RedisHandler

class NotificationPolicy:
    """Handles throttling and time-awareness policies for notifications"""
    
    def __init__(self, config: Dict, redis_handler: RedisHandler):
        self.config = config
        self.redis_handler = redis_handler
        self.logger = logging.getLogger("app")
        
        self._load_throttling_configs()
        self._load_time_configs()

    def _load_throttling_configs(self):
        """Load configurable throttling settings"""
        self.global_throttle = self.config.get(
            "global_throttle",
            {
                "max_per_minute": 15,
                "max_per_hour": 50,
                "emergency_cooldown": 300,
            },
        )

        self.default_asset_throttle = self.config.get(
            "default_asset_throttle",
            {
                "cooldown_seconds": 300,
                "max_per_hour": 6,
                "priority_override": True,
                "signal_type_specific": {
                    "breakout": {"cooldown": 600, "max_per_hour": 3},
                    "composite": {"cooldown": 300, "max_per_hour": 4},
                    "rsi": {"cooldown": 900, "max_per_hour": 2},
                    "ma_confluence": {"cooldown": 450, "max_per_hour": 3},
                },
            },
        )
        self.asset_throttle_overrides = self.config.get("asset_throttle_overrides", {})

    def _load_time_configs(self):
        """Load time-aware configuration"""
        self.time_config = self.config.get(
            "time_awareness",
            {
                "enabled": True,
                "timezone": "Asia/Kolkata",
                "features": {
                    "active_hours_check": True,
                    "quiet_hours_check": True,
                    "weekend_schedule": True,
                    "session_multipliers": True,
                    "holiday_check": False,
                    "emergency_override": True
                },
                "active_hours": {
                    "enabled": True,
                    "weekdays": {"start": "09:00", "end": "17:30"},
                    "weekends": {"start": "10:00", "end": "16:00"},
                },
                "quiet_hours": {
                    "enabled": True,
                    "start": "22:00",
                    "end": "08:00",
                    "emergency_only": True,
                    "block_all": False,
                },
                "session_specific": {
                    "enabled": True,
                    "asia": {"start": "09:00", "end": "17:00", "multiplier": 1.0},
                    "europe": {"start": "13:30", "end": "22:00", "multiplier": 1.2},
                    "us": {"start": "19:30", "end": "02:00", "multiplier": 1.5},
                },
                "market_holidays": {
                    "enabled": False,
                    "dates": [],
                    "behavior": "quiet_hours"
                }
            },
        )

    def _get_asset_throttle_config(self, asset: str, signal_type: str = "default") -> Dict:
        config = self.default_asset_throttle.copy()
        if asset in self.asset_throttle_overrides:
            config.update(self.asset_throttle_overrides[asset])
        if signal_type in config.get("signal_type_specific", {}):
            config.update(config["signal_type_specific"][signal_type])
        return config

    async def check_throttle(self, asset: str, signal_type: str, priority: str = "normal") -> bool:
        """Check if notification should be throttled"""
        try:
            current_time = datetime.now()
            throttle_config = self._get_asset_throttle_config(asset, signal_type)

            if priority == "high" and throttle_config.get("priority_override", True):
                emergency_key = f"throttle:emergency:{asset}"
                if await self.redis_handler.get(emergency_key):
                    self.logger.warning(f"Emergency throttle active for {asset}")
                    return False
                return True

            cooldown_key = f"throttle:cooldown:{asset}:{signal_type}"
            last_notification = await self.redis_handler.get(cooldown_key)

            if last_notification:
                last_time = datetime.fromisoformat(last_notification)
                time_diff = (current_time - last_time).total_seconds()
                required_cooldown = throttle_config.get("cooldown_seconds", 300)
                if time_diff < required_cooldown:
                    return False

            hourly_key = f"throttle:hourly:{asset}:{current_time.strftime('%Y%m%d%H')}"
            hourly_count = int(await self.redis_handler.get(hourly_key) or 0)
            if hourly_count >= throttle_config.get("max_per_hour", 6):
                return False

            global_key = f"throttle:global:{current_time.strftime('%Y%m%d%H%M')}"
            global_count = int(await self.redis_handler.get(global_key) or 0)
            if global_count >= self.global_throttle["max_per_minute"]:
                return False

            return True
        except Exception as e:
            self.logger.error(f"Error checking throttle: {e}")
            return False

    async def update_throttle(self, asset: str, signal_type: str, priority: str = "normal"):
        """Update throttling counters"""
        try:
            current_time = datetime.now()
            throttle_config = self._get_asset_throttle_config(asset, signal_type)

            cooldown_key = f"throttle:cooldown:{asset}:{signal_type}"
            await self.redis_handler.set(
                cooldown_key, 
                current_time.isoformat(), 
                ttl=throttle_config.get("cooldown_seconds", 300)
            )

            hourly_key = f"throttle:hourly:{asset}:{current_time.strftime('%Y%m%d%H')}"
            await self.redis_handler.incr(hourly_key, ttl=3600)

            global_key = f"throttle:global:{current_time.strftime('%Y%m%d%H%M')}"
            await self.redis_handler.incr(global_key, ttl=60)

        except Exception as e:
            self.logger.error(f"Error updating throttle: {e}")

    def check_time_rules(self, priority: str = "normal") -> Dict[str, Any]:
        """Check time-based rules"""
        current_time = datetime.now()
        
        if not self.time_config.get("enabled", True):
            return {"allowed": True, "session": "none", "multiplier": 1.0, "reason": "disabled"}

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

        if priority == "emergency" and features.get("emergency_override", True):
            return {"allowed": True, "reason": "emergency_override"}

        if features.get("holiday_check", False):
            holiday_res = self._check_market_holidays(local_time, priority)
            if not holiday_res["allowed"]: return holiday_res

        if features.get("quiet_hours_check", True):
            quiet_res = self._check_quiet_hours(current_hour_min, priority)
            if not quiet_res["allowed"]: return quiet_res

        if features.get("active_hours_check", True):
            active_res = self._check_active_hours(current_hour_min, is_weekend, priority)
            if not active_res["allowed"]: return active_res

        if features.get("session_multipliers", True):
            result.update(self._calculate_session_multiplier(current_hour_min))

        return result

    def _check_market_holidays(self, local_time: datetime, priority: str) -> Dict[str, Any]:
        config = self.time_config.get("market_holidays", {})
        if not config.get("enabled", False): return {"allowed": True}
        
        if local_time.strftime("%Y-%m-%d") in config.get("dates", []):
            behavior = config.get("behavior", "quiet_hours")
            if behavior == "block_all": return {"allowed": False, "reason": "holiday_blocked"}
            if behavior == "quiet_hours": return {"allowed": priority == "emergency", "reason": "holiday_quiet"}
        return {"allowed": True}

    def _check_quiet_hours(self, current_hour_min: str, priority: str) -> Dict[str, Any]:
        config = self.time_config.get("quiet_hours", {})
        if not config.get("enabled", True): return {"allowed": True}
        
        start, end = config["start"], config["end"]
        is_quiet = (current_hour_min >= start or current_hour_min <= end) if start > end else (start <= current_hour_min <= end)
        
        if is_quiet:
            if config.get("block_all", False): return {"allowed": priority == "emergency", "reason": "quiet_block"}
            if config.get("emergency_only", True): return {"allowed": priority == "emergency", "reason": "quiet_emergency"}
        return {"allowed": True}

    def _check_active_hours(self, current_hour_min: str, is_weekend: bool, priority: str) -> Dict[str, Any]:
        config = self.time_config.get("active_hours", {})
        if not config.get("enabled", True): return {"allowed": True}
        
        schedule = config.get("weekends" if is_weekend and self.time_config["features"].get("weekend_schedule") else "weekdays", {})
        if not schedule: return {"allowed": True}
        
        if schedule["start"] <= current_hour_min <= schedule["end"]: return {"allowed": True}
        return {"allowed": False, "reason": "outside_active_hours"}

    def _calculate_session_multiplier(self, current_hour_min: str) -> Dict[str, Any]:
        config = self.time_config.get("session_specific", {})
        if not config.get("enabled", True): return {"session": "none", "multiplier": 1.0}
        
        for session, data in config.items():
            if session == "enabled": continue
            start, end = data.get("start"), data.get("end")
            if not (start and end): continue
            
            is_session = (current_hour_min >= start or current_hour_min <= end) if start > end else (start <= current_hour_min <= end)
            if is_session: return {"session": session, "multiplier": data.get("multiplier", 1.0)}
            
        return {"session": "none", "multiplier": 1.0}

    async def check_global_throttle(self) -> bool:
        try:
            return not bool(await self.redis_handler.get("throttle:global:emergency"))
        except:
            return True
