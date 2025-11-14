import logging
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, Optional, List

import yaml

from app.models.AssetIndicatorConfig import AssetIndicatorConfig


class ConfigurationManager:
    """Singleton configuration manager"""

    _instance: Optional['ConfigurationManager'] = None
    _lock = threading.Lock()

    def __new__(cls, config_dir: str = "..") -> 'ConfigurationManager':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_dir: str = ".."):
        # Prevent re-initialization
        if hasattr(self, '_initialized'):
            return

        self.config_dir = Path(config_dir)
        self.logger = logging.getLogger("app")
        self._config_lock = threading.RLock()

        # Asset configurations - just stores configs, no regime logic
        self.asset_configs: Dict[str, AssetIndicatorConfig] = {}
        self.global_config: Dict[str, Any] = {}

        # Runtime config overrides (for regime adaptation)
        self._runtime_overrides: Dict[str, AssetIndicatorConfig] = {}
        self._override_timestamps: Dict[str, float] = {}

        self._initialize_configuration()
        self._initialized = True

    def _initialize_configuration(self):
        """Initialize configuration system - only loads configs"""
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            self._load_global_config()
            self._load_assets_config()
            self.logger.info("Configuration manager initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize configuration manager: {e}")

    def _load_global_config(self):
        """Load global configuration from YAML"""
        global_config_path = self.config_dir / "global_config.yaml"

        if not global_config_path.exists():
            self._create_default_global_config(global_config_path)

        try:
            with open(global_config_path, 'r') as f:
                self.global_config = yaml.safe_load(f) or {}
                self.logger.info(f"Loaded global configuration from {global_config_path}")
        except Exception as e:
            self.logger.error(f"Failed to load global config: {e}")
            self.global_config = {}

    def _create_default_global_config(self, config_path: Path):
        """Create default global configuration"""
        default_config = {
            'regime_adaptation': {
                'enabled': False,  # Easy toggle
                'update_interval': '15m',
                'cache_duration': 900,
                'min_regime_duration': '1h',
                'confidence_threshold': 0.6,
                'volatility_spike_threshold': 90,
                'price_move_threshold': 0.05,
                'volume_spike_threshold': 2.0
            },
            'default_timeframes': ['15m', '30m', '1h', '4h'],
            'min_data_points': 20,
            'calculation_settings': {
                'parallel_processing': True,
                'cache_results': True,
                'max_workers': 4
            },
            'logging': {
                'level': 'INFO',
                'log_regime_changes': True,
                'log_config_updates': True
            }
        }

        try:
            with open(config_path, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False, indent=2)
            self.logger.info(f"Created default global config at {config_path}")
        except Exception as e:
            self.logger.error(f"Failed to create default global config: {e}")

    def _load_assets_config(self):
        """Load all asset configurations from single YAML file"""
        assets_config_path = self.config_dir / "assets_config.yaml"

        if not assets_config_path.exists():
            self._create_default_assets_config(assets_config_path)

        try:
            with open(assets_config_path, 'r') as f:
                config_data = yaml.safe_load(f) or {}

            assets_list = config_data.get('assets', [])

            for asset_data in assets_list:
                asset_name = asset_data.get('asset')
                if asset_name:
                    try:
                        self.asset_configs[asset_name] = AssetIndicatorConfig(**asset_data)
                        self.logger.debug(f"Loaded config for {asset_name}")
                    except Exception as e:
                        self.logger.error(f"Failed to parse config for {asset_name}: {e}")

            self.logger.info(f"Loaded configurations for {len(self.asset_configs)} assets")

        except Exception as e:
            self.logger.error(f"Failed to load assets config: {e}")
            self.asset_configs = {}

    def _create_default_assets_config(self, config_path: Path):
        """Create default assets configuration file"""
        default_assets = [
            {
                'asset': 'BTCUSDT',
                'enabled': True,
                'regime_adaptation_enabled': True,
                'ma_configs': {
                    'fast': {'period': 14, 'source': 'close'},
                    'medium': {'period': 21, 'source': 'close'},
                    'slow': {'period': 50, 'source': 'close'}
                },
                'supertrend_configs': {
                    'default': {'atr_len': 10, 'atr_mult': 3.0, 'span': 14}
                },
                'oscillator_configs': {
                    'rsi': {'period': 14, 'gaussian_weights': True}
                },
                'timeframe_overrides': {
                    '15m': [14, 21, 50],
                    '30m': [8, 13, 21],
                    '4h': [8, 13, 21]
                }
            }
        ]

        config_data = {'assets': default_assets}

        try:
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            self.logger.info(f"Created default assets config at {config_path}")
        except Exception as e:
            self.logger.error(f"Failed to create default assets config: {e}")

    def get_base_asset_config(self, asset: str) -> AssetIndicatorConfig:
        """Get base configuration for an asset (no regime adaptation)"""
        with self._config_lock:
            if asset not in self.asset_configs:
                self.logger.info(f"No config found for {asset}, creating default")
                base_config = AssetIndicatorConfig(asset=asset)
                self.asset_configs[asset] = base_config
                self._add_asset_to_config_file(asset, base_config)
                return base_config

            return self.asset_configs[asset]

    def get_effective_asset_config(self, asset: str) -> AssetIndicatorConfig:
        """Get effective configuration (base + runtime overrides)"""
        with self._config_lock:
            # Check for runtime override first
            if asset in self._runtime_overrides:
                override_time = self._override_timestamps.get(asset, 0)
                cache_duration = self.global_config.get('regime_adaptation', {}).get('cache_duration', 900)

                if time.time() - override_time < cache_duration:
                    self.logger.debug(f"Using runtime override for {asset}")
                    return self._runtime_overrides[asset]
                else:
                    # Override expired, clean it up
                    self._cleanup_expired_override(asset)

            # Return base config
            return self.get_base_asset_config(asset)

    def set_runtime_override(self, asset: str, override_config: AssetIndicatorConfig):
        """Set runtime configuration override (used by regime adaptation)"""
        with self._config_lock:
            self._runtime_overrides[asset] = override_config
            self._override_timestamps[asset] = time.time()

            if self.global_config.get('logging', {}).get('log_config_updates', True):
                self.logger.info(f"Set runtime override for {asset}")

    def clear_runtime_override(self, asset: str):
        """Clear runtime configuration override"""
        with self._config_lock:
            removed = False
            if asset in self._runtime_overrides:
                del self._runtime_overrides[asset]
                removed = True
            if asset in self._override_timestamps:
                del self._override_timestamps[asset]

            if removed and self.global_config.get('logging', {}).get('log_config_updates', True):
                self.logger.debug(f"Cleared runtime override for {asset}")

    def clear_all_runtime_overrides(self):
        """Clear all runtime configuration overrides"""
        with self._config_lock:
            cleared_count = len(self._runtime_overrides)
            self._runtime_overrides.clear()
            self._override_timestamps.clear()

            if cleared_count > 0:
                self.logger.info(f"Cleared {cleared_count} runtime overrides")

    def _cleanup_expired_override(self, asset: str):
        """Clean up expired runtime override"""
        if asset in self._runtime_overrides:
            del self._runtime_overrides[asset]
        if asset in self._override_timestamps:
            del self._override_timestamps[asset]
        self.logger.debug(f"Cleaned up expired override for {asset}")

    def _add_asset_to_config_file(self, asset: str, config: AssetIndicatorConfig):
        """Add new asset to the assets config file"""
        try:
            assets_config_path = self.config_dir / "assets_config.yaml"

            # Load existing config
            if assets_config_path.exists():
                with open(assets_config_path, 'r') as f:
                    config_data = yaml.safe_load(f) or {}
            else:
                config_data = {}

            # Ensure assets list exists
            if 'assets' not in config_data:
                config_data['assets'] = []

            # Check if asset already exists
            existing_assets = {a.get('asset') for a in config_data['assets']}
            if asset not in existing_assets:
                # Add new asset
                config_data['assets'].append(asdict(config))

                # Save back to file
                with open(assets_config_path, 'w') as f:
                    yaml.dump(config_data, f, default_flow_style=False, indent=2)

                self.logger.info(f"Added {asset} to assets config file")

        except Exception as e:
            self.logger.error(f"Failed to add {asset} to config file: {e}")

    def get_enabled_assets(self) -> List[str]:
        """Get list of enabled assets"""
        return [asset for asset, config in self.asset_configs.items() if config.enabled]

    def get_disabled_assets(self) -> List[str]:
        """Get list of disabled assets"""
        return [asset for asset, config in self.asset_configs.items() if not config.enabled]

    def get_regime_enabled_assets(self) -> List[str]:
        """Get list of assets with regime adaptation enabled"""
        return [
            asset for asset, config in self.asset_configs.items()
            if config.enabled and config.regime_adaptation_enabled
        ]

    def is_regime_adaptation_enabled(self) -> bool:
        """Check if regime adaptation is globally enabled"""
        return self.global_config.get('regime_adaptation', {}).get('enabled', False)

    def get_regime_config(self) -> Dict[str, Any]:
        """Get regime adaptation configuration"""
        return self.global_config.get('regime_adaptation', {})

    def get_calculation_settings(self) -> Dict[str, Any]:
        """Get calculation settings"""
        return self.global_config.get('calculation_settings', {})

    def get_default_timeframes(self) -> List[str]:
        """Get default timeframes"""
        return self.global_config.get('default_timeframes', ['15m', '30m', '1h'])

    def update_asset_config(self, asset: str, **updates):
        """Update base asset configuration (saves to file)"""
        with self._config_lock:
            if asset in self.asset_configs:
                config = self.asset_configs[asset]
                updated_fields = []

                for field, value in updates.items():
                    if hasattr(config, field):
                        old_value = getattr(config, field)
                        setattr(config, field, value)
                        updated_fields.append(f"{field}: {old_value} → {value}")

                if updated_fields and self.global_config.get('logging', {}).get('log_config_updates', True):
                    self.logger.info(f"Updated {asset}: {', '.join(updated_fields)}")

                # Clear runtime override for this asset since base config changed
                self.clear_runtime_override(asset)

                # Save to file
                self._save_assets_config()
            else:
                self.logger.warning(f"Asset {asset} not found in configurations")

    def update_global_config(self, **updates):
        """Update global configuration"""
        with self._config_lock:
            updated_fields = []

            for key, value in updates.items():
                if key in self.global_config:
                    old_value = self.global_config[key]
                    self.global_config[key] = value
                    updated_fields.append(f"{key}: {old_value} → {value}")
                else:
                    self.global_config[key] = value
                    updated_fields.append(f"{key}: {value} (new)")

            if updated_fields:
                self.logger.info(f"Updated global config: {', '.join(updated_fields)}")

            # Save to file
            self._save_global_config()

    def enable_asset(self, asset: str):
        """Enable an asset"""
        self.update_asset_config(asset, enabled=True)

    def disable_asset(self, asset: str):
        """Disable an asset"""
        self.update_asset_config(asset, enabled=False)
        # Clear any runtime overrides
        self.clear_runtime_override(asset)

    def enable_regime_adaptation_for_asset(self, asset: str):
        """Enable regime adaptation for specific asset"""
        self.update_asset_config(asset, regime_adaptation_enabled=True)

    def disable_regime_adaptation_for_asset(self, asset: str):
        """Disable regime adaptation for specific asset"""
        self.update_asset_config(asset, regime_adaptation_enabled=False)
        # Clear any runtime overrides
        self.clear_runtime_override(asset)

    def enable_regime_adaptation_globally(self):
        """Enable regime adaptation globally"""
        self.update_global_config(**{'regime_adaptation.enabled': True})

    def disable_regime_adaptation_globally(self):
        """Disable regime adaptation globally"""
        self.update_global_config(**{'regime_adaptation.enabled': False})
        # Clear all runtime overrides
        self.clear_all_runtime_overrides()

    def _save_assets_config(self):
        """Save all asset configurations back to file"""
        try:
            assets_config_path = self.config_dir / "assets_config.yaml"

            # Convert to list format
            assets_list = [asdict(config) for config in self.asset_configs.values()]
            config_data = {'assets': assets_list}

            with open(assets_config_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)

            self.logger.debug("Saved assets configuration to file")

        except Exception as e:
            self.logger.error(f"Failed to save assets config: {e}")

    def _save_global_config(self):
        """Save global configuration to file"""
        try:
            global_config_path = self.config_dir / "global_config.yaml"
            with open(global_config_path, 'w') as f:
                yaml.dump(self.global_config, f, default_flow_style=False, indent=2)
            self.logger.debug("Saved global configuration to file")
        except Exception as e:
            self.logger.error(f"Failed to save global config: {e}")

    def reload_configuration(self):
        """Reload configuration from files"""
        with self._config_lock:
            self.logger.info("Reloading configuration...")

            # Clear runtime overrides
            self.clear_all_runtime_overrides()

            # Reload from files
            self._load_global_config()
            self._load_assets_config()

            self.logger.info("Configuration reloaded successfully")

    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get summary of current configuration with detailed asset information"""
        enabled_assets = self.get_enabled_assets()
        disabled_assets = self.get_disabled_assets()
        regime_enabled_assets = self.get_regime_enabled_assets()

        return {
            'total_assets': len(self.asset_configs),
            'enabled_assets': enabled_assets,
            'disabled_assets': disabled_assets,
            'regime_adaptation_global': self.is_regime_adaptation_enabled(),
            'regime_enabled_assets': regime_enabled_assets,
            'runtime_overrides_count': len(self._runtime_overrides),
            'runtime_overrides_assets': list(self._runtime_overrides.keys()),
            'config_dir': str(self.config_dir),
            'last_reload': getattr(self, '_last_reload', 'Never'),
            'asset_details': {
                asset: asdict(config) for asset, config in self.asset_configs.items()
            },
            'runtime_override_details': {
                asset: asdict(override) for asset, override in self._runtime_overrides.items()
            },
            'global_config': self.global_config.copy()
        }

    def validate_configuration(self) -> Dict[str, List[str]]:
        """Validate configuration and return any issues"""
        issues = {
            'errors': [],
            'warnings': [],
            'info': []
        }

        # Check if any assets are configured
        if not self.asset_configs:
            issues['warnings'].append("No assets configured")

        # Check for assets with invalid configurations
        for asset, config in self.asset_configs.items():
            if not config.asset:
                issues['errors'].append(f"Asset {asset} has no asset name")

            # Check MA configs
            if config.ma_configs:
                for ma_type, ma_config in config.ma_configs.items():
                    if 'period' not in ma_config or ma_config['period'] <= 0:
                        issues['errors'].append(f"{asset}: Invalid MA period for {ma_type}")

            # Check SuperTrend configs
            if config.supertrend_configs:
                for st_type, st_config in config.supertrend_configs.items():
                    if 'atr_len' not in st_config or st_config['atr_len'] <= 0:
                        issues['errors'].append(f"{asset}: Invalid SuperTrend ATR length for {st_type}")
                    if 'atr_mult' not in st_config or st_config['atr_mult'] <= 0:
                        issues['errors'].append(f"{asset}: Invalid SuperTrend ATR multiplier for {st_type}")

        # Check global config
        if not self.global_config:
            issues['warnings'].append("No global configuration found")

        # Check regime adaptation settings
        if self.is_regime_adaptation_enabled():
            regime_config = self.get_regime_config()
            if regime_config.get('confidence_threshold', 0) <= 0:
                issues['warnings'].append("Regime adaptation confidence threshold should be > 0")

        return issues

    def export_configuration(self) -> Dict[str, Any]:
        """Export current configuration for backup/transfer"""
        return {
            'global_config': self.global_config.copy(),
            'asset_configs': {asset: asdict(config) for asset, config in self.asset_configs.items()},
            'export_timestamp': time.time(),
            'export_version': '1.0'
        }

    def import_configuration(self, config_data: Dict[str, Any], merge: bool = False):
        """Import configuration from backup/transfer"""
        with self._config_lock:
            try:
                if not merge:
                    # Replace everything
                    self.global_config = config_data.get('global_config', {})
                    self.asset_configs.clear()
                else:
                    # Merge with existing
                    self.global_config.update(config_data.get('global_config', {}))

                # Import asset configs
                asset_configs_data = config_data.get('asset_configs', {})
                for asset, asset_data in asset_configs_data.items():
                    self.asset_configs[asset] = AssetIndicatorConfig(**asset_data)

                # Save to files
                self._save_global_config()
                self._save_assets_config()

                # Clear runtime overrides
                self.clear_all_runtime_overrides()

                action = "merged" if merge else "imported"
                self.logger.info(f"Configuration {action} successfully")

            except Exception as e:
                self.logger.error(f"Failed to import configuration: {e}")
                raise

    def get_asset_names(self) -> List[str]:
        """Get all configured asset names"""
        return list(self.asset_configs.keys())

    def has_asset(self, asset: str) -> bool:
        """Check if asset is configured"""
        return asset in self.asset_configs

    def is_asset_enabled(self, asset: str) -> bool:
        """Check if asset is enabled"""
        if asset not in self.asset_configs:
            return False
        return self.asset_configs[asset].enabled

    def is_asset_regime_enabled(self, asset: str) -> bool:
        """Check if regime adaptation is enabled for asset"""
        if asset not in self.asset_configs:
            return False
        return (self.asset_configs[asset].enabled and
                self.asset_configs[asset].regime_adaptation_enabled and
                self.is_regime_adaptation_enabled())

    def cleanup_expired_overrides(self):
        """Clean up all expired runtime overrides"""
        with self._config_lock:
            current_time = time.time()
            cache_duration = self.get_regime_config().get('cache_duration', 900)

            expired_assets = []
            for asset, timestamp in self._override_timestamps.items():
                if current_time - timestamp >= cache_duration:
                    expired_assets.append(asset)

            for asset in expired_assets:
                self._cleanup_expired_override(asset)

            if expired_assets:
                self.logger.debug(f"Cleaned up {len(expired_assets)} expired overrides")

    def __str__(self) -> str:
        """String representation of configuration manager"""
        summary = self.get_configuration_summary()
        return (f"ConfigurationManager("
                f"assets={summary['total_assets']}, "
                f"enabled={len(summary['enabled_assets'])}, "
                f"regime_global={summary['regime_adaptation_global']}, "
                f"overrides={summary['runtime_overrides_count']})")

    def __repr__(self) -> str:
        return self.__str__()
