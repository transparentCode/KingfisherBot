import yaml
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Simple YAML config loader with validation"""
    
    @staticmethod
    def load(config_path: str) -> dict:
        """
        Load YAML configuration file
        
        Args:
            config_path: Path to config file (relative to project root)
        
        Returns:
            Parsed configuration dictionary
        """
        try:
            # Convert to absolute path if relative
            if not config_path.startswith('/'):
                # Assume relative to project root
                config_path = Path(__file__).parent.parent.parent / config_path
            else:
                config_path = Path(config_path)
            
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
            
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            logger.info(f"Loaded configuration from {config_path}")
            return config
        
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
    
    @staticmethod
    def validate_sr_config(config: dict) -> bool:
        """
        Validate S/R configuration structure
        
        Returns:
            True if valid, raises ValueError otherwise
        """
        required_sections = [
            'general',
            'timeframes',
            'pivot_detection',
            'zone_formation',
            'validation',
            'confluence',
            'fibonacci',
            'ema_confluence',
            'round_numbers',
            'priority_scoring',
            'dynamic_updates',
            'merging',
            'output'
        ]
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")
        
        return True