from flask import Blueprint, jsonify, request
import logging
from config.asset_indicator_config import ConfigurationManager

config_bp = Blueprint('config', __name__)
logger = logging.getLogger(__name__)

@config_bp.route('/api/config', methods=['GET'])
def get_config():
    """Get current configuration"""
    try:
        manager = ConfigurationManager()
        summary = manager.get_configuration_summary()
        return jsonify({'success': True, 'config': summary})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@config_bp.route('/api/config/assets', methods=['GET'])
def get_assets():
    """Get list of configured assets"""
    try:
        manager = ConfigurationManager()
        assets = manager.get_enabled_assets()
        return jsonify({'success': True, 'assets': assets})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@config_bp.route('/api/config/reload', methods=['POST'])
def reload_config():
    """Reload configuration from disk"""
    try:
        manager = ConfigurationManager()
        manager.reload_configuration()
        return jsonify({'success': True, 'message': 'Configuration reloaded successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
