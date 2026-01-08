from flask import Blueprint, jsonify, request
import logging
import json
import asyncio
from app.db.redis_handler import RedisHandler
from config.asset_indicator_config import ConfigurationManager

signal_bp = Blueprint('signal', __name__)
logger = logging.getLogger(__name__)

@signal_bp.route('/api/signals/latest/<symbol>', methods=['GET'])
def get_latest_signals(symbol):
    """Get latest aggregated signals for a symbol from Redis"""
    try:
        async def fetch_signals():
            redis_handler = RedisHandler()
            await redis_handler.initialize()
            
            data = await redis_handler.redis_client.get(f"signals:{symbol}:latest")
            if data:
                return json.loads(data)
            return []

        signals = asyncio.run(fetch_signals())
        return jsonify({'success': True, 'signals': signals})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@signal_bp.route('/api/signals/all', methods=['GET'])
def get_all_signals():
    """Get latest signals for all configured assets"""
    try:
        config_manager = ConfigurationManager()
        assets = config_manager.get_enabled_assets()
        
        async def fetch_all():
            redis_handler = RedisHandler()
            await redis_handler.initialize()
            
            results = {}
            for asset in assets:
                data = await redis_handler.redis_client.get(f"signals:{asset}:latest")
                if data:
                    results[asset] = json.loads(data)
                else:
                    results[asset] = []
            return results

        all_signals = asyncio.run(fetch_all())
        return jsonify({'success': True, 'signals': all_signals})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
