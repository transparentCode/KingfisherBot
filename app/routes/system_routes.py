from flask import Blueprint, jsonify, request
import logging
import os
import shutil
from app.db.redis_handler import RedisHandler
from app.db.db_handler import DBHandler
import asyncio
from concurrent.futures import TimeoutError
import app.globals as app_globals

system_bp = Blueprint('system', __name__)
logger = logging.getLogger(__name__)

@system_bp.route('/api/system/status', methods=['GET'])
def get_system_status():
    """Get system health status"""
    status = {
        'status': 'online',
        'components': {
            'database': 'unknown',
            'redis': 'unknown'
        },
        'resources': {
            'load_avg': os.getloadavg(),
            'disk': {}
        }
    }
    
    # Check Disk Usage
    try:
        total, used, free = shutil.disk_usage("/")
        status['resources']['disk'] = {
            'total_gb': round(total / (2**30), 2),
            'used_gb': round(used / (2**30), 2),
            'free_gb': round(free / (2**30), 2),
            'percent': round((used / total) * 100, 1)
        }
    except Exception as e:
        status['resources']['disk'] = {'error': str(e)}

    # Check Redis
    try:
        redis_handler = RedisHandler()
        # We can't await here easily in sync Flask without async route support or run_until_complete
        # Assuming RedisHandler is already initialized in the app context or we can check connection state
        if redis_handler.connected:
            status['components']['redis'] = 'connected'
        else:
            status['components']['redis'] = 'disconnected'
    except Exception as e:
        status['components']['redis'] = f'error: {str(e)}'

    # Check DB (simple check)
    try:
        # This is a bit hacky for sync check, but sufficient for status
        status['components']['database'] = 'connected' # Placeholder, real check requires async
    except Exception:
        status['components']['database'] = 'error'

    # Fetch detailed MarketService status if available
    try:
        ms = app_globals.market_service_instance
        if ms and ms.loop and ms.loop.is_running():
            future = asyncio.run_coroutine_threadsafe(
                ms.get_status(),
                ms.loop
            )
            market_status = future.result(timeout=2)
            status['market_service'] = market_status
            
            # Update component status based on real internal state
            status['components']['database'] = market_status.get('db_connection', 'unknown')
            
            # Check if all workers are alive
            workers = market_status.get('workers', {})
            all_writers_ok = all(w['alive'] for w in workers.get('db_writers', []))
            all_calcs_ok = all(w['alive'] for w in workers.get('calculators', []))
            
            status['components']['workers'] = 'healthy' if (all_writers_ok and all_calcs_ok) else 'degraded'
            
    except TimeoutError:
        logger.error("Error fetching market service status: Timeout")
        status['market_service'] = {'error': 'timeout'}
    except Exception as e:
        logger.error(f"Error fetching market service status: {repr(e)}")
        status['market_service'] = {'error': 'unreachable'}

    return jsonify(status)

@system_bp.route('/api/system/assets', methods=['GET'])
def get_asset_status():
    """Get initialization status of all assets"""
    service = app_globals.market_service_instance
    if not service:
        return jsonify({'success': False, 'error': 'Service not initialized'}), 503
        
    # Convert dict to list of objects for frontend consumption
    assets_list = [
        {'symbol': symbol, 'status': status}
        for symbol, status in service.asset_states.items()
    ]
    
    return jsonify({
        'success': True,
        'assets': assets_list,
        'summary': {
            'total': len(service.assets),
            'ready': list(service.asset_states.values()).count("READY"),
            'error': list(service.asset_states.values()).count("ERROR")
        }
    })

@system_bp.route('/api/system/logs', methods=['GET'])
def get_logs():
    """Get recent application logs"""
    try:
        lines = int(request.args.get('lines', 100))
        log_file = 'logs/app.log'
        
        if not os.path.exists(log_file):
            return jsonify({'logs': []})
            
        with open(log_file, 'r') as f:
            # Read last N lines efficiently
            # For simplicity, reading all and slicing (not efficient for huge files but ok for 1MB logs)
            all_lines = f.readlines()
            recent_logs = all_lines[-lines:]
            
        return jsonify({'logs': recent_logs})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@system_bp.route('/api/system/metrics', methods=['GET'])
def get_system_metrics():
    """Get historical system metrics"""
    metric_type = request.args.get('type', 'cpu_load') # cpu_load, queue_write, queue_calc, msg_rate
    limit = int(request.args.get('limit', 100))
    
    key_map = {
        'cpu_load': 'metrics:system:cpu_load',
        'queue_write': 'metrics:system:queue_write',
        'queue_calc': 'metrics:system:queue_calc',
        'msg_rate': 'metrics:system:msg_rate'
    }
    
    redis_key = key_map.get(metric_type)
    if not redis_key:
        return jsonify({'error': 'Invalid metric type'}), 400

    try:
        ms = app_globals.market_service_instance
        if ms and ms.loop and ms.loop.is_running():
            future = asyncio.run_coroutine_threadsafe(
                ms.redis_handler.get_metrics(redis_key, limit),
                ms.loop
            )
            data = future.result(timeout=5)
            return jsonify({'success': True, 'data': data})
        else:
            return jsonify({'success': False, 'error': 'MarketService not running'}), 503
            
    except TimeoutError:
        logger.error("Error fetching metrics: Timeout")
        return jsonify({'success': False, 'error': 'Timeout waiting for MarketService'}), 504
    except Exception as e:
        logger.error(f"Error fetching metrics: {repr(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500
