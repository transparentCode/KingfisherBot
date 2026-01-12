from flask import Blueprint, jsonify, request
import logging
import app.globals as app_globals
from app.db.redis_handler import RedisHandler
import asyncio
from concurrent.futures import TimeoutError
from app.utils.system_utils import (
    get_disk_usage, 
    get_system_load, 
    read_log_tail, 
    get_market_service_status
)

system_bp = Blueprint('system', __name__)
logger = logging.getLogger(__name__)

@system_bp.route('/api/system/status', methods=['GET'])
def get_system_status():
    """Get system health status"""
    # 1. Base Structure
    status = {
        'status': 'online',
        'components': {
            'database': 'unknown',
            'redis': 'unknown'
        },
        'resources': {
            'load_avg': get_system_load(),
            'disk': get_disk_usage()
        }
    }
    
    # 2. Redis Check (Simple connectivity check)
    try:
        # Note: Ideally this should use an async check or the one from MarketService
        # For lightweight sync check we assume if object exists it's partly ok, 
        # but real status comes from MarketService below.
        redis_handler = RedisHandler()
        status['components']['redis'] = 'connected' if redis_handler.connected else 'disconnected'
    except Exception as e:
        status['components']['redis'] = f'error: {str(e)}'

    # 3. MarketService Integration (The Truth Source)
    ms = app_globals.market_service_instance
    market_status = get_market_service_status(ms)
    
    status['market_service'] = market_status
    
    # 4. Consolidate Component Status
    if 'error' not in market_status:
        # Use the deep insights from MarketService to update components
        status['components']['database'] = market_status.get('db_connection', 'unknown')
        
        # Check workers health
        workers = market_status.get('workers', {})
        all_writers = all(w.get('alive') for w in workers.get('db_writers', []))
        all_calcs = all(w.get('alive') for w in workers.get('calculators', []))
        
        status['components']['workers'] = 'healthy' if (all_writers and all_calcs) else 'degraded'
    else:
        status['components']['database'] = 'error'

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
    
    # Calculate summary stats based on frontend expectations (healthy/degraded/down)
    statuses = list(service.asset_states.values())
    healthy_count = statuses.count("READY")
    down_count = statuses.count("ERROR")
    # All other states are considered "working on it" or degraded
    degraded_count = len(statuses) - healthy_count - down_count
    
    return jsonify({
        'success': True,
        'assets': assets_list,
        'summary': {
            'total': len(service.assets),
            'healthy': healthy_count,
            'degraded': degraded_count,
            'down': down_count
        }
    })

@system_bp.route('/api/system/logs', methods=['GET'])
def get_logs():
    """Get recent application logs"""
    try:
        lines = int(request.args.get('lines', 100))
        log_file = 'logs/app.log'
        
        recent_logs = read_log_tail(log_file, lines)
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
            
            # Format data for TVLC / Frontend (Normalize to {time: seconds, value: val})
            formatted_data = []
            for item in data:
                if 'ts' in item and 'val' in item:
                    formatted_data.append({
                        # Convert ms timestamp to seconds for standard charting
                        'time': int(item['ts'] / 1000),
                        'value': item['val']
                    })
            
            return jsonify({'success': True, 'data': formatted_data})
        else:
            return jsonify({'success': False, 'error': 'MarketService not running'}), 503
            
    except TimeoutError:
        logger.error("Error fetching metrics: Timeout")
        return jsonify({'success': False, 'error': 'Timeout waiting for MarketService'}), 504
    except Exception as e:
        logger.error(f"Error fetching metrics: {repr(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500
