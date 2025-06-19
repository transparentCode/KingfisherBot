import os
import time
import psutil
import json
from datetime import datetime
from flask import Blueprint, jsonify, Response, current_app, stream_with_context

monitoring_bp = Blueprint('monitoring', __name__)


def is_running_in_container():
    """Detect if running inside a container"""
    # Method 1: Check for .dockerenv file
    if os.path.exists('/.dockerenv'):
        return True

    # Method 2: Check cgroup
    try:
        with open('/proc/1/cgroup', 'r') as f:
            if any('docker' in line or 'kubepods' in line for line in f):
                return True
    except (IOError, FileNotFoundError):
        pass

    return False


def get_monitoring_system():
    """Helper to get monitoring system from current app context"""
    market_service = current_app.market_service
    return market_service.monitor


def get_system_metrics():
    """Get system resource metrics"""
    metrics = {
        'cpu': {
            'percent': psutil.cpu_percent(interval=0.1),
            'count': psutil.cpu_count(logical=True)
        },
        'memory': {
            'total': psutil.virtual_memory().total,
            'available': psutil.virtual_memory().available,
            'used': psutil.virtual_memory().used,
            'percent': psutil.virtual_memory().percent
        },
        'disk': {
            'total': psutil.disk_usage('/').total,
            'used': psutil.disk_usage('/').used,
            'free': psutil.disk_usage('/').free,
            'percent': psutil.disk_usage('/').percent
        }
    }

    # Add container-specific metrics if running in container
    if is_running_in_container():
        container_metrics = get_container_metrics()
        if container_metrics:
            metrics['container'] = container_metrics

    return metrics


def get_process_metrics():
    """Get current process resource metrics"""
    try:
        process = psutil.Process()
        return {
            'cpu_percent': process.cpu_percent(interval=0.1),
            'memory_percent': process.memory_percent(),
            'threads': process.num_threads(),
            'connections': len(process.connections())
        }
    except Exception as e:
        return {'error': str(e)}


def get_container_metrics():
    """Get Docker container cgroup metrics"""
    if not is_running_in_container():
        return {}

    metrics = {}

    # Try cgroups v1 path
    try:
        with open('/sys/fs/cgroup/memory/memory.usage_in_bytes', 'r') as f:
            metrics['memory_usage'] = int(f.read().strip())

        with open('/sys/fs/cgroup/memory/memory.limit_in_bytes', 'r') as f:
            metrics['memory_limit'] = int(f.read().strip())
            if metrics['memory_limit'] > 0:
                metrics['memory_percent'] = (metrics['memory_usage'] / metrics['memory_limit']) * 100
    except FileNotFoundError:
        # Try cgroups v2 path
        try:
            with open('/sys/fs/cgroup/memory.current', 'r') as f:
                metrics['memory_usage'] = int(f.read().strip())
            with open('/sys/fs/cgroup/memory.max', 'r') as f:
                max_value = f.read().strip()
                # 'max' means unlimited
                if max_value != 'max':
                    metrics['memory_limit'] = int(max_value)
                    metrics['memory_percent'] = (metrics['memory_usage'] / metrics['memory_limit']) * 100
        except FileNotFoundError:
            pass

    return metrics


@monitoring_bp.route('/api/monitoring/status', methods=['GET'])
def get_system_status():
    """Get current system status"""
    monitoring = get_monitoring_system()
    system = monitoring.system

    status = {
        'timestamp': datetime.now().isoformat(),
        'assets': {
            asset: {
                'connected': asset in monitoring.connected_assets,
                'message_count': monitoring.message_counts.get(asset, 0),
                'message_rate': monitoring.message_rate.get(asset, 0.0)
            } for asset in system.assets
        },
        'queues': {
            'write_queue': system.write_queue.qsize(),
            'calc_queue': system.calc_queue.qsize()
        },
        'thresholds': {
            'write_queue': monitoring.config.write_queue_threshold,
            'calc_queue': monitoring.config.calc_queue_threshold
        }
    }

    return jsonify(status)


@monitoring_bp.route('/api/monitoring/resources', methods=['GET'])
def get_resources():
    """Get system and process resource metrics"""
    return jsonify({
        'timestamp': datetime.now().isoformat(),
        'system': get_system_metrics(),
        'process': get_process_metrics(),
        'container_detected': is_running_in_container()
    })


@monitoring_bp.route('/api/monitoring/stream', methods=['GET'])
def stream_metrics():
    """Stream metrics using Server-Sent Events (SSE)"""
    monitoring = get_monitoring_system()
    system = monitoring.system

    def generate():
        yield "data: {}\n\n"  # Initial event to establish connection

        while True:
            status = {
                'timestamp': datetime.now().isoformat(),
                'assets': {
                    asset: {
                        'connected': asset in monitoring.connected_assets,
                        'message_count': monitoring.message_counts.get(asset, 0),
                        'message_rate': monitoring.message_rate.get(asset, 0.0)
                    } for asset in system.assets
                },
                'queues': {
                    'write_queue': system.write_queue.qsize(),
                    'calc_queue': system.calc_queue.qsize()
                },
                'resources': {
                    'system': get_system_metrics(),
                    'process': get_process_metrics()
                }
            }

            yield f"data: {json.dumps(status)}\n\n"
            time.sleep(1)

    return Response(
        stream_with_context(generate()),
        content_type='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive'
        }
    )