from flask_socketio import SocketIO, emit, join_room, leave_room
from flask import request
import asyncio
import logging
import json
import redis.asyncio as redis
from app.models.RedisConfig import RedisConfig
from datetime import datetime
from threading import Thread

logger = logging.getLogger(__name__)


class LiveDataManager:
    def __init__(self, socketio):
        self.socketio = socketio
        self.active_subscriptions = {}  # {room: {symbol, interval, last_update}}
        self.redis_client = None
        self.loop = asyncio.new_event_loop()
        self.thread = None
        self.redis_config = RedisConfig()

    def start(self):
        """Start the dedicated event loop in a background thread"""
        def run_loop():
            asyncio.set_event_loop(self.loop)
            self.loop.create_task(self.broadcast_updates_loop())
            self.loop.run_forever()
        
        self.thread = Thread(target=run_loop, daemon=True)
        self.thread.start()

    def run_async(self, coro):
        """Submit a coroutine to the dedicated event loop"""
        if self.loop.is_running():
            return asyncio.run_coroutine_threadsafe(coro, self.loop)
        else:
            logger.error("LiveDataManager loop is not running")

    async def ensure_redis(self):
        if not self.redis_client:
            try:
                self.redis_client = await redis.from_url(
                    self.redis_config.redis_url,
                    encoding="utf-8",
                    decode_responses=True
                )
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")

    async def subscribe_to_symbol(self, session_id, symbol, interval):
        """Subscribe client to symbol updates"""
        await self.ensure_redis()
        room = f"{symbol}_{interval}"

        if room not in self.active_subscriptions:
            self.active_subscriptions[room] = {
                'symbol': symbol,
                'interval': interval,
                'clients': set(),
                'last_update': None
            }

        self.active_subscriptions[room]['clients'].add(session_id)
        join_room(room)

        # Send initial data
        await self.send_latest_candle(room)

        logger.info(f"Client {session_id} subscribed to {room}")

    async def unsubscribe_from_symbol(self, session_id, symbol, interval):
        """Unsubscribe client from symbol updates"""
        room = f"{symbol}_{interval}"

        if room in self.active_subscriptions:
            self.active_subscriptions[room]['clients'].discard(session_id)
            leave_room(room)

            # Remove room if no more clients
            if not self.active_subscriptions[room]['clients']:
                del self.active_subscriptions[room]

        logger.info(f"Client {session_id} unsubscribed from {room}")

    async def send_latest_candle(self, room):
        """Send latest candle data to room"""
        if room not in self.active_subscriptions:
            return

        subscription = self.active_subscriptions[room]
        symbol = subscription['symbol']
        interval = subscription['interval']

        try:
            await self.ensure_redis()
            if not self.redis_client:
                return

            # Get latest candle from Redis
            # Key format matches RedisHandler: market_data:{symbol}:{timeframe}
            key = f"market_data:{symbol}:{interval}"
            data = await self.redis_client.get(key)

            if data:
                latest_candle = json.loads(data)
                
                # Check if this is a new update
                # Timestamp in Redis is usually int (ms) or ISO string depending on source
                # WebSocketListenerService saves it as int (ms) from Binance 't'
                candle_ts = latest_candle.get('timestamp')
                
                if subscription['last_update'] != candle_ts:
                    subscription['last_update'] = candle_ts
                    
                    # Convert timestamp to ISO for frontend if needed, or keep as is
                    # Frontend usually expects ISO string
                    if isinstance(candle_ts, (int, float)):
                        ts_iso = datetime.fromtimestamp(candle_ts / 1000).isoformat()
                    else:
                        ts_iso = str(candle_ts)

                    # Prepare payload
                    is_closed = latest_candle.get('is_closed', False)
                    payload = {
                        'symbol': symbol,
                        'interval': interval,
                        'timestamp': ts_iso,
                        'open': float(latest_candle['open']),
                        'high': float(latest_candle['high']),
                        'low': float(latest_candle['low']),
                        'close': float(latest_candle['close']),
                        'volume': float(latest_candle['volume']),
                        'is_closed': is_closed
                    }

                    # Emit to room
                    self.socketio.emit('price_update', payload, room=room)
                    
                    # Emit specific bar closed event if applicable
                    if is_closed:
                        self.socketio.emit('bar_closed', payload, room=room)

        except Exception as e:
            logger.error(f"Error sending latest candle for {room}: {str(e)}")

    async def broadcast_updates_loop(self):
        """Periodically broadcast updates to all subscribed rooms"""
        await self.ensure_redis()
        while True:
            try:
                for room in list(self.active_subscriptions.keys()):
                    await self.send_latest_candle(room)
                await asyncio.sleep(1)  # Update every 1 second (Redis is fast)
            except Exception as e:
                logger.error(f"Error in background updates: {e}")
                await asyncio.sleep(5)


def setup_websocket_routes(socketio):
    live_manager = LiveDataManager(socketio)
    live_manager.start()

    @socketio.on('connect')
    def handle_connect():
        logger.info(f"Client connected: {request.sid}")
        emit('connection_status', {'status': 'connected'})

    @socketio.on('disconnect')
    def handle_disconnect():
        logger.info(f"Client disconnected: {request.sid}")

    @socketio.on('subscribe_symbol')
    def handle_subscribe_symbol(data):
        symbol = data.get('symbol', 'BTCUSDT')
        interval = data.get('interval', '1h')
        live_manager.run_async(live_manager.subscribe_to_symbol(request.sid, symbol, interval))

    @socketio.on('unsubscribe_symbol')
    def handle_unsubscribe_symbol(data):
        symbol = data.get('symbol', 'BTCUSDT')
        interval = data.get('interval', '1h')
        live_manager.run_async(live_manager.unsubscribe_from_symbol(request.sid, symbol, interval))

    def log_streamer():
        """
        Background task to tail log files and emit new lines to clients.
        """
        import time
        import os
        import re
        
        # Regex to strip ANSI color codes
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

        # Construct absolute path to ensure we find the file regardless of CWD
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        
        # List of files to tail. 
        log_filenames = ['app.log'] 
        
        open_files = []
        
        # Initialize file handles
        for filename in log_filenames:
            path = os.path.join(project_root, 'logs', filename)
            # Wait for file to exist
            while not os.path.exists(path):
                time.sleep(1)
            
            f = open(path, 'r')
            # Read the last 2000 bytes to give some context on startup
            try:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - 5000), os.SEEK_SET) # Last ~5KB
                # Discard the first partial line
                f.readline()
            except Exception:
                f.seek(0, os.SEEK_END)
                
            open_files.append(f)
        
        while True:
            data_read = False
            for f in open_files:
                line = f.readline()
                if line:
                    data_read = True
                    # Strip ANSI codes
                    clean_line = ansi_escape.sub('', line)
                    
                    # Parse the line if possible to structure it
                    # Format: 2025-01-01 12:00:00 - app.module - INFO - Message
                    try:
                        parts = clean_line.split(' - ')
                        if len(parts) >= 4:
                            timestamp = parts[0]
                            name = parts[1]
                            level = parts[2]
                            message = ' - '.join(parts[3:]).strip()
                            
                            log_entry = {
                                'timestamp': timestamp,
                                'name': name,
                                'level': level,
                                'message': message
                            }
                            socketio.emit('log_message', log_entry)
                        else:
                            # Fallback for unstructured lines (stack traces etc)
                            # Only emit if it's not an empty line
                            if clean_line.strip():
                                socketio.emit('log_message', {
                                    'timestamp': datetime.now().isoformat(),
                                    'name': 'system',
                                    'level': 'INFO',
                                    'message': clean_line.strip()
                                })
                    except Exception as e:
                        print(f"Error parsing log line: {e}")
            
            if not data_read:
                time.sleep(0.1)

    socketio.start_background_task(log_streamer)
    
    return live_manager
