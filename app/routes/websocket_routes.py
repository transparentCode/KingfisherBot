from flask_socketio import SocketIO, emit, join_room, leave_room
from flask import request
import asyncio
import logging
from app.db.db_handler import DBHandler
from datetime import datetime

logger = logging.getLogger(__name__)


class LiveDataManager:
    def __init__(self, socketio):
        self.socketio = socketio
        self.active_subscriptions = {}  # {room: {symbol, interval, last_update}}
        self.db_handler = None

    async def initialize(self):
        if not self.db_handler:
            self.db_handler = DBHandler()
            await self.db_handler.initialize()

    async def subscribe_to_symbol(self, session_id, symbol, interval):
        """Subscribe client to symbol updates"""
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
            await self.initialize()

            # Get latest candle
            candles = await self.db_handler.read_candles(
                symbol=symbol,
                interval=interval,
                limit=1
            )

            if candles:
                latest_candle = candles[0]

                # Check if this is a new update
                candle_time = latest_candle['timestamp']
                if subscription['last_update'] != candle_time:
                    subscription['last_update'] = candle_time

                    # Emit to room
                    self.socketio.emit('price_update', {
                        'symbol': symbol,
                        'interval': interval,
                        'timestamp': candle_time.isoformat(),
                        'open': float(latest_candle['open']),
                        'high': float(latest_candle['high']),
                        'low': float(latest_candle['low']),
                        'close': float(latest_candle['close']),
                        'volume': float(latest_candle['volume'])
                    }, room=room)

        except Exception as e:
            logger.error(f"Error sending latest candle for {room}: {str(e)}")

    async def broadcast_updates(self):
        """Periodically broadcast updates to all subscribed rooms"""
        for room in list(self.active_subscriptions.keys()):
            await self.send_latest_candle(room)


def setup_websocket_routes(socketio):
    live_manager = LiveDataManager(socketio)

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

        # Use socketio.start_background_task for async operations
        socketio.start_background_task(
            lambda: asyncio.run(live_manager.subscribe_to_symbol(request.sid, symbol, interval))
        )

    @socketio.on('unsubscribe_symbol')
    def handle_unsubscribe_symbol(data):
        symbol = data.get('symbol', 'BTCUSDT')
        interval = data.get('interval', '1h')

        socketio.start_background_task(
            lambda: asyncio.run(live_manager.unsubscribe_from_symbol(request.sid, symbol, interval))
        )

    # Fixed background task with proper event loop
    def background_updates():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def update_loop():
            while True:
                try:
                    await live_manager.broadcast_updates()
                    await asyncio.sleep(30)  # Update every 30 seconds
                except Exception as e:
                    logger.error(f"Error in background updates: {e}")
                    await asyncio.sleep(10)  # Wait before retrying

        try:
            loop.run_until_complete(update_loop())
        except KeyboardInterrupt:
            pass
        finally:
            loop.close()

    socketio.start_background_task(background_updates)
    return live_manager
