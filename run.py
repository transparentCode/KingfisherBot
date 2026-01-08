import asyncio
import os
import logging.config
from threading import Thread

from dotenv import load_dotenv
from app import create_app
from app.services import IndicatorRegistry
from app.services.strategy_registry import StrategyRegistry
import atexit

from app.services.market_service import MarketService
import app.globals as app_globals

market_service = None

def run_market_service():
    global market_service
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    market_service = MarketService()
    app_globals.market_service_instance = market_service
    loop.run_until_complete(market_service.start())
    loop.run_forever()

if __name__ == '__main__':
    load_dotenv()

    app_port = int(os.getenv('APP_PORT', 8080))

    # Ensure log directory exists before configuring logging
    log_dir = os.path.join(os.getcwd(), 'logs')
    os.makedirs(log_dir, exist_ok=True)

    logging_conf_path = os.path.join(os.getcwd(), 'logging.conf')

    logging.config.fileConfig(logging_conf_path, disable_existing_loggers=False)
    app, socketio = create_app()

    app.logger.info('Bot starting.....')

    IndicatorRegistry.register_default_indicators()
    StrategyRegistry.register_default_strategies()

    market_thread = Thread(target=run_market_service, daemon=True)
    market_thread.start()

    @atexit.register
    def cleanup():
        if market_service:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(market_service.stop())

    socketio.run(
        app,
        debug=False,
        host='0.0.0.0', # for docker and podman use 0.0.0.0 (loopback address)
        port=app_port,
        allow_unsafe_werkzeug=True
    )
