import asyncio
import os
import logging.config
from threading import Thread

from dotenv import load_dotenv
from app import create_app
from app.services import IndicatorRegisters
import atexit

from app.services.market_service import MarketService

market_service = None

def run_market_service():
    global market_service
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    market_service = MarketService()
    loop.run_until_complete(market_service.start())
    loop.run_forever()

if __name__ == '__main__':
    load_dotenv()

    app_port = int(os.getenv('APP_PORT', 8080))

    logging.config.fileConfig('logging.conf', disable_existing_loggers=False)
    app = create_app()

    app.logger.info('Bot starting.....')

    IndicatorRegisters.register_indicators()

    market_thread = Thread(target=run_market_service, daemon=True)
    market_thread.start()

    @atexit.register
    def cleanup():
        if market_service:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(market_service.stop())

    app.run(
        debug=True,
        host='0.0.0.0', # for docker and podman use 0.0.0.0 (loopback address)
        port=app_port
    )
