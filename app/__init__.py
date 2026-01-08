from flask import Flask, render_template
from flask_socketio import SocketIO

from app.routes.backtest_routes import backtest_bp
from app.routes.indicator_routes import indicator_bp
from app.routes.market_routes import market_bp
from app.routes.system_routes import system_bp
from app.routes.signal_routes import signal_bp
from app.routes.config_routes import config_bp
from app.routes.auth_routes import auth_bp
from app.routes.strategy_routes import strategy_bp
from app.routes.standard_indicator_routes import standard_indicator_bp
from config.configs import DevelopmentConfig
from app.routes.websocket_routes import setup_websocket_routes


def create_app():
    flask_app = Flask(__name__)
    flask_app.config.from_object(DevelopmentConfig)

    socketio = SocketIO(flask_app, cors_allowed_origins="*", async_mode="threading")

    flask_app.register_blueprint(backtest_bp)
    flask_app.register_blueprint(indicator_bp)
    flask_app.register_blueprint(market_bp)
    flask_app.register_blueprint(system_bp)
    flask_app.register_blueprint(signal_bp)
    flask_app.register_blueprint(config_bp)
    flask_app.register_blueprint(auth_bp)
    flask_app.register_blueprint(strategy_bp)
    flask_app.register_blueprint(standard_indicator_bp)

    live_manager = setup_websocket_routes(socketio)

    @flask_app.route('/')
    def dashboard():
        return render_template('dashboard.html')

    return flask_app, socketio
