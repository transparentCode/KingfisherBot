from flask import Flask, render_template
from flask_socketio import SocketIO

from app.routes.backtest_routes import backtest_bp
from app.routes.indicator_routes import indicator_bp
from app.routes.market_routes import market_bp
from config.configs import DevelopmentConfig
from app.routes.websocket_routes import setup_websocket_routes


def create_app():
    flask_app = Flask(__name__)
    flask_app.config.from_object(DevelopmentConfig)

    socketio = SocketIO(flask_app, cors_allowed_origins="*", async_mode="threading")

    flask_app.register_blueprint(backtest_bp)
    flask_app.register_blueprint(indicator_bp)
    flask_app.register_blueprint(market_bp)

    live_manager = setup_websocket_routes(socketio)

    @flask_app.route('/')
    def dashboard():
        return render_template('dashboard.html')

    return flask_app, socketio
