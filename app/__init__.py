from flask import Flask
from app.routes.backtest_routes import backtest_bp
from app.routes.indicator_routes import indicator_bp
from config.configs import DevelopmentConfig


def create_app():
    flask_app = Flask(__name__)
    flask_app.config.from_object(DevelopmentConfig)

    flask_app.register_blueprint(backtest_bp)
    flask_app.register_blueprint(indicator_bp)

    return flask_app
