from flask import Flask
from app.routes.backtest_routes import backtest_bp
from app.routes.fyers_auth_routes import fyers_auth_bp
from app.routes.indicator_routes import indicator_bp
from app.routes.monitoring_routes import monitoring_bp
from app.routes.asset_info_routes import asset_info_bp
from config.configs import DevelopmentConfig


def create_app():
    flask_app = Flask(__name__)
    flask_app.config.from_object(DevelopmentConfig)

    flask_app.register_blueprint(backtest_bp)
    flask_app.register_blueprint(indicator_bp)
    flask_app.register_blueprint(fyers_auth_bp)
    app.register_blueprint(monitoring_bp)
    app.register_blueprint(asset_info_bp)

    return flask_app
