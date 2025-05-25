import logging
import os


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'optimus'
    DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///trading_bot.db'

class DevelopmentConfig(Config):
    DEBUG = True
    logging.basicConfig(level=logging.INFO)

class ProductionConfig(Config):
    DEBUG = False
    logging.basicConfig(level=logging.INFO)