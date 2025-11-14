import json

import plotly
from flask import Blueprint, request, jsonify, render_template
import inspect
from datetime import datetime, timedelta
import pandas as pd
import logging
import asyncio

from app.exchange import BinanceConnector
from app.services.indicator_registers import IndicatorRegistry
from app.db.db_handler import DBHandler

indicator_bp = Blueprint('indicator', __name__)

logger = logging.getLogger(__name__)

indicator_register = IndicatorRegistry()

@indicator_bp.route('/api/indicators/available', methods=['GET'])
def get_available_indicators_for_ui():
    """Get all indicators formatted for UI with parameter schemas"""
    try:
        IndicatorRegistry.register_indicator()  # Ensure indicators are registered
        registry = IndicatorRegistry()
        categories = registry.get_all_indicators_for_ui()

        return jsonify({
            'success': True,
            'categories': categories
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@indicator_bp.route('/api/indicators/<indicator_id>/schema', methods=['GET'])
def get_indicator_schema_for_ui(indicator_id):
    """Get detailed schema for specific indicator"""
    try:
        registry = IndicatorRegistry()
        indicator_info = registry.get_indicator_for_ui(indicator_id)

        if not indicator_info:
            return jsonify({
                'success': False,
                'error': f'Indicator {indicator_id} not found'
            })

        return jsonify({
            'success': True,
            'indicator': indicator_info
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@indicator_bp.route('/api/indicator/<indicator_id>/calculate', methods=['GET'])
def calculate_indicator(indicator_id):
    """Calculate and return indicator data using database candles"""
    try:
        # Get request parameters
        symbol = request.args.get('symbol', 'BTCUSDT')
        interval = request.args.get('interval', '1h')
        limit = int(request.args.get('limit', 500))  # Use limit instead of lookback_days

        # Get indicator registry
        IndicatorRegistry.register_indicator()
        registry = IndicatorRegistry()

        # Check if indicator exists
        indicator_info = registry.get_indicator_for_ui(indicator_id)
        if not indicator_info:
            return jsonify({
                'success': False,
                'error': f'Indicator {indicator_id} not found'
            }), 404

        # Fetch candles from database using sync wrapper
        async def fetch_candles():
            db_handler = DBHandler()
            await db_handler.initialize()

            candles = await db_handler.read_candles(
                symbol=symbol,
                interval='1m',
                limit=limit * 10
            )

            await db_handler.close()
            return candles

        candles = asyncio.run(fetch_candles())

        if not candles:
            return jsonify({
                'success': False,
                'error': f'No data found for {symbol}'
            }), 404

        # Convert to DataFrame
        df = pd.DataFrame([dict(row) for row in candles])
        df = df.set_index('bucket')
        df.index = pd.to_datetime(df.index)

        # Convert to numeric types
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna().sort_index()

        # Resample to requested timeframe if needed
        if interval != '1m':
            df = resample_ohlcv_data(df, interval)

        # IMPORTANT: Limit BEFORE calculation to ensure alignment
        df = df.tail(limit)

        if len(df) < 50:
            return jsonify({
                'success': False,
                'error': f'Insufficient data for {symbol}. Got {len(df)} candles, need at least 50'
            }), 400

        # Get indicator parameters from query string
        params = {}
        for key, value in request.args.items():
            if key not in ['symbol', 'interval', 'lookback_days']:
                try:
                    if '.' in value:
                        params[key] = float(value)
                    else:
                        params[key] = int(value)
                except ValueError:
                    params[key] = value

        # Create indicator instance (FIX: Use create_indicator_instance instead of get_indicator_for_ui)
        indicator = registry.create_indicator_instance(indicator_id, **params)

        # Calculate indicator values
        indicator_data = indicator.calculate(df)

        logger.info(f"Indicator data calculated successfully for {indicator_id}: {df.shape}")

        # Generate plot data
        plot_data = indicator._get_plot_trace(indicator_data)

        return jsonify({
            'success': True,
            'indicator_id': indicator_id,
            'indicator_name': indicator_info['display_name'],
            'symbol': symbol,
            'interval': interval,
            'data_points': len(df),
            'plot_data': plot_data,
            'parameters': params
        })

    except Exception as e:
        logger.error(f"Error calculating indicator {indicator_id}: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e),
            'indicator_id': indicator_id
        }), 500

def resample_ohlcv_data(df, timeframe):
    """Resample OHLCV data to a different timeframe"""
    try:
        # Ensure the index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Define resampling rules for different timeframes
        timeframe_map = {
            '1m': '1T',  # 1 minute
            '5m': '5T',  # 5 minutes
            '15m': '15T',  # 15 minutes
            '30m': '30T',  # 30 minutes
            '1h': '1H',  # 1 hour
            '4h': '4H',  # 4 hours
            '1d': '1D',  # 1 day
            '1w': '1W',  # 1 week
        }

        if timeframe not in timeframe_map:
            logger.warning(f"Unsupported timeframe: {timeframe}, returning original data")
            return df

        freq = timeframe_map[timeframe]

        # Resample OHLCV data
        resampled = df.resample(freq).agg({
            'open': 'first',  # First open price in the period
            'high': 'max',  # Highest price in the period
            'low': 'min',  # Lowest price in the period
            'close': 'last',  # Last close price in the period
            'volume': 'sum'  # Sum of volume in the period
        }).dropna()

        return resampled

    except Exception as e:
        logger.error(f"Error resampling data to {timeframe}: {e}")
        return df

@indicator_bp.route('/view', methods=['GET'])
def view_backtest():
    return render_template('indicators.html')