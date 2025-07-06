from flask import Blueprint, request, jsonify
from app.database.db_handler import DBHandler
import pandas as pd
from datetime import datetime, timedelta
import logging
import asyncio

market_bp = Blueprint('market', __name__)
logger = logging.getLogger(__name__)


@market_bp.route('/api/search-symbols', methods=['GET'])
def search_symbols():
    try:
        query = request.args.get('q', '').upper()

        # Mock symbol data - replace with your actual symbol list
        all_symbols = [
            {'symbol': 'BTCUSDT', 'name': 'Bitcoin'},
            {'symbol': 'ETHUSDT', 'name': 'Ethereum'},
            {'symbol': 'BNBUSDT', 'name': 'Binance Coin'},
            {'symbol': 'ADAUSDT', 'name': 'Cardano'},
            {'symbol': 'SOLUSDT', 'name': 'Solana'},
            {'symbol': 'DOTUSDT', 'name': 'Polkadot'},
            {'symbol': 'AVAXUSDT', 'name': 'Avalanche'},
            {'symbol': 'MATICUSDT', 'name': 'Polygon'},
            # Add more symbols as needed
        ]

        # Filter symbols based on query
        filtered_symbols = [
            symbol for symbol in all_symbols
            if query in symbol['symbol'] or query in (symbol['name'] or '').upper()
        ]

        return jsonify({
            'success': True,
            'symbols': filtered_symbols[:20]  # Limit results
        })

    except Exception as e:
        logger.error(f"Error searching symbols: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })


@market_bp.route('/api/candle-data/<symbol>')
def get_candle_data(symbol):
    """Get candlestick data for a symbol"""
    logger.info(f"📊 Fetching candle data for {symbol}")
    try:
        timeframe = request.args.get('timeframe', '1h')
        limit = int(request.args.get('limit', 500))

        logger.info(f"Parameters: timeframe={timeframe}, limit={limit}")

        async def fetch_candle_data():
            db_handler = DBHandler()
            await db_handler.initialize()
            logger.info("✅ DB handler initialized")

            # Use the working read_candles method
            candles = await db_handler.read_candles(
                symbol=symbol,
                interval='1m',  # Always get 1m data first
                limit=limit * 20  # Get more data for resampling
            )

            logger.info(f"📈 Retrieved {len(candles) if candles else 0} raw candles from database")

            await db_handler.close()
            return candles

        # Get candle data
        candles = asyncio.run(fetch_candle_data())

        if not candles:
            logger.warning(f"❌ No candles found for {symbol}")
            return jsonify({
                'success': False,
                'error': f'No data found for {symbol}'
            }), 404

        logger.info(f"📊 Processing {len(candles)} candles")

        # Convert to DataFrame
        df = pd.DataFrame([dict(row) for row in candles])
        logger.info(f"DataFrame columns: {df.columns.tolist()}")
        logger.info(f"DataFrame shape: {df.shape}")

        df = df.set_index('bucket')
        df.index = pd.to_datetime(df.index)

        # Convert to proper numeric types
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Remove any rows with NaN values
        df = df.dropna()
        logger.info(f"After cleaning: {df.shape}")

        if df.empty:
            logger.warning(f"❌ No valid numeric data for {symbol}")
            return jsonify({
                'success': False,
                'error': f'No valid numeric data for {symbol}'
            }), 404

        # Sort by timestamp ascending (oldest first)
        df = df.sort_index()

        # Resample to requested timeframe if needed
        if timeframe != '1m':
            logger.info(f"🔄 Resampling from 1m to {timeframe}")
            df = resample_ohlcv_data(df, timeframe)
            logger.info(f"After resampling: {df.shape}")

        # Limit the results (get the most recent data)
        df = df.tail(limit)
        logger.info(f"Final data shape: {df.shape}")

        # Convert to the format expected by the frontend
        candles = []
        for timestamp, row in df.iterrows():
            candles.append({
                'time': int(timestamp.timestamp()),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume'])
            })

        logger.info(f"✅ Returning {len(candles)} formatted candles")

        return jsonify({
            'success': True,
            'data': {
                'symbol': symbol,
                'timeframe': timeframe,
                'candles': candles
            }
        })

    except Exception as e:
        logger.error(f"❌ Error getting candle data for {symbol}: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
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