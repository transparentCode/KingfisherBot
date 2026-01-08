from flask import Blueprint, request, jsonify
from app.db.db_handler import DBHandler
import pandas as pd
from datetime import datetime, timedelta
import logging
import asyncio

market_bp = Blueprint("market", __name__)
logger = logging.getLogger(__name__)


from config.asset_indicator_config import ConfigurationManager


@market_bp.route("/api/search-symbols", methods=["GET"])
def search_symbols():
    try:
        query = request.args.get("q", "").upper()

        manager = ConfigurationManager()
        enabled_assets = manager.get_enabled_assets()

        # Convert to list of dicts
        all_symbols = [{"symbol": asset, "name": asset} for asset in enabled_assets]

        # Filter symbols based on query
        filtered_symbols = [
            symbol for symbol in all_symbols if query in symbol["symbol"]
        ]

        return jsonify({"success": True, "symbols": filtered_symbols})

    except Exception as e:
        logger.error(f"Error searching symbols: {str(e)}")
        return jsonify({"success": False, "error": str(e)})


@market_bp.route("/api/candle-data/<symbol>")
def get_candle_data(symbol):
    """Get candlestick data for a symbol within a time range"""
    logger.info(f"📊 Fetching candle data for {symbol}")
    try:
        timeframe = request.args.get("timeframe", "1h")
        start_dt_str = request.args.get("startDateTime")
        end_dt_str = request.args.get("endDateTime")
        limit_param = request.args.get("limit")

        if not start_dt_str or not end_dt_str:
            return (
                jsonify({
                    "success": False,
                    "error": "Missing required params: startDateTime and endDateTime"
                }),
                400,
            )

        # Basic validation; parse to timezone-aware datetimes for asyncpg
        try:
            start_dt = pd.to_datetime(start_dt_str, utc=True)
            end_dt = pd.to_datetime(end_dt_str, utc=True)
        except Exception:
            return (
                jsonify({"success": False, "error": "Invalid datetime format"}),
                400,
            )

        limit = int(limit_param) if limit_param is not None else None

        logger.info(
            f"Parameters: timeframe={timeframe}, start={start_dt_str}, end={end_dt_str}, limit={limit}"
        )

        async def fetch_candle_data():
            db_handler = DBHandler()
            await db_handler.initialize()
            logger.info("✅ DB handler initialized")

            candles = await db_handler.read_candles(
                symbol=symbol,
                interval="1m",  # Always get base 1m data, then resample
                start_time=start_dt.to_pydatetime(),
                end_time=end_dt.to_pydatetime(),
                limit=limit,
            )

            logger.info(
                f"📈 Retrieved {len(candles) if candles else 0} raw candles from database"
            )

            await db_handler.close()
            return candles

        # Get candle data
        candles = asyncio.run(fetch_candle_data())

        if not candles:
            logger.warning(f"❌ No candles found for {symbol}")
            return jsonify(
                {"success": False, "error": f"No data found for {symbol}"}
            ), 404

        logger.info(f"📊 Processing {len(candles)} candles")

        # Convert to DataFrame
        df = pd.DataFrame([dict(row) for row in candles])
        logger.info(f"DataFrame columns: {df.columns.tolist()}")
        logger.info(f"DataFrame shape: {df.shape}")

        df = df.set_index("bucket")
        df.index = pd.to_datetime(df.index)

        # Convert to proper numeric types
        numeric_columns = ["open", "high", "low", "close", "volume"]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Remove any rows with NaN values
        df = df.dropna()
        logger.info(f"After cleaning: {df.shape}")

        if df.empty:
            logger.warning(f"❌ No valid numeric data for {symbol}")
            return jsonify(
                {"success": False, "error": f"No valid numeric data for {symbol}"}
            ), 404

        # Sort by timestamp ascending (oldest first)
        df = df.sort_index()

        # Resample to requested timeframe if needed
        if timeframe != "1m":
            logger.info(f"🔄 Resampling from 1m to {timeframe}")
            df = resample_ohlcv_data(df, timeframe)
            logger.info(f"After resampling: {df.shape}")

        # Only trim if caller explicitly provided a limit; otherwise return full range
        if limit:
            df = df.tail(limit)

        logger.info(f"Final data shape: {df.shape}")

        # Convert to the format expected by the frontend
        candles = []
        tvlc_candles = [] # New TVLC format
        
        for timestamp, row in df.iterrows():
            # Existing format
            candles.append(
                {
                    "time": int(timestamp.timestamp()),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                }
            )
            
            # TVLC format
            tvlc_candles.append(
                {
                    "time": int(timestamp.timestamp()),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                }
            )

        logger.info(f"✅ Returning {len(candles)} formatted candles")

        return jsonify(
            {
                "success": True,
                "data": {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "startDateTime": start_dt.isoformat(),
                    "endDateTime": end_dt.isoformat(),
                    "candles": candles,
                    "tvlc_data": { "candles": tvlc_candles }, # New field
                },
            }
        )

    except Exception as e:
        logger.error(f"❌ Error getting candle data for {symbol}: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


def resample_ohlcv_data(df, timeframe):
    """Resample OHLCV data to a different timeframe"""
    try:
        # Ensure the index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Define resampling rules for different timeframes
        timeframe_map = {
            "1m": "1T",  # 1 minute
            "5m": "5T",  # 5 minutes
            "15m": "15T",  # 15 minutes
            "30m": "30T",  # 30 minutes
            "1h": "1H",  # 1 hour
            "4h": "4H",  # 4 hours
            "1d": "1D",  # 1 day
            "1w": "1W",  # 1 week
        }

        if timeframe not in timeframe_map:
            logger.warning(
                f"Unsupported timeframe: {timeframe}, returning original data"
            )
            return df

        freq = timeframe_map[timeframe]

        # Resample OHLCV data
        resampled = (
            df.resample(freq)
            .agg(
                {
                    "open": "first",  # First open price in the period
                    "high": "max",  # Highest price in the period
                    "low": "min",  # Lowest price in the period
                    "close": "last",  # Last close price in the period
                    "volume": "sum",  # Sum of volume in the period
                }
            )
            .dropna()
        )

        return resampled

    except Exception as e:
        logger.error(f"Error resampling data to {timeframe}: {e}")
        return df
