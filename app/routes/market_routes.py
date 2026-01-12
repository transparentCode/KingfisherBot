from flask import Blueprint, request, jsonify
import logging
import asyncio

from app.db.mtf_data_manager import MTFDataManager
from app.utils.market_analysis_utils import get_market_status_with_fallback
from app.utils.async_db_utils import precise_db_task
from app.utils.validation_utils import validate_asset_readiness, validate_candle_request_params
from app.utils.tvlc_utils import format_candles_for_response

market_bp = Blueprint("market", __name__)
logger = logging.getLogger(__name__)


@market_bp.route("/api/asset/regime/<symbol>", methods=["GET"])
def get_asset_regime(symbol):
    """
    Get latest Regime and Hilbert Cycle metrics for a symbol and timeframe.
    If database data is missing (cold start), it calculates on-demand.
    """
    try:
        # 0. Asset Readiness Validation
        if error_response := validate_asset_readiness(symbol):
            return error_response

        timeframe = request.args.get("timeframe", "1h")
        logger.info(f"🔍 Fetching market status for {symbol} {timeframe}")

        async def fetch_task(db_handler):
            return await get_market_status_with_fallback(db_handler, symbol, timeframe)

        result = asyncio.run(precise_db_task(fetch_task))
        
        if not result:
             return jsonify({
                "success": False, 
                "error": "Could not determine market status (No data)"
            }), 404

        return jsonify({"success": True, "data": result})

    except Exception as e:
        logger.error(f"Error getting market status: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@market_bp.route("/api/asset/search-symbols", methods=["GET"])
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


@market_bp.route("/api/asset/candle-data/<symbol>")
def get_candle_data(symbol):
    """Get candlestick data for a symbol within a time range"""
    logger.info(f"📊 Fetching candle data for {symbol}")
    try:
        if error_response := validate_asset_readiness(symbol):
            return error_response
        
        timeframe = request.args.get("timeframe", "1h")
        start_dt_str = request.args.get("startDateTime")
        end_dt_str = request.args.get("endDateTime")
        limit_param = request.args.get("limit")

        # 1. Validate Params
        start_dt, end_dt, error_response = validate_candle_request_params(start_dt_str, end_dt_str)
        if error_response:
            return error_response

        limit = int(limit_param) if limit_param is not None else None

        logger.info(
            f"Parameters: timeframe={timeframe}, start={start_dt_str}, end={end_dt_str}, limit={limit}"
        )

        async def fetch_task(db_handler):
            # Use MTF manager logic which we added to execute the read
            mtf = MTFDataManager(db_handler)
            return await mtf.get_candles_df(symbol, start_dt, end_dt, limit)

        # 2. Get processed DataFrame directly
        df = asyncio.run(precise_db_task(fetch_task))

        if df.empty:
            logger.warning(f"❌ No valid numeric data for {symbol}")
            return jsonify(
                {"success": False, "error": f"No valid numeric data for {symbol}"}
            ), 404

        # Resample to requested timeframe if needed
        if timeframe != "1m":
            logger.info(f"🔄 Resampling from 1m to {timeframe}")
            df = MTFDataManager.resample_ohlcv(df, timeframe)
            logger.info(f"After resampling: {df.shape}")

        # Only trim if caller explicitly provided a limit; otherwise return full range
        if limit:
            df = df.tail(limit)

        logger.info(f"Final data shape: {df.shape}")

        # Format for response using utility
        formatted_data = format_candles_for_response(df)
        candles = formatted_data['candles']
        tvlc_candles = formatted_data['tvlc_candles']

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
                    "tvlc_data": { "candles": tvlc_candles }, 
                },
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error getting candle data for {symbol}: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
