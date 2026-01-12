from flask import Blueprint, request, jsonify
import logging
import asyncio

from app.services.indicator_registers import IndicatorRegistry
from app.db.mtf_data_manager import MTFDataManager
from app.utils.async_db_utils import precise_db_task
from app.utils.validation_utils import validate_candle_request_params
from app.utils.tvlc_utils import convert_indicator_output_to_tvlc, normalize_plotly_trace_for_json

indicator_bp = Blueprint('indicator', __name__)

logger = logging.getLogger(__name__)

@indicator_bp.route('/api/indicators/available', methods=['GET'])
def get_available_indicators_for_ui():
    """Get all indicators formatted for UI with parameter schemas"""
    try:
        registry = IndicatorRegistry.register_default_indicators()  # Ensure indicators are registered
        categories_map = registry.get_all_indicators_for_ui()

        # UI expects a list of { category, indicators: [...] }
        categories = []
        for category, indicators in categories_map.items():
            normalized = []
            for ind in indicators:
                # Ensure indicator_id key exists (frontend flattener expects it)
                norm = {
                    "indicator_id": ind.get("id"),
                    "display_name": ind.get("display_name"),
                    "description": ind.get("description"),
                    "category": ind.get("category"),
                    "parameters": ind.get("parameters"),
                    "input_columns": ind.get("input_columns"),
                }
                normalized.append(norm)

            categories.append({"category": category, "indicators": normalized})

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
        registry = IndicatorRegistry.register_default_indicators()
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
    """Calculate and return indicator data using database candles (supports date range)."""
    try:
        # 1. Validation & Param Parsing
        symbol = request.args.get('symbol', 'BTCUSDT')
        interval = request.args.get('interval', '1h')
        start_dt_str = request.args.get('startDateTime')
        end_dt_str = request.args.get('endDateTime')
        limit_param = request.args.get('limit')

        start_dt, end_dt, error_response = validate_candle_request_params(start_dt_str, end_dt_str)
        if error_response:
            return error_response
            
        limit = int(limit_param) if limit_param is not None else None

        registry = IndicatorRegistry.register_default_indicators()
        indicator_info = registry.get_indicator_for_ui(indicator_id)
        if not indicator_info:
            return jsonify({'success': False, 'error': f'Indicator {indicator_id} not found'}), 404

        # 2. Fetch Data
        async def fetch_task(db_handler):
            mtf = MTFDataManager(db_handler)
            return await mtf.get_candles_df(symbol, start_dt, end_dt, limit)

        df = asyncio.run(precise_db_task(fetch_task))

        if df.empty:
            return jsonify({'success': False, 'error': f'No data found for {symbol}'}), 404

        # 3. Resample
        if interval != '1m':
             df = MTFDataManager.resample_ohlcv(df, interval)
             
        if limit:
            df = df.tail(limit)

        # 4. Parse Params
        params = {}
        for key, value in request.args.items():
            if key not in ['symbol', 'interval', 'lookback_days', 'startDateTime', 'endDateTime', 'limit']:
                try:
                    if '.' in value:
                        params[key] = float(value)
                    else:
                        params[key] = int(value)
                except ValueError:
                    params[key] = value

        # 5. Calculate
        indicator = registry.create_indicator_instance(indicator_id, **params)
        indicator_data = indicator.calculate(df)
        plot_data = indicator._get_plot_trace(indicator_data)

        # 6. Format Output
        tvlc_data = convert_indicator_output_to_tvlc(df, plot_data)
        plot_data_normalized = normalize_plotly_trace_for_json(plot_data)

        logger.info(f"Indicator calculated: {indicator_id} {symbol} {interval}")

        return jsonify({
            'success': True,
            'indicator_id': indicator_id,
            'indicator_name': indicator_info['display_name'],
            'symbol': symbol,
            'interval': interval,
            'data_points': len(df),
            'plot_data': plot_data_normalized,
            'tvlc_data': tvlc_data,
            'parameters': params
        })

    except Exception as e:
        logger.error(f"Error calculating indicator {indicator_id}: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e), 'indicator_id': indicator_id}), 500
