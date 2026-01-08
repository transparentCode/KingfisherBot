from flask import Blueprint, request, jsonify, render_template
import pandas as pd
import numpy as np
import logging
import asyncio
from app.services.indicator_registers import IndicatorRegistry
from app.db.db_handler import DBHandler

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
        # Get request parameters
        symbol = request.args.get('symbol', 'BTCUSDT')
        interval = request.args.get('interval', '1h')
        start_dt_str = request.args.get('startDateTime')
        end_dt_str = request.args.get('endDateTime')
        limit_param = request.args.get('limit')

        # Parse optional date range
        start_dt = pd.to_datetime(start_dt_str, utc=True) if start_dt_str else None
        end_dt = pd.to_datetime(end_dt_str, utc=True) if end_dt_str else None
        limit = int(limit_param) if limit_param is not None else None

        # Get indicator registry
        registry = IndicatorRegistry.register_default_indicators()

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
                start_time=start_dt.to_pydatetime() if start_dt is not None else None,
                end_time=end_dt.to_pydatetime() if end_dt is not None else None,
                limit=limit * 10 if limit else None
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

        # IMPORTANT: Limit BEFORE calculation to ensure alignment (only if limit provided)
        if limit:
            df = df.tail(limit)

        # Get indicator parameters from query string
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

        # Create indicator instance (FIX: Use create_indicator_instance instead of get_indicator_for_ui)
        indicator = registry.create_indicator_instance(indicator_id, **params)

        # Calculate indicator values
        indicator_data = indicator.calculate(df)

        logger.info(f"Indicator data calculated successfully for {indicator_id}: {df.shape}")

        # Generate plot data
        plot_data = indicator._get_plot_trace(indicator_data)

        # --- TVLC FORMATTING START ---
        tvlc_data = {
            'candles': [],
            'lines': [],
            'markers': [],
            'histograms': []
        }

        # A. Candles (Main Series)
        for idx, row in df.iterrows():
            tvlc_data['candles'].append({
                'time': int(idx.timestamp()),
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close']
            })

        # B. Lines, Markers & Histograms (From Plotly Traces)
        traces = []
        if isinstance(plot_data, list):
            traces = plot_data
        elif isinstance(plot_data, dict) and 'data' in plot_data:
            traces = plot_data['data']

        for trace in traces:
            trace_type = trace.get('type', 'scatter')
            mode = trace.get('mode', '')
            name = trace.get('name', 'Unknown')
            
            # Skip main candlestick if present
            if trace_type == 'candlestick':
                continue

            x_vals = trace.get('x', [])
            y_vals = trace.get('y', [])
            
            if not len(x_vals) or not len(y_vals):
                continue

            # Convert timestamps
            try:
                if isinstance(x_vals[0], str):
                    x_ts = [int(pd.Timestamp(x).timestamp()) for x in x_vals]
                else:
                    x_ts = [int(pd.Timestamp(x).timestamp()) for x in x_vals]
            except:
                continue

            # 1. Lines
            if 'lines' in mode or trace_type == 'scatter':
                # Check if it's actually a line (Plotly default is lines+markers sometimes)
                if 'lines' in mode or (mode == '' and trace_type == 'scatter'):
                    line_series = {
                        'name': name,
                        'color': trace.get('line', {}).get('color', '#2962FF'),
                        'lineWidth': trace.get('line', {}).get('width', 2),
                        'lineStyle': 2 if trace.get('line', {}).get('dash') == 'dot' else 0,
                        'data': []
                    }
                    for i, t in enumerate(x_ts):
                        val = y_vals[i]
                        if val is not None and not (isinstance(val, float) and np.isnan(val)):
                            line_series['data'].append({'time': t, 'value': val})
                    
                    if line_series['data']:
                        tvlc_data['lines'].append(line_series)

            # 2. Markers
            if 'markers' in mode:
                color = trace.get('marker', {}).get('color', '#2962FF')
                # Heuristic for position
                position = 'aboveBar' if 'High' in name or 'Sell' in name or 'Bearish' in name else 'belowBar'
                shape = 'arrowDown' if position == 'aboveBar' else 'arrowUp'
                
                for i, t in enumerate(x_ts):
                    val = y_vals[i]
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        tvlc_data['markers'].append({
                            'time': t,
                            'position': position,
                            'color': color,
                            'shape': shape,
                            'text': name[:1] # First letter as label
                        })

            # 3. Bar/Histogram (e.g. MACD)
            if trace_type == 'bar':
                hist_series = {
                    'name': name,
                    'color': trace.get('marker', {}).get('color', '#26a69a'),
                    'data': []
                }
                for i, t in enumerate(x_ts):
                    val = y_vals[i]
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        hist_series['data'].append({'time': t, 'value': val})
                
                if hist_series['data']:
                    tvlc_data['histograms'].append(hist_series)
        # --- TVLC FORMATTING END ---

        # Normalize datetime objects so the frontend chart can render them reliably
        def _normalize_trace_x(trace):
            x_vals = trace.get('x')
            if isinstance(x_vals, (list, tuple)):
                trace['x'] = [x.isoformat() if hasattr(x, 'isoformat') else x for x in x_vals]
            return trace

        if isinstance(plot_data, list):
            plot_data = [_normalize_trace_x(t) for t in plot_data]
        elif isinstance(plot_data, dict) and 'data' in plot_data:
            plot_data['data'] = [_normalize_trace_x(t) for t in plot_data['data']]

        return jsonify({
            'success': True,
            'indicator_id': indicator_id,
            'indicator_name': indicator_info['display_name'],
            'symbol': symbol,
            'interval': interval,
            'data_points': len(df),
            'plot_data': plot_data,
            'tvlc_data': tvlc_data, # New field
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