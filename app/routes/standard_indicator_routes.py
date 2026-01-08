from flask import Blueprint, request, jsonify
import pandas as pd
import numpy as np
import logging
import asyncio
from app.db.db_handler import DBHandler
from app.routes.indicator_routes import resample_ohlcv_data
from app.indicators.exponential_moving_average import ExponentialMovingAverage
from app.indicators.volume_profile import VolumeProfile
from app.indicators.volume_weighted_regression import VolumeWeightedRegression
from app.indicators.regime_metrices import RegimeMetrics
from app.indicators.fractal_channel import FractalChannel
from app.indicators.rsi import RSI

standard_indicator_bp = Blueprint('standard_indicator', __name__)
logger = logging.getLogger(__name__)

import app.globals as app_globals

def fetch_candles_sync(symbol, start_dt, end_dt):
    """
    Fetch candles using the shared MarketService pool if available,
    otherwise fallback to creating a new connection.
    """
    ms = app_globals.market_service_instance
    
    if ms and ms.db_pool and ms.loop and ms.loop.is_running():
        # Use MarketService's pool via thread-safe execution
        future = asyncio.run_coroutine_threadsafe(
            ms.db_pool.read_candles(
                symbol=symbol,
                interval='1m',
                start_time=start_dt.to_pydatetime() if start_dt else None,
                end_time=end_dt.to_pydatetime() if end_dt else None,
                limit=None
            ),
            ms.loop
        )
        return future.result()
    else:
        # Fallback: Create temporary connection
        async def _fetch():
            db_handler = DBHandler()
            await db_handler.initialize()
            candles = await db_handler.read_candles(
                symbol=symbol,
                interval='1m',
                start_time=start_dt.to_pydatetime() if start_dt else None,
                end_time=end_dt.to_pydatetime() if end_dt else None,
                limit=None
            )
            await db_handler.close()
            return candles
            
        return asyncio.run(_fetch())

def prepare_dataframe(candles, interval):
    if not candles:
        return None
    
    df = pd.DataFrame([dict(row) for row in candles])
    df = df.set_index('bucket')
    df.index = pd.to_datetime(df.index)
    
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna().sort_index()

    if interval != '1m':
        df = resample_ohlcv_data(df, interval)
        
    return df

@standard_indicator_bp.route('/api/ema/calculate', methods=['GET'])
def calculate_ema():
    try:
        symbol = request.args.get('symbol', 'BTCUSDT')
        interval = request.args.get('interval', '1h')
        start_dt_str = request.args.get('startDateTime')
        end_dt_str = request.args.get('endDateTime')
        
        period = int(request.args.get('period', 20))
        source = request.args.get('source', 'close')

        start_dt = pd.to_datetime(start_dt_str, utc=True) if start_dt_str else None
        end_dt = pd.to_datetime(end_dt_str, utc=True) if end_dt_str else None

        candles = fetch_candles_sync(symbol, start_dt, end_dt)
        df = prepare_dataframe(candles, interval)
        
        if df is None or df.empty:
            return jsonify({'success': False, 'error': 'No data found'}), 404

        ema = ExponentialMovingAverage(period=period, source=source)
        df = ema.calculate(df)
        
        col_name = f'ema_{period}_{source}'
        
        tvlc_data = {'lines': []}
        line_data = []
        
        for idx, row in df.iterrows():
            if pd.notna(row.get(col_name)):
                line_data.append({
                    'time': int(idx.timestamp()),
                    'value': row[col_name]
                })
                
        if line_data:
            tvlc_data['lines'].append({
                'name': f'EMA {period}',
                'color': '#2962FF',
                'lineWidth': 2,
                'data': line_data
            })
            
        return jsonify({'success': True, 'tvlc_data': tvlc_data})
        
    except Exception as e:
        logger.error(f"Error in EMA calculation: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@standard_indicator_bp.route('/api/volume_profile/calculate', methods=['GET'])
def calculate_volume_profile():
    try:
        symbol = request.args.get('symbol', 'BTCUSDT')
        interval = request.args.get('interval', '1h')
        start_dt_str = request.args.get('startDateTime')
        end_dt_str = request.args.get('endDateTime')
        
        bins = int(request.args.get('bins', 100))
        lookback = int(request.args.get('lookback', 200))
        session_mode = request.args.get('session_mode', 'false').lower() == 'true'
        
        start_dt = pd.to_datetime(start_dt_str, utc=True) if start_dt_str else None
        end_dt = pd.to_datetime(end_dt_str, utc=True) if end_dt_str else None

        candles = fetch_candles_sync(symbol, start_dt, end_dt)
        df = prepare_dataframe(candles, interval)
        
        if df is None or df.empty:
            return jsonify({'success': False, 'error': 'No data found'}), 404

        vp = VolumeProfile(bins=bins, lookback=lookback, session_mode=session_mode)
        # Calculate developing series for plotting lines
        df_dev = vp.calculate_developing_series(df)
        
        tvlc_data = {'lines': []}
        
        # Helper to add line
        def add_line(name, color, col_name, style=0):
            data = []
            for idx, row in df_dev.iterrows():
                if pd.notna(row.get(col_name)):
                    data.append({
                        'time': int(idx.timestamp()),
                        'value': row[col_name]
                    })
            if data:
                tvlc_data['lines'].append({
                    'name': name,
                    'color': color,
                    'lineWidth': 2,
                    'lineStyle': style,
                    'data': data
                })

        add_line('Developing POC', 'yellow', 'developing_poc')
        add_line('Developing VAH', 'rgba(0, 255, 0, 0.7)', 'developing_vah', style=2) # Dashed
        add_line('Developing VAL', 'rgba(255, 0, 0, 0.7)', 'developing_val', style=2) # Dashed
            
        return jsonify({'success': True, 'tvlc_data': tvlc_data})
        
    except Exception as e:
        logger.error(f"Error in Volume Profile calculation: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@standard_indicator_bp.route('/api/vwr/calculate', methods=['GET'])
def calculate_vwr():
    try:
        symbol = request.args.get('symbol', 'BTCUSDT')
        interval = request.args.get('interval', '1h')
        start_dt_str = request.args.get('startDateTime')
        end_dt_str = request.args.get('endDateTime')
        
        lookback = int(request.args.get('lookback', 100))
        std_multiplier = float(request.args.get('std_multiplier', 2.0))
        
        start_dt = pd.to_datetime(start_dt_str, utc=True) if start_dt_str else None
        end_dt = pd.to_datetime(end_dt_str, utc=True) if end_dt_str else None

        candles = fetch_candles_sync(symbol, start_dt, end_dt)
        df = prepare_dataframe(candles, interval)
        
        if df is None or df.empty:
            return jsonify({'success': False, 'error': 'No data found'}), 404

        # Use the last 'lookback' candles for the regression
        subset = df.iloc[-lookback:] if len(df) >= lookback else df
        if subset.empty:
             return jsonify({'success': True, 'tvlc_data': {'lines': []}})

        vwr = VolumeWeightedRegression(lookback=lookback)
        vwr.calculate(subset)
        metrics = vwr.metrics
        
        if not metrics:
             return jsonify({'success': True, 'tvlc_data': {'lines': []}})

        # Calculate line points
        vw_slope = metrics['vw_slope']
        vw_price = metrics['vw_price'] # Value at the end
        std_dev = metrics.get('std_dev', 0)
        
        # Calculate intercept (start value)
        # y = mx + c => c = y - mx
        # At last point (x = lookback - 1), y = vw_price
        vw_intercept = vw_price - vw_slope * (len(subset) - 1)
        
        timestamps = subset.index.astype(np.int64) // 10**9
        
        center_line = []
        upper_band = []
        lower_band = []
        
        for i in range(len(subset)):
            ts = int(timestamps[i])
            
            # Base value
            val = vw_slope * i + vw_intercept
            
            center_line.append({'time': ts, 'value': val})
            upper_band.append({'time': ts, 'value': val + (std_dev * std_multiplier)})
            lower_band.append({'time': ts, 'value': val - (std_dev * std_multiplier)})
            
        tvlc_data = {'lines': [
            {
                'name': 'VWR Center',
                'color': 'cyan',
                'lineWidth': 2,
                'data': center_line
            },
            {
                'name': f'+{std_multiplier} StdDev',
                'color': 'rgba(0, 255, 255, 0.5)',
                'lineWidth': 1,
                'lineStyle': 2, # Dashed
                'data': upper_band
            },
            {
                'name': f'-{std_multiplier} StdDev',
                'color': 'rgba(0, 255, 255, 0.5)',
                'lineWidth': 1,
                'lineStyle': 2, # Dashed
                'data': lower_band
            }
        ]}
            
        return jsonify({'success': True, 'tvlc_data': tvlc_data})
        
    except Exception as e:
        logger.error(f"Error in VWR calculation: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@standard_indicator_bp.route('/api/regime/calculate', methods=['GET'])
def calculate_regime():
    try:
        symbol = request.args.get('symbol', 'BTCUSDT')
        interval = request.args.get('interval', '1h')
        start_dt_str = request.args.get('startDateTime')
        end_dt_str = request.args.get('endDateTime')
        
        hurst_lookback = int(request.args.get('hurst_lookback', 250))
        
        start_dt = pd.to_datetime(start_dt_str, utc=True) if start_dt_str else None
        end_dt = pd.to_datetime(end_dt_str, utc=True) if end_dt_str else None

        candles = fetch_candles_sync(symbol, start_dt, end_dt)
        df = prepare_dataframe(candles, interval)
        
        if df is None or df.empty:
            return jsonify({'success': False, 'error': 'No data found'}), 404

        rm = RegimeMetrics(hurst_lookback=hurst_lookback)
        df = rm.calculate(df)
        
        tvlc_data = {'lines': []}
        
        # Hurst Exponent
        hurst_data = []
        for idx, row in df.iterrows():
            if pd.notna(row.get('hurst')):
                hurst_data.append({
                    'time': int(idx.timestamp()),
                    'value': row['hurst']
                })
                
        if hurst_data:
            tvlc_data['lines'].append({
                'name': 'Hurst Exponent',
                'color': 'purple',
                'lineWidth': 2,
                'data': hurst_data
            })
            
        # Add threshold lines for Hurst (0.5)
        # TVLC doesn't support static lines easily in data, but we can add a constant series
        threshold_data = [{'time': d['time'], 'value': 0.5} for d in hurst_data]
        tvlc_data['lines'].append({
            'name': 'Random Walk (0.5)',
            'color': 'gray',
            'lineStyle': 2,
            'lineWidth': 1,
            'data': threshold_data
        })
            
        return jsonify({'success': True, 'tvlc_data': tvlc_data})
        
    except Exception as e:
        logger.error(f"Error in Regime calculation: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@standard_indicator_bp.route('/api/fractal/calculate', methods=['GET'])
def calculate_fractal_channel():
    """
    Dedicated endpoint for Fractal Channel calculation.
    Returns plot data compatible with the frontend chart.
    """
    try:
        # 1. Parse Parameters (Standardized)
        symbol = request.args.get('symbol', 'BTCUSDT')
        interval = request.args.get('interval', '1h')
        start_dt_str = request.args.get('startDateTime')
        end_dt_str = request.args.get('endDateTime')
        limit_param = request.args.get('limit')

        start_dt = pd.to_datetime(start_dt_str, utc=True) if start_dt_str else None
        end_dt = pd.to_datetime(end_dt_str, utc=True) if end_dt_str else None
        limit = int(limit_param) if limit_param is not None else None

        # 2. Fetch Data (Using optimized sync fetcher)
        candles = fetch_candles_sync(symbol, start_dt, end_dt)
        df = prepare_dataframe(candles, interval)

        if df is None or df.empty:
            return jsonify({'success': False, 'error': f'No data found for {symbol}'}), 404

        # 4. Initialize Indicator (Notebook Defaults)
        # These defaults match the notebook exactly
        fc = FractalChannel(
            mode='geometric',
            pivot_method='fractal',
            zigzag_dev=0.05,
            pivot_window=5,
            lookback=150
        )

        # 5. Calculate
        df_result = fc.calculate(df)
        
        # 6. Generate Plot Traces
        # This uses the updated _get_plot_trace which now matches the notebook logic
        plot_data = fc._get_plot_trace(df_result)

        # --- TVLC FORMATTING START ---
        tvlc_data = {
            'candles': [],
            'lines': [],
            'markers': []
        }

        # A. Candles (Main Series)
        # We use the resampled df directly
        for idx, row in df.iterrows():
            tvlc_data['candles'].append({
                'time': int(idx.timestamp()),
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close']
            })

        # B. Lines & Markers (From Plotly Traces)
        # We iterate through the Plotly traces generated by the indicator
        # and convert them to TVLC format.
        if isinstance(plot_data, dict) and 'data' in plot_data:
            for trace in plot_data['data']:
                trace_type = trace.get('type', 'scatter')
                mode = trace.get('mode', '')
                name = trace.get('name', 'Unknown')
                
                # Skip the main candlestick trace if it exists in plot_data (we built it above)
                if trace_type == 'candlestick':
                    continue

                x_vals = trace.get('x', [])
                y_vals = trace.get('y', [])
                
                # Ensure we have data
                if not len(x_vals) or not len(y_vals):
                    continue

                # Convert timestamps to unix
                # x_vals might be datetime objects or strings
                try:
                    if isinstance(x_vals[0], str):
                        x_ts = [int(pd.Timestamp(x).timestamp()) for x in x_vals]
                    else:
                        x_ts = [int(pd.Timestamp(x).timestamp()) for x in x_vals]
                except:
                    # Fallback if conversion fails
                    continue

                # 1. Lines
                if 'lines' in mode:
                    line_series = {
                        'name': name,
                        'color': trace.get('line', {}).get('color', '#2962FF'),
                        'lineWidth': trace.get('line', {}).get('width', 2),
                        'lineStyle': 2 if trace.get('line', {}).get('dash') == 'dot' else 0, # 0=Solid, 2=Dashed
                        'data': []
                    }
                    
                    for i, t in enumerate(x_ts):
                        val = y_vals[i]
                        # TVLC doesn't like NaNs in line data usually, but we can skip them
                        if val is not None and not (isinstance(val, float) and np.isnan(val)):
                            line_series['data'].append({'time': t, 'value': val})
                    
                    if line_series['data']:
                        tvlc_data['lines'].append(line_series)

                # 2. Markers (Pivots)
                if 'markers' in mode:
                    # Determine shape/color based on name or marker props
                    color = trace.get('marker', {}).get('color', '#2962FF')
                    # Simple heuristic for position
                    position = 'aboveBar' if 'High' in name or 'Bearish' in name else 'belowBar'
                    shape = 'arrowDown' if position == 'aboveBar' else 'arrowUp'
                    
                    for i, t in enumerate(x_ts):
                        val = y_vals[i]
                        if val is not None and not (isinstance(val, float) and np.isnan(val)):
                            tvlc_data['markers'].append({
                                'time': t,
                                'position': position,
                                'color': color,
                                'shape': shape,
                                'text': 'P' # P for Pivot
                            })
        # --- TVLC FORMATTING END ---

        # 7. Normalize Dates and Sanitize NaNs for JSON
        def _sanitize_trace(trace):
            # Normalize X (Dates)
            x_vals = trace.get('x')
            if isinstance(x_vals, (list, tuple, np.ndarray)):
                trace['x'] = [x.isoformat() if hasattr(x, 'isoformat') else str(x) for x in x_vals]
            
            # Sanitize Y (Values) - Replace NaN/Inf with None
            y_vals = trace.get('y')
            if isinstance(y_vals, (list, tuple, np.ndarray)):
                # Convert to list if it's numpy array
                if hasattr(y_vals, 'tolist'):
                    y_vals = y_vals.tolist()
                
                # Replace NaN/Inf with None
                trace['y'] = [None if (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) else v for v in y_vals]
            
            return trace

        if isinstance(plot_data, dict) and 'data' in plot_data:
            plot_data['data'] = [_sanitize_trace(t) for t in plot_data['data']]

        return jsonify({
            'success': True,
            'symbol': symbol,
            'interval': interval,
            'plot_data': plot_data,
            'tvlc_data': tvlc_data # New field
        })

    except Exception as e:
        logger.error(f"Error in fractal route: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@standard_indicator_bp.route('/api/rsi/calculate', methods=['GET'])
def calculate_rsi():
    """
    Dedicated endpoint for RSI with Dynamic Fractal Channel calculation.
    Returns plot data compatible with the frontend chart.
    """
    try:
        # 1. Parse Parameters
        symbol = request.args.get('symbol', 'BTCUSDT')
        interval = request.args.get('interval', '1h')
        start_dt_str = request.args.get('startDateTime')
        end_dt_str = request.args.get('endDateTime')
        limit_param = request.args.get('limit')
        
        # RSI Specific Params
        length = int(request.args.get('length', 14))
        source_col = request.args.get('source', 'close')
        
        # Fractal Channel Params
        fc_enabled = request.args.get('fc_enabled', 'true').lower() == 'true'
        fc_lookback = int(request.args.get('fc_lookback', 50))
        fc_mult = float(request.args.get('fc_mult', 2.0))
        fc_zigzag_dev = float(request.args.get('fc_zigzag_dev', 0.05))

        start_dt = pd.to_datetime(start_dt_str, utc=True) if start_dt_str else None
        end_dt = pd.to_datetime(end_dt_str, utc=True) if end_dt_str else None
        limit = int(limit_param) if limit_param is not None else None

        # 2. Fetch Data (Using optimized sync fetcher)
        candles = fetch_candles_sync(symbol, start_dt, end_dt)
        df = prepare_dataframe(candles, interval)

        if df is None or df.empty:
            return jsonify({'success': False, 'error': f'No data found for {symbol}'}), 404

        # 4. Initialize Indicator
        rsi = RSI(
            length=length,
            source=source_col,
            fc_enabled=fc_enabled,
            fc_lookback=fc_lookback,
            fc_mult=fc_mult,
            fc_zigzag_dev=fc_zigzag_dev
        )

        # 5. Calculate
        df_result = rsi.calculate(df)
        
        # 6. Format for TVLC
        tvlc_data = {
            'main': [], # RSI Line Data for the main series
            'lines': [],
            'markers': []
        }

        # RSI Line (Main Series)
        rsi_col = f'rsi_{length}_{source_col}'
        for idx, row in df_result.iterrows():
            if pd.notna(row.get(rsi_col)):
                tvlc_data['main'].append({
                    'time': int(idx.timestamp()),
                    'value': row[rsi_col]
                })

        # Helper to add line series
        def add_line(name, color, data_col, style=0, width=1):
            line_data = []
            for idx, row in df_result.iterrows():
                if pd.notna(row.get(data_col)):
                    line_data.append({
                        'time': int(idx.timestamp()),
                        'value': row[data_col]
                    })
            if line_data:
                tvlc_data['lines'].append({
                    'name': name,
                    'color': color,
                    'lineWidth': width,
                    'lineStyle': style, # 0=Solid, 1=Dotted, 2=Dashed
                    'data': line_data
                })

        # Fractal Channel Lines
        if fc_enabled:
            upper_col = f'fc_upper_{fc_lookback}_dynamic'
            lower_col = f'fc_lower_{fc_lookback}_dynamic'
            mid_col = f'fc_mid_{fc_lookback}_dynamic'
            
            add_line('RSI Channel Upper', 'rgba(0, 255, 0, 0.5)', upper_col, style=2)
            add_line('RSI Channel Lower', 'rgba(255, 0, 0, 0.5)', lower_col, style=2)
            add_line('RSI Channel Mid', 'gray', mid_col)

            # Signals
            signal_col = 'fc_signal'
            if signal_col in df_result.columns:
                for idx, row in df_result.iterrows():
                    sig = row[signal_col]
                    if sig == 1: # Bullish
                        tvlc_data['markers'].append({
                            'time': int(idx.timestamp()),
                            'position': 'belowBar',
                            'color': 'lime',
                            'shape': 'arrowUp',
                            'text': 'Buy'
                        })
                    elif sig == -1: # Bearish
                        tvlc_data['markers'].append({
                            'time': int(idx.timestamp()),
                            'position': 'aboveBar',
                            'color': 'red',
                            'shape': 'arrowDown',
                            'text': 'Sell'
                        })

        # Overbought/Oversold Lines (Static)
        ob_data = [{'time': int(idx.timestamp()), 'value': 70} for idx in df_result.index]
        os_data = [{'time': int(idx.timestamp()), 'value': 30} for idx in df_result.index]
        
        tvlc_data['lines'].append({
            'name': 'Overbought',
            'color': 'rgba(255, 255, 255, 0.3)',
            'lineStyle': 1, # Dotted
            'lineWidth': 1,
            'data': ob_data
        })
        
        tvlc_data['lines'].append({
            'name': 'Oversold',
            'color': 'rgba(255, 255, 255, 0.3)',
            'lineStyle': 1,
            'lineWidth': 1,
            'data': os_data
        })

        return jsonify({'success': True, 'tvlc_data': tvlc_data})

    except Exception as e:
        logger.error(f"Error in RSI calculation: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500
