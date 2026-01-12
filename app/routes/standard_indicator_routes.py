from flask import Blueprint, request, jsonify
import pandas as pd
import numpy as np
import logging
import asyncio

from app.db.mtf_data_manager import MTFDataManager
from app.utils.async_db_utils import precise_db_task
from app.utils.validation_utils import validate_candle_request_params
from app.utils.tvlc_utils import convert_indicator_output_to_tvlc, normalize_plotly_trace_for_json

from app.indicators.exponential_moving_average import ExponentialMovingAverage
from app.indicators.volume_profile import VolumeProfile
from app.indicators.volume_weighted_regression import VolumeWeightedRegression
from app.indicators.regime_metrices import RegimeMetrics
from app.indicators.fractal_channel import FractalChannel
from app.indicators.rsi import RSI

standard_indicator_bp = Blueprint('standard_indicator', __name__)
logger = logging.getLogger(__name__)

async def _fetch_and_prepare_data(symbol, interval, start_dt, end_dt, limit=None):
    """
    Helper to fetch and resample candles for standard indicators.
    """
    async def fetch_task(db_handler):
        mtf = MTFDataManager(db_handler)
        return await mtf.get_candles_df(symbol, start_dt, end_dt, limit)
        
    df = await precise_db_task(fetch_task)
    
    if df.empty:
        return None
        
    if interval != '1m':
        df = MTFDataManager.resample_ohlcv(df, interval)
        
    return df

@standard_indicator_bp.route('/api/ema/calculate', methods=['GET'])
def calculate_ema():
    try:
        # Params
        symbol = request.args.get('symbol', 'BTCUSDT')
        interval = request.args.get('interval', '1h')
        period = int(request.args.get('period', 20))
        source = request.args.get('source', 'close')

        start_dt, end_dt, err = validate_candle_request_params(
            request.args.get('startDateTime'), 
            request.args.get('endDateTime')
        )
        if err: return err

        # Data
        df = asyncio.run(_fetch_and_prepare_data(symbol, interval, start_dt, end_dt))
        if df is None or df.empty:
            return jsonify({'success': False, 'error': 'No data found'}), 404

        # Calc
        ema = ExponentialMovingAverage(period=period, source=source)
        df = ema.calculate(df)
        
        # Format
        col_name = f'ema_{period}_{source}'
        tvlc_data = {'lines': []}
        line_data = []
        
        for idx, row in df.iterrows():
            if pd.notna(row.get(col_name)):
                line_data.append({'time': int(idx.timestamp()), 'value': row[col_name]})
                
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
        bins = int(request.args.get('bins', 100))
        lookback = int(request.args.get('lookback', 200))
        session_mode = request.args.get('session_mode', 'false').lower() == 'true'
        
        start_dt, end_dt, err = validate_candle_request_params(
            request.args.get('startDateTime'), 
            request.args.get('endDateTime')
        )
        if err: return err

        df = asyncio.run(_fetch_and_prepare_data(symbol, interval, start_dt, end_dt))
        if df is None or df.empty:
            return jsonify({'success': False, 'error': 'No data found'}), 404

        vp = VolumeProfile(bins=bins, lookback=lookback, session_mode=session_mode)
        df_dev = vp.calculate_developing_series(df)
        
        tvlc_data = {'lines': []}
        
        def add_line(name, color, col_name, style=0):
            data = [{'time': int(ts.timestamp()), 'value': val} 
                   for ts, val in df_dev[col_name].dropna().items()]
            if data:
                tvlc_data['lines'].append({
                    'name': name, 'color': color, 'lineWidth': 2, 'lineStyle': style, 'data': data
                })

        add_line('Developing POC', 'yellow', 'developing_poc')
        add_line('Developing VAH', 'rgba(0, 255, 0, 0.7)', 'developing_vah', style=2)
        add_line('Developing VAL', 'rgba(255, 0, 0, 0.7)', 'developing_val', style=2)
            
        return jsonify({'success': True, 'tvlc_data': tvlc_data})
        
    except Exception as e:
        logger.error(f"Error in Volume Profile calculation: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@standard_indicator_bp.route('/api/vwr/calculate', methods=['GET'])
def calculate_vwr():
    try:
        symbol = request.args.get('symbol', 'BTCUSDT')
        interval = request.args.get('interval', '1h')
        lookback = int(request.args.get('lookback', 100))
        std_multiplier = float(request.args.get('std_multiplier', 2.0))
        
        start_dt, end_dt, err = validate_candle_request_params(
            request.args.get('startDateTime'), 
            request.args.get('endDateTime')
        )
        if err: return err

        df = asyncio.run(_fetch_and_prepare_data(symbol, interval, start_dt, end_dt))
        if df is None or df.empty:
            return jsonify({'success': False, 'error': 'No data found'}), 404

        subset = df.iloc[-lookback:] if len(df) >= lookback else df
        if subset.empty:
             return jsonify({'success': True, 'tvlc_data': {'lines': []}})

        vwr = VolumeWeightedRegression(lookback=lookback)
        vwr.calculate(subset)
        metrics = vwr.metrics
        
        if not metrics:
             return jsonify({'success': True, 'tvlc_data': {'lines': []}})

        vw_slope = metrics['vw_slope']
        vw_price = metrics['vw_price']
        std_dev = metrics.get('std_dev', 0)
        vw_intercept = vw_price - vw_slope * (len(subset) - 1)
        
        timestamps = subset.index.astype(np.int64) // 10**9
        indices = np.arange(len(subset))
        
        vals = vw_slope * indices + vw_intercept
        upper = vals + (std_dev * std_multiplier)
        lower = vals - (std_dev * std_multiplier)
        
        # Vectorized construction of list of dicts is faster locally, but iterating is fine for API response sizes
        center_line = [{'time': int(t), 'value': float(v)} for t, v in zip(timestamps, vals)]
        upper_band = [{'time': int(t), 'value': float(v)} for t, v in zip(timestamps, upper)]
        lower_band = [{'time': int(t), 'value': float(v)} for t, v in zip(timestamps, lower)]
        
        tvlc_data = {'lines': [
            {'name': 'VWR Center', 'color': 'cyan', 'lineWidth': 2, 'data': center_line},
            {'name': f'+{std_multiplier} StdDev', 'color': 'rgba(0, 255, 255, 0.5)', 'lineWidth': 1, 'lineStyle': 2, 'data': upper_band},
            {'name': f'-{std_multiplier} StdDev', 'color': 'rgba(0, 255, 255, 0.5)', 'lineWidth': 1, 'lineStyle': 2, 'data': lower_band}
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
        hurst_lookback = int(request.args.get('hurst_lookback', 250))
        
        start_dt, end_dt, err = validate_candle_request_params(
            request.args.get('startDateTime'), 
            request.args.get('endDateTime')
        )
        if err: return err

        df = asyncio.run(_fetch_and_prepare_data(symbol, interval, start_dt, end_dt))
        if df is None or df.empty:
            return jsonify({'success': False, 'error': 'No data found'}), 404

        rm = RegimeMetrics(hurst_lookback=hurst_lookback)
        df = rm.calculate(df)
        
        hurst_data = [{'time': int(ts.timestamp()), 'value': val} 
                     for ts, val in df['hurst'].dropna().items()]
                
        tvlc_data = {'lines': []}
        if hurst_data:
            tvlc_data['lines'].append({
                'name': 'Hurst Exponent', 'color': 'purple', 'lineWidth': 2, 'data': hurst_data
            })
            tvlc_data['lines'].append({
                'name': 'Random Walk (0.5)', 'color': 'gray', 'lineStyle': 2, 'lineWidth': 1,
                'data': [{'time': d['time'], 'value': 0.5} for d in hurst_data]
            })
            
        return jsonify({'success': True, 'tvlc_data': tvlc_data})
        
    except Exception as e:
        logger.error(f"Error in Regime calculation: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@standard_indicator_bp.route('/api/fractal/calculate', methods=['GET'])
def calculate_fractal_channel():
    """Fractal Channel with optimized TVLC conversion"""
    try:
        symbol = request.args.get('symbol', 'BTCUSDT')
        interval = request.args.get('interval', '1h')
        limit_param = request.args.get('limit')
        limit = int(limit_param) if limit_param else None
        
        start_dt, end_dt, err = validate_candle_request_params(
            request.args.get('startDateTime'), 
            request.args.get('endDateTime')
        )
        if err: return err

        df = asyncio.run(_fetch_and_prepare_data(symbol, interval, start_dt, end_dt, limit))
        if df is None or df.empty:
            return jsonify({'success': False, 'error': f'No data found for {symbol}'}), 404

        fc = FractalChannel(
            mode='geometric', pivot_method='fractal', zigzag_dev=0.05, pivot_window=5, lookback=150
        )

        df_result = fc.calculate(df)
        plot_data = fc._get_plot_trace(df_result)

        # Uses the new utility to auto-convert Plotly traces to TVLC
        tvlc_data = convert_indicator_output_to_tvlc(df, plot_data)
        plot_data_normalized = normalize_plotly_trace_for_json(plot_data)

        return jsonify({
            'success': True, 'symbol': symbol, 'interval': interval,
            'plot_data': plot_data_normalized,
            'tvlc_data': tvlc_data
        })

    except Exception as e:
        logger.error(f"Error in fractal route: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@standard_indicator_bp.route('/api/rsi/calculate', methods=['GET'])
def calculate_rsi():
    try:
        symbol = request.args.get('symbol', 'BTCUSDT')
        interval = request.args.get('interval', '1h')
        limit_param = request.args.get('limit')
        limit = int(limit_param) if limit_param else None

        length = int(request.args.get('length', 14))
        source_col = request.args.get('source', 'close')
        fc_enabled = request.args.get('fc_enabled', 'true').lower() == 'true'
        fc_lookback = int(request.args.get('fc_lookback', 50))
        fc_mult = float(request.args.get('fc_mult', 2.0))
        fc_zigzag_dev = float(request.args.get('fc_zigzag_dev', 0.05))

        start_dt, end_dt, err = validate_candle_request_params(
            request.args.get('startDateTime'), 
            request.args.get('endDateTime')
        )
        if err: return err

        df = asyncio.run(_fetch_and_prepare_data(symbol, interval, start_dt, end_dt, limit))
        if df is None or df.empty:
            return jsonify({'success': False, 'error': f'No data found for {symbol}'}), 404

        rsi = RSI(
            length=length, source=source_col, fc_enabled=fc_enabled,
            fc_lookback=fc_lookback, fc_mult=fc_mult, fc_zigzag_dev=fc_zigzag_dev
        )

        df_result = rsi.calculate(df)
        
        # Manual TVLC construction for complex custom display
        tvlc_data = {'main': [], 'lines': [], 'markers': []}

        # RSI Line
        rsi_col = f'rsi_{length}_{source_col}'
        tvlc_data['main'] = [{'time': int(ts.timestamp()), 'value': val} 
                            for ts, val in df_result[rsi_col].dropna().items()]

        def add_line(name, color, data_col, style=0, width=1):
            data = [{'time': int(ts.timestamp()), 'value': val} 
                   for ts, val in df_result[data_col].dropna().items()]
            if data:
                tvlc_data['lines'].append({
                    'name': name, 'color': color, 'lineWidth': width, 'lineStyle': style, 'data': data
                })

        if fc_enabled:
            add_line('RSI Channel Upper', 'rgba(0, 255, 0, 0.5)', f'fc_upper_{fc_lookback}_dynamic', style=2)
            add_line('RSI Channel Lower', 'rgba(255, 0, 0, 0.5)', f'fc_lower_{fc_lookback}_dynamic', style=2)
            add_line('RSI Channel Mid', 'gray', f'fc_mid_{fc_lookback}_dynamic')

            if 'fc_signal' in df_result.columns:
                for idx, row in df_result.iterrows():
                    sig = row['fc_signal']
                    if sig == 1:
                        tvlc_data['markers'].append({
                            'time': int(idx.timestamp()), 'position': 'belowBar', 'color': 'lime', 'shape': 'arrowUp', 'text': 'Buy'
                        })
                    elif sig == -1:
                        tvlc_data['markers'].append({
                            'time': int(idx.timestamp()), 'position': 'aboveBar', 'color': 'red', 'shape': 'arrowDown', 'text': 'Sell'
                        })

        # Static Bounds
        ref_times = [int(ts.timestamp()) for ts in df_result.index]
        tvlc_data['lines'].append({
            'name': 'Overbought', 'color': 'rgba(255, 255, 255, 0.3)', 'lineStyle': 1, 'lineWidth': 1,
            'data': [{'time': t, 'value': 70} for t in ref_times]
        })
        tvlc_data['lines'].append({
            'name': 'Oversold', 'color': 'rgba(255, 255, 255, 0.3)', 'lineStyle': 1, 'lineWidth': 1,
            'data': [{'time': t, 'value': 30} for t in ref_times]
        })

        return jsonify({'success': True, 'tvlc_data': tvlc_data})

    except Exception as e:
        logger.error(f"Error in RSI calculation: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500
