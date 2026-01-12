import numpy as np
import pandas as pd
from typing import List, Dict, Any

def format_candles_for_response(df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    """
    Format DataFrame candles into standard and TVLC compatible list of dictionaries.
    
    Args:
        df: DataFrame with DatetimeIndex and OHLCV columns (open, high, low, close, volume)
        
    Returns:
        Dictionary containing 'candles' and 'tvlc_candles' lists.
    """
    candles = []
    tvlc_candles = []
    
    if df.empty:
        return {"candles": [], "tvlc_candles": []}

    # Iterate efficiently
    # Note: timestamp() returns float, casting to int for standard seconds
    for timestamp, row in df.iterrows():
        # Prepare valid row data
        candle_data = {
            "time": int(timestamp.timestamp()),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
        }
        
        candles.append(candle_data)
        
        # TVLC format currently mimics standard, but kept separate for future divergence 
        # or specific formatting requirements of the TradingView Lightweight Charts library
        tvlc_candles.append(candle_data)
        
    return {
        "candles": candles,
        "tvlc_candles": tvlc_candles
    }


def convert_indicator_output_to_tvlc(df: pd.DataFrame, plot_data: Any) -> Dict[str, List[Dict[str, Any]]]:
    """
    Convert indicator calculation results (DataFrame + Plotly traces) into TVLC format.
    
    Args:
        df: DataFrame containing the indicator calculation context (candles + values)
        plot_data: Plotly trace object(s) returned by indicator._get_plot_trace()
        
    Returns:
        Dictionary with keys: 'candles', 'lines', 'markers', 'histograms'
    """
    tvlc_data = {
        'candles': [],
        'lines': [],
        'markers': [],
        'histograms': []
    }

    # A. Candles (Main Series)
    # We use the DF which should be aligned with the plot data
    for idx, row in df.iterrows():
        tvlc_data['candles'].append({
            'time': int(idx.timestamp()),
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close'])
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
        
        # Skip main candlestick if present (we handled it above)
        if trace_type == 'candlestick':
            continue

        x_vals = trace.get('x', [])
        y_vals = trace.get('y', [])
        
        if not len(x_vals) or not len(y_vals):
            continue

        # Convert timestamps
        try:
            # Handle string vs timestamp input
            if isinstance(x_vals[0], str):
                x_ts = [int(pd.Timestamp(x).timestamp()) for x in x_vals]
            else:
                x_ts = [int(pd.Timestamp(x).timestamp()) for x in x_vals]
        except Exception:
            continue

        # 1. Lines
        # Plotly 'lines' or plain scatter usually map to TVLC LineSeries
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
                    line_series['data'].append({'time': t, 'value': float(val)})
            
            if line_series['data']:
                tvlc_data['lines'].append(line_series)

        # 2. Markers
        # Plotly 'markers' map to TVLC Series Markers
        if 'markers' in mode:
            color = trace.get('marker', {}).get('color', '#2962FF')
            # Heuristic for position based on name/signal
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
                        'text': name[:1] # First letter as short label
                    })

        # 3. Bar/Histogram
        if trace_type == 'bar':
            hist_series = {
                'name': name,
                'color': trace.get('marker', {}).get('color', '#26a69a'),
                'data': []
            }
            for i, t in enumerate(x_ts):
                val = y_vals[i]
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    hist_series['data'].append({'time': t, 'value': float(val)})
            
            if hist_series['data']:
                tvlc_data['histograms'].append(hist_series)

    return tvlc_data


def normalize_plotly_trace_for_json(plot_data: Any) -> Any:
    """
    Normalize datetime objects in Plotly traces so they are JSON serializable (isoformat).
    Also replaces NaN/Infinity with None (JSON null).
    """
    import math

    def _sanitize_value(v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v

    def _normalize_trace(trace):
        # 1. Handle X (Datetime to ISO)
        x_vals = trace.get('x')
        if isinstance(x_vals, (list, tuple, np.ndarray)):
             trace['x'] = [x.isoformat() if hasattr(x, 'isoformat') else x for x in x_vals]
        
        # 2. Handle Y (NaN to None)
        y_vals = trace.get('y')
        if isinstance(y_vals, (list, tuple, np.ndarray)):
            trace['y'] = [_sanitize_value(y) for y in y_vals]
            
        return trace

    if isinstance(plot_data, list):
        return [_normalize_trace(t) for t in plot_data]
    elif isinstance(plot_data, dict) and 'data' in plot_data:
        plot_data_copy = plot_data.copy()
        plot_data_copy['data'] = [_normalize_trace(t) for t in plot_data['data']]
        return plot_data_copy
    return plot_data
