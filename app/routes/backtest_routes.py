# In your Flask routes file (e.g., app/routes/backtest_routes.py)

from flask import Blueprint, jsonify, request, render_template
import plotly
import json

from app.backtest.backtest_engine import VectorBTBacktest
from app.exchange.BinanceConnector import BinanceConnector
from app.orchestration.strategy.strategy_orchestrator import StrategyFactory

backtest_bp = Blueprint('backtest', __name__)


@backtest_bp.route('/api/backtest', methods=['POST'])
def run_backtest():
    """API endpoint to run backtest and return results"""
    data = request.json

    # Extract parameters
    symbol = data.get('symbol', 'BTCUSDT')
    interval = data.get('interval', '5m')
    lookback_days = data.get('lookback_days', 30)
    strategy_id = data.get('strategy_id', 'sma_confluence')
    strategy_params = data.get('strategy_params', {})

    # Get historical data
    connector = BinanceConnector()
    df = connector.get_historical_data(symbol, interval, lookback_days)

    # Create and execute strategy
    factory = StrategyFactory()
    strategy = factory.create_strategy(strategy_id, **strategy_params)
    strategy_results = strategy.execute(df)

    # Run backtest
    backtest = VectorBTBacktest(strategy)
    portfolio = backtest.run_backtest(strategy_results)

    # Get performance metrics
    metrics = backtest.get_performance_metrics()

    # Return JSON results
    return jsonify({
        'metrics': metrics,
        'plot_endpoint': f'/api/backtest/plot?strategy_id={strategy_id}&symbol={symbol}&interval={interval}'
    })


@backtest_bp.route('/api/backtest/plot', methods=['GET'])
def get_backtest_plot():
    """API endpoint to return plotly visualization"""
    try:
        # Get parameters from request
        strategy_id = request.args.get('strategy_id', 'sma_confluence')
        symbol = request.args.get('symbol', 'BTCUSDT')
        interval = request.args.get('interval', '4h')
        lookback_days = int(request.args.get('lookback_days', 30))

        # Get historical data
        connector = BinanceConnector()
        df = connector.get_historical_data(symbol, interval, lookback_days)

        # Create and execute strategy
        factory = None
        strategy = factory.create_strategy(strategy_id)
        strategy_results = strategy.execute(df)

        # Run backtest
        backtest = VectorBTBacktest(strategy)
        backtest.run_backtest(strategy_results)

        # Generate the plot
        fig = backtest.plot_results()

        # Convert the plot to JSON
        plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        plot_data = json.loads(plot_json)

        # --- TVLC FORMATTING START ---
        tvlc_data = {
            'candles': [],
            'lines': [],
            'markers': []
        }
        
        # Backtest results usually contain price data and equity curve
        # We can extract price from the input df
        for idx, row in df.iterrows():
            tvlc_data['candles'].append({
                'time': int(idx.timestamp()),
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close']
            })

        # Extract Equity Curve from Plotly data if possible, or from backtest object
        # VectorBT plot_results usually puts Equity in a scatter trace
        if isinstance(plot_data, dict) and 'data' in plot_data:
            for trace in plot_data['data']:
                name = trace.get('name', '')
                if 'Equity' in name or 'Value' in name:
                    x_vals = trace.get('x', [])
                    y_vals = trace.get('y', [])
                    
                    line_series = {
                        'name': 'Equity',
                        'color': '#2962FF',
                        'lineWidth': 2,
                        'priceScaleId': 'equity', # Separate scale for equity
                        'data': []
                    }
                    
                    for i, x in enumerate(x_vals):
                        try:
                            ts = int(pd.Timestamp(x).timestamp())
                            val = y_vals[i]
                            if val is not None:
                                line_series['data'].append({'time': ts, 'value': val})
                        except:
                            continue
                            
                    if line_series['data']:
                        tvlc_data['lines'].append(line_series)
                        
                # Extract Buy/Sell Signals
                if 'Buy' in name or 'Sell' in name:
                    color = 'lime' if 'Buy' in name else 'red'
                    position = 'belowBar' if 'Buy' in name else 'aboveBar'
                    shape = 'arrowUp' if 'Buy' in name else 'arrowDown'
                    
                    x_vals = trace.get('x', [])
                    y_vals = trace.get('y', [])
                    
                    for i, x in enumerate(x_vals):
                        try:
                            ts = int(pd.Timestamp(x).timestamp())
                            val = y_vals[i]
                            if val is not None:
                                tvlc_data['markers'].append({
                                    'time': ts,
                                    'position': position,
                                    'color': color,
                                    'shape': shape,
                                    'text': name[:1]
                                })
                        except:
                            continue
        # --- TVLC FORMATTING END ---

        return jsonify({
            'success': True,
            'plot_data': plot_data,
            'tvlc_data': tvlc_data, # New field
            'symbol': symbol,
            'interval': interval,
            'strategy_id': strategy_id
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# HTML page to display the interactive chart
@backtest_bp.route('/backtest/view', methods=['GET'])
def view_backtest():
    return render_template('backtest_view.html')


@backtest_bp.route('/api/strategy/plot', methods=['GET'])
def get_strategy_plot():
    """API endpoint to generate a strategy-specific chart view"""
    try:
        # Get parameters from request
        strategy_id = request.args.get('strategy_id', 'sma_confluence')
        symbol = request.args.get('symbol', 'BTCUSDT')
        interval = request.args.get('interval', '5m')
        lookback_days = int(request.args.get('lookback_days', 30))

        # Parse strategy params if provided
        strategy_params = {}
        for key, value in request.args.items():
            if key.startswith('param_'):
                param_name = key.replace('param_', '')
                try:
                    # Try to convert to appropriate type
                    if '.' in value:
                        strategy_params[param_name] = float(value)
                    else:
                        strategy_params[param_name] = int(value)
                except ValueError:
                    strategy_params[param_name] = value

        # Get historical data
        connector = BinanceConnector()
        df = connector.get_historical_data(symbol, interval, lookback_days)

        # Create and execute strategy
        factory = StrategyFactory()
        strategy = factory.create_strategy(strategy_id, **strategy_params)
        strategy_results = strategy.execute(df)

        # Generate strategy-specific plot
        fig = strategy.plot(strategy_results)

        # Convert the plot to JSON
        plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        plot_data = json.loads(plot_json)

        return jsonify({
            'success': True,
            'plot_data': plot_data,
            'symbol': symbol,
            'interval': interval,
            'strategy_id': strategy_id,
            'parameters': strategy.get_parameters()
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500