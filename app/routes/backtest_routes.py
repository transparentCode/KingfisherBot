# In your Flask routes file (e.g., app/routes/backtest_routes.py)

from flask import Blueprint, jsonify, request, render_template
import plotly
import json

from app.factory.startegy_factory import StrategyFactory
from app.backtest.backtest_engine import VectorBTBacktest
from app.exchange.BinanceConnector import BinanceConnector

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
        factory = StrategyFactory()
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

        return jsonify({
            'success': True,
            'plot_data': plot_data,
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