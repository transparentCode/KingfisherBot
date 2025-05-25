import json

import plotly
from flask import Blueprint, request, jsonify, render_template
import inspect

from app.exchange import BinanceConnector
from app.services.indicator_registers import IndicatorRegisters

indicator_bp = Blueprint('indicator', __name__)

indicator_register = IndicatorRegisters()


@indicator_bp.route('/api/indicators', methods=['GET'])
def get_registered_indicators():
    """Return a list of all registered indicators"""
    indicators_list = []
    for indicator_id, info in indicator_register.registered_indicators.items():
        indicators_list.append({
            'id': indicator_id,
            'name': info['display_name'],
            'description': info['description']
        })
    return jsonify({
        'success': True,
        'indicators': indicators_list
    })


@indicator_bp.route('/api/indicator/<indicator_id>', methods=['GET'])
def get_indicator(indicator_id):
    """Calculate and return indicator data by ID"""
    try:
        # Check if indicator exists
        if indicator_id not in indicator_register.registered_indicators:
            return jsonify({
                'success': False,
                'error': f'Indicator {indicator_id} not found'
            })

        # Get indicator info
        indicator_info = indicator_register.registered_indicators[indicator_id]
        indicator_class = indicator_info['class']

        # Get parameters from request
        symbol = request.args.get('symbol', 'BTCUSDT')
        interval = request.args.get('interval', '15m')
        lookback_days = int(request.args.get('lookback_days', 30))

        # Get data from exchange
        binance_connector = BinanceConnector()
        data = binance_connector.get_historical_data(
            symbol=symbol,
            interval=interval,
            lookback_days=lookback_days
        )

        # Get the signature of the indicator's __init__ method
        sig = inspect.signature(indicator_class.__init__)
        params = {
            k: v.default for k, v in sig.parameters.items()
            if v.default != inspect.Parameter.empty and k != 'self'
        }

        # Update with request parameters
        for param_name in params:
            param_type = type(params[param_name])
            if request.args.get(param_name):
                if param_type == bool:
                    params[param_name] = request.args.get(param_name).lower() == 'true'
                else:
                    params[param_name] = param_type(request.args.get(param_name))

        indicator = indicator_class(**params)

        # Calculate indicator values
        result = indicator.calculate(data)

        # Generate plot
        fig = indicator.plot(result)

        plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        plot_data = json.loads(plot_json)

        return jsonify({
            'success': True,
            'indicator_id': indicator_id,
            'indicator_name': indicator_info['display_name'],
            'symbol': symbol,
            'interval': interval,
            'parameters': params,
            'plot_data': plot_data,
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@indicator_bp.route('/indicator/view', methods=['GET'])
def view_backtest():
    return render_template('indicators.html')


@indicator_bp.route('/api/indicator/params/<indicator_id>', methods=['GET'])
def get_indicator_params(indicator_id):
    """Get the parameters for a specific indicator"""
    try:
        if indicator_id not in indicator_register.registered_indicators:
            return jsonify({
                'success': False,
                'error': f'Indicator {indicator_id} not found'
            })

        indicator_class = indicator_register.registered_indicators[indicator_id]['class']

        # Get the signature of the indicator's __init__ method
        sig = inspect.signature(indicator_class.__init__)
        params = {}

        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue

            param_info = {
                'name': param_name,
                'required': param.default == inspect.Parameter.empty
            }

            # Add default value if available
            if param.default != inspect.Parameter.empty:
                param_info['default'] = param.default
                param_info['type'] = type(param.default).__name__

            # Add to params dictionary
            params[param_name] = param_info

        return jsonify({
            'success': True,
            'indicator_id': indicator_id,
            'indicator_name': indicator_register.registered_indicators[indicator_id]['display_name'],
            'parameters': params
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })
