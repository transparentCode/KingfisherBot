from flask import Blueprint, jsonify
from app.orchestration.strategy.strategy_orchestrator import StrategyFactory

strategy_bp = Blueprint('strategy', __name__)

@strategy_bp.route('/api/strategies/available', methods=['GET'])
def get_available_strategies():
    """Get list of available strategies"""
    try:
        factory = StrategyFactory()
        # Get list of strategy IDs
        strategies = list(factory.strategies.keys())
        
        # In a real app, we might want more details (description, params)
        # For now, just return the IDs
        return jsonify({
            'success': True,
            'strategies': strategies
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@strategy_bp.route('/api/strategies/active', methods=['GET'])
def get_active_strategies():
    """Get list of currently active/deployed strategies"""
    # Placeholder: In the future, this would query the database or runtime state
    # to see which strategies are running on which assets.
    return jsonify({
        'success': True,
        'strategies': [] 
    })
