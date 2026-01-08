import logging
from typing import Dict, Any, List
from app.orchestration.indicator.indicator_orchestrator import BaseIndicatorOrchestrator
from app.models.indicator_context import IndicatorExecutionContext
from app.services.strategy_registry import StrategyRegistry
from app.strategy.models import StrategySignal, AggregatedSignal

# New Brain Components
from app.signals.trend_factor import TrendFactor
from app.signals.momentum_factor import MomentumFactor
from app.signals.volume_factor import VolumeFactor
from app.brain.confluence_engine import ConfluenceEngine
from app.enums.MarketRegime import MarketRegime

class StrategyOrchestrator(BaseIndicatorOrchestrator):
    """
    The Brain: Orchestrates Signal Factors and runs the Confluence Engine.
    """
    
    def __init__(self, indicator_registry, config_manager, logger_name="app"):
        super().__init__(indicator_registry, logger_name, config_manager)
        self.registry = StrategyRegistry()
        self.confluence_engine = ConfluenceEngine()
        
        # Initialize Factors (Layer A)
        # In a real system, these would be loaded dynamically from config
        self.factors = {
            'TrendFactor': TrendFactor(),
            'MomentumFactor': MomentumFactor(),
            'VolumeFactor': VolumeFactor()
        }

    async def execute(self, context: IndicatorExecutionContext) -> Dict[str, Any]:
        if not self._validate_context(context):
            return {}

        asset = context.asset
        
        # 1. Get Data (Dual-Layer MTF)
        # Tactical Layer (15m) - For Signals
        tactical_tf = context.primary_timeframe # e.g. 15m
        tactical_df = context.data_cache.get(tactical_tf)
        
        # Strategic Layer (1H/4H) - For Regime
        # We need to find a higher timeframe. 
        # Assuming context has it or we pick the next one up.
        strategic_tf = "1h" # Default
        if tactical_tf == "1h": strategic_tf = "4h"
        if tactical_tf == "4h": strategic_tf = "1d"
        
        strategic_df = context.data_cache.get(strategic_tf)
        
        if tactical_df is None or tactical_df.empty:
            self.logger.warning(f"No tactical data for {asset} {tactical_tf}")
            return {}

        # 2. Get Regime Metrics (Layer B Input)
        # We fetch this from context.metadata which is populated by IndicatorCalcService
        regime_data = context.metadata.get('regime', {})
        
        regime_metrics = {
            'hurst': float(regime_data.get('hurst', 0.5)),
            'regime': regime_data.get('regime', MarketRegime.CONSOLIDATION),
            'volatility': float(regime_data.get('volatility', 0.0))
        }
        
        self.logger.debug(f"Using Regime Metrics for {asset}: {regime_metrics}")
        
        # 3. Calculate Signals (Layer A)
        signal_scores = {}
        for name, factor in self.factors.items():
            try:
                # Initialize with params from config if needed
                # factor.initialize(**params)
                score = factor.calculate(tactical_df)
                signal_scores[name] = score
                self.logger.debug(f"Factor {name}: {score}")
            except Exception as e:
                self.logger.error(f"Error calculating factor {name}: {e}")
                signal_scores[name] = 0.0

        # 4. Confluence Engine (Layer B)
        final_score = self.confluence_engine.calculate_score(signal_scores, regime_metrics)
        
        # 5. Decision (Layer C)
        # Thresholds
        buy_threshold = 0.6
        sell_threshold = -0.6
        
        action = "NEUTRAL"
        if final_score > buy_threshold:
            action = "BUY"
        elif final_score < sell_threshold:
            action = "SELL"
            
        confidence = abs(final_score)
        
        self.logger.info(f"Brain Result for {asset}: {action} (Score: {final_score:.2f})")

        return {
            "final_signal": {
                "action": action,
                "confidence": confidence,
                "score": final_score,
                "factors": signal_scores,
                "regime": str(regime_metrics['regime']),
                "current_price": tactical_df['close'].iloc[-1]
            }
        }


class StrategyFactory:
    """
    Factory to create strategy instances from the registry.
    Used by backtesting and other components that need to instantiate strategies by name.
    """
    def __init__(self):
        self.registry = StrategyRegistry()

    def create_strategy(self, strategy_id: str, **kwargs):
        strategy_class = self.registry.get_strategy_class(strategy_id)
        if not strategy_class:
            # Try case-insensitive lookup
            all_strategies = self.registry.get_all_strategies()
            for name, cls in all_strategies.items():
                if name.lower() == strategy_id.lower():
                    strategy_class = cls
                    break
            
            if not strategy_class:
                raise ValueError(f"Strategy {strategy_id} not found in registry")
        
        return strategy_class(**kwargs)

