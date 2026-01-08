import logging
from typing import Dict, Any, Optional
import pandas as pd

from app.brain.bot_brain import BotBrain
from app.db.redis_handler import RedisHandler
from app.brain.models import TradeSignal

class BotBrainProcessor:
    """
    Processor that wraps the BotBrain to run within the IndicatorPipeline.
    
    Responsibilities:
    1. Extract MTF data from results/context
    2. Run BotBrain.analyze_market()
    3. Inject TradeSignal into results for ExecutionProcessor
    4. Cache signal to Redis for UI/API
    """
    
    def __init__(self, redis_handler: Optional[RedisHandler] = None, db_pool = None):
        self.logger = logging.getLogger("app")
        # Initialize BotBrain
        self.brain = BotBrain(redis_handler=redis_handler, db_pool=db_pool)
        self.redis_handler = redis_handler

    async def process_results(self, asset: str, all_results: Dict[str, Any]):
        """
        Execute BotBrain analysis.
        
        Args:
            asset: Symbol name
            all_results: Dict containing indicator results AND 'context' (MTF data)
                         Note: We need to ensure 'context' or 'mtf_data' is passed here.
                         The IndicatorPipeline calls process_results(context.asset, all_results).
                         But IndicatorExecutionContext isn't passed directly!
                         
                         Workaround: We need to access the data cache.
                         In IndicatorCalcService, 'valid_timeframes' is the data cache.
                         But the pipeline only passes 'all_results'.
                         
                         FIX: We will likely need to inject the data cache into 'all_results' 
                         or modify the pipeline to pass context.
                         
                         For now, let's assume 'all_results' contains a special key '__mtf_data__' 
                         or we change IndicatorPipeline to inject it.
        """
        try:
            # 1. Retrieve MTF Data
            # Check if '__mtf_data__' exists (we'll need to modify pipeline or service to inject this)
            mtf_data = all_results.get('__mtf_data__')
            
            if not mtf_data:
                # Attempt to reconstruct from results if 'data' is present in indicator outputs
                # This is risky/hacky. Better to fix the service.
                self.logger.warning(f"No MTF data available for BotBrain for {asset}")
                return

            self.logger.info(f"BotBrain analyzing {asset}...")
            
            # 2. Run Brain
            signal: Optional[TradeSignal] = await self.brain.analyze_market(asset, mtf_data)
            
            if signal:
                self.logger.info(f"BotBrain GENERATED SIGNAL: {signal.direction} {signal.setup_type} (Conf: {signal.confidence:.2f})")
                
                # 3. Inject into results for ExecutionProcessor
                # ExecutionProcessor looks for results['final_signal']
                all_results['final_signal'] = {
                    'action': 'BUY' if signal.direction == 'LONG' else 'SELL',
                    'confidence': signal.confidence,
                    'signal_object': signal,  # Pass full object for detailed logs/risk
                    'reasoning': signal.reasoning
                }
                
                # 4. Cache to Redis for UI
                if self.redis_handler:
                    await self.redis_handler.set_key(
                        f"signals:bot_brain:{asset}", 
                        signal.to_dict(),
                        ttl=300 # 5 min TTL
                    )
            else:
                self.logger.debug(f"BotBrain: No signal for {asset}")
                
        except Exception as e:
            self.logger.error(f"Error in BotBrainProcessor for {asset}: {e}", exc_info=True)
