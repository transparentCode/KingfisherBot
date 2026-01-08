import logging
from typing import Dict, Any
from app.execution.execution_router import ExecutionRouter
from app.services.safety_service import SafetyService
from app.risk.risk_manager import RiskManager
from app.risk.models import TradePlan

class ExecutionProcessor:
    """
    Processor that takes signals, calculates risk, and executes trades.
    """
    def __init__(self, execution_router: ExecutionRouter, safety_service: SafetyService, risk_manager: RiskManager):
        self.execution = execution_router
        self.safety = safety_service
        self.risk_manager = risk_manager
        self.logger = logging.getLogger("app")

    async def process_results(self, asset: str, results: Dict[str, Any]):
        """
        Process analysis results, check for signals, and execute trades.
        """
        # 1. Extract Signal
        # This depends on what the previous processors output.
        # Assuming 'aggregated_signal' or similar is present.
        # For now, let's look for a 'signal' key which might be 'BUY', 'SELL', or None.
        
        # Note: The SignalAggregationProcessor usually outputs a 'final_signal' or similar.
        # We need to standardize this. Let's assume 'final_signal' dict with 'action', 'confidence'.
        
        signal_data = results.get('final_signal')
        if not signal_data:
            return

        action = signal_data.get('action') # 'BUY', 'SELL', 'NEUTRAL'
        confidence = signal_data.get('confidence', 1.0)
        
        if action not in ['BUY', 'SELL']:
            return

        self.logger.info(f"ExecutionProcessor: Received {action} signal for {asset} (Confidence: {confidence})")

        # 2. Safety Check
        is_safe = await self.safety.check_health(asset)
        if not is_safe:
            self.logger.warning(f"Safety check failed for {asset}. Skipping trade.")
            return

        # 3. Risk Calculation
        # Get current price (close price from results)
        # We need the latest close price. 'market_data' might be in results or we fetch it.
        # Assuming results contains 'current_price' or we use the last close from the dataframe if available.
        
        current_price = results.get('current_price')
        if not current_price:
            # Fallback: try to find it in the dataframe if passed, or fetch from execution
            # For now, let's assume the pipeline puts 'current_price' in results.
            self.logger.warning(f"No current_price found for {asset}. Skipping execution.")
            return

        # Get Account Balance (Equity)
        # We use the execution router to get the balance for the specific mode (Paper/Live)
        account_balance = await self.execution.get_balance_for_symbol(asset)
        
        # Calculate Trade Plan
        try:
            trade_plan: TradePlan = self.risk_manager.calculate_trade_plan(
                symbol=asset,
                signal_type=action,
                entry_price=current_price,
                equity=account_balance,
                confidence_score=confidence
            )
        except Exception as e:
            self.logger.error(f"Error calculating trade plan for {asset}: {e}")
            return

        if not trade_plan:
            self.logger.info(f"Risk Manager returned no trade plan for {asset} (Risk too high or invalid).")
            return

        self.logger.info(f"Executing Trade Plan for {asset}: {trade_plan}")

        # 4. Execution
        # Place Entry Order
        try:
            # Market Order for Entry
            await self.execution.place_order(
                symbol=asset,
                side=action,
                order_type="MARKET",
                quantity=trade_plan.position_size,
                current_price=current_price # For paper trading simulation
            )
            
            # Place Stop Loss
            sl_side = "SELL" if action == "BUY" else "BUY"
            await self.execution.place_order(
                symbol=asset,
                side=sl_side,
                order_type="STOP_MARKET",
                quantity=trade_plan.position_size,
                stopPrice=trade_plan.stop_loss,
                reduceOnly=True
            )
            
            # Place Take Profits
            for tp in trade_plan.take_profits:
                await self.execution.place_order(
                    symbol=asset,
                    side=sl_side,
                    order_type="LIMIT",
                    quantity=tp.quantity,
                    price=tp.price,
                    reduceOnly=True
                )
                
        except Exception as e:
            self.logger.error(f"Execution failed for {asset}: {e}")
            # TODO: Implement rollback/cleanup if partial execution happens
