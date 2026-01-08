import logging
from typing import Dict, List
from dataclasses import dataclass
from app.execution.execution_router import ExecutionRouter

@dataclass
class SafetyConfig:
    max_daily_drawdown_percent: float = 5.0
    max_total_drawdown_percent: float = 10.0
    max_open_positions: int = 5
    max_consecutive_losses: int = 3
    kill_switch_enabled: bool = False

class SafetyService:
    """
    Monitors account health and enforces safety limits (Circuit Breakers).
    """
    def __init__(self, execution_router: ExecutionRouter, config: SafetyConfig = None):
        self.execution = execution_router
        self.config = config if config else SafetyConfig()
        self.logger = logging.getLogger("app")
        self._daily_starting_balance = {} # asset -> balance

    async def initialize(self):
        # Snapshot starting balance for drawdown calculation
        # This is simplified; ideally we persist this or reset at 00:00 UTC
        # For now, we just snapshot on startup
        self._daily_starting_balance['USDT'] = await self.execution.get_balance_for_symbol("BTCUSDT") # Proxy

    async def check_health(self, symbol: str) -> bool:
        """
        Run safety checks. Returns True if safe to trade, False if circuit breaker tripped.
        """
        if self.config.kill_switch_enabled:
            self.logger.warning("Safety: Kill Switch is ENABLED. Trading halted.")
            return False

        # Check Max Open Positions
        # This requires iterating all assets or asking ExecutionRouter
        # For now, skipped as ExecutionRouter doesn't have 'get_all_open_positions_count' easily
        
        # Check Drawdown
        current_balance = await self.execution.get_balance_for_symbol(symbol)
        start_balance = self._daily_starting_balance.get('USDT', current_balance)
        
        if start_balance > 0:
            drawdown = (start_balance - current_balance) / start_balance * 100
            if drawdown > self.config.max_daily_drawdown_percent:
                self.logger.error(f"Safety: Max Daily Drawdown exceeded ({drawdown:.2f}% > {self.config.max_daily_drawdown_percent}%). Halted.")
                return False

        return True

    async def emergency_close_all(self, symbol: str = None):
        """
        Panic button: Close all positions.
        """
        self.logger.warning(f"Safety: EMERGENCY CLOSE TRIGGERED for {symbol if symbol else 'ALL'}")
        if symbol:
            await self.execution.cancel_all_orders(symbol)
            # Logic to close position (Market Sell)
            pos = await self.execution.get_position(symbol)
            amt = float(pos.get('positionAmt', 0))
            if amt != 0:
                side = "SELL" if amt > 0 else "BUY"
                await self.execution.place_order(symbol, side, "MARKET", abs(amt), reduceOnly=True)
        else:
            # Close everything (requires list of symbols)
            pass
