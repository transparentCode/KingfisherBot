import logging
import json
import uuid
from typing import Dict, List
from app.execution.base_execution import BaseExecutionService
from app.db.redis_handler import RedisHandler

class PaperExecutionService(BaseExecutionService):
    def __init__(self):
        self.redis = RedisHandler()
        self.logger = logging.getLogger("app")
        self.initial_balance = 10000.0  # Default paper balance

    async def _get_redis(self):
        if not self.redis.redis_client:
            await self.redis.initialize()
        return self.redis.redis_client

    async def get_balance(self, asset: str = "USDT") -> float:
        r = await self._get_redis()
        bal = await r.get(f"paper:balance:{asset}")
        if bal is None:
            await r.set(f"paper:balance:{asset}", self.initial_balance)
            return self.initial_balance
        return float(bal)

    async def get_position(self, symbol: str) -> Dict:
        r = await self._get_redis()
        pos_data = await r.get(f"paper:position:{symbol}")
        if pos_data:
            return json.loads(pos_data)
        return {
            "symbol": symbol,
            "positionAmt": "0.0",
            "entryPrice": "0.0",
            "unRealizedProfit": "0.0",
            "leverage": "1"
        }

    async def get_all_positions(self) -> List[Dict]:
        r = await self._get_redis()
        keys = await r.keys("paper:position:*")
        positions = []
        for key in keys:
            data = await r.get(key)
            if data:
                pos = json.loads(data)
                if float(pos['positionAmt']) != 0:
                    positions.append(pos)
        return positions

    async def place_order(self, symbol: str, side: str, order_type: str, quantity: float, price: float = None, **kwargs) -> Dict:
        # SIMPLIFIED: Fills immediately.
        # We assume the caller passes 'current_price' in kwargs if it's a market order.
        
        exec_price = price if price else kwargs.get('current_price', 0.0)
        if exec_price == 0.0:
             self.logger.warning(f"PaperTrade: No price provided for {symbol} {side}. Using 0.0")

        r = await self._get_redis()
        
        # Update Position
        current_pos = await self.get_position(symbol)
        curr_amt = float(current_pos['positionAmt'])
        curr_entry = float(current_pos['entryPrice'])
        
        qty = quantity if side == 'BUY' else -quantity
        new_amt = curr_amt + qty
        
        # Calculate new entry price (Weighted Average)
        new_entry = curr_entry
        pnl = 0.0
        
        if new_amt == 0:
            # Closed completely
            new_entry = 0.0
            # PnL on the whole amount
            pnl = (exec_price - curr_entry) * abs(curr_amt) * (1 if curr_amt > 0 else -1)
            
        elif (curr_amt > 0 and qty > 0) or (curr_amt < 0 and qty < 0):
            # Increasing position
            total_cost = (curr_amt * curr_entry) + (qty * exec_price)
            new_entry = total_cost / new_amt
            
        elif (curr_amt > 0 > qty) or (curr_amt < 0 < qty):
            # Reducing position (Realize PnL)
            # Entry price doesn't change when reducing
            new_entry = curr_entry
            # Calculate PnL (Realized)
            pnl = (exec_price - curr_entry) * abs(qty) * (1 if curr_amt > 0 else -1)
            
        else:
            # Flip position (Long -> Short or Short -> Long)
            # 1. Close existing
            pnl = (exec_price - curr_entry) * abs(curr_amt) * (1 if curr_amt > 0 else -1)
            # 2. Open new
            new_entry = exec_price

        # Update Balance if PnL realized
        if pnl != 0:
            curr_bal = await self.get_balance("USDT")
            await r.set("paper:balance:USDT", curr_bal + pnl)

        new_pos = {
            "symbol": symbol,
            "positionAmt": str(new_amt),
            "entryPrice": str(new_entry),
            "unRealizedProfit": "0.0", # Needs live price to update
            "leverage": kwargs.get('leverage', current_pos.get('leverage', '1'))
        }
        
        await r.set(f"paper:position:{symbol}", json.dumps(new_pos))
        
        return {
            "symbol": symbol,
            "orderId": str(uuid.uuid4()),
            "status": "FILLED",
            "price": str(exec_price),
            "avgPrice": str(exec_price),
            "executedQty": str(quantity),
            "side": side,
            "type": order_type
        }

    async def cancel_order(self, symbol: str, order_id: str) -> Dict:
        return {"status": "CANCELED", "orderId": order_id}

    async def cancel_all_orders(self, symbol: str) -> List[Dict]:
        return []
