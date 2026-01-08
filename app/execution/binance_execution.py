import logging
import asyncio
from typing import Dict, List, Optional
from app.execution.base_execution import BaseExecutionService
from app.exchange.BinanceConnector import BinanceConnector

class BinanceExecutionService(BaseExecutionService):
    """
    Live execution service using BinanceConnector.
    """

    def __init__(self, connector: BinanceConnector):
        self.connector = connector
        self.logger = logging.getLogger("app")

    async def get_balance(self, asset: str = "USDT") -> float:
        account_info = await asyncio.to_thread(self.connector.get_account_info)
        for asset_balance in account_info.get('assets', []):
            if asset_balance['asset'] == asset:
                return float(asset_balance['availableBalance'])
        return 0.0

    async def get_position(self, symbol: str) -> Dict:
        positions = await asyncio.to_thread(self.connector.get_position_risk, symbol=symbol)
        # Binance returns a list for get_position_risk even for single symbol
        if positions and isinstance(positions, list):
            for pos in positions:
                if pos['symbol'] == symbol:
                    return pos
        return {}

    async def get_all_positions(self) -> List[Dict]:
        # Filter for non-zero positions
        all_positions = await asyncio.to_thread(self.connector.get_position_risk)
        return [p for p in all_positions if float(p['positionAmt']) != 0]

    async def place_order(self, symbol: str, side: str, order_type: str, quantity: float, price: float = None, **kwargs) -> Dict:
        return await asyncio.to_thread(
            self.connector.place_order, 
            symbol, side, order_type, quantity, price, **kwargs
        )

    async def cancel_order(self, symbol: str, order_id: str) -> Dict:
        return await asyncio.to_thread(
            self.connector.cancel_order, 
            symbol, int(order_id)
        )

    async def cancel_all_orders(self, symbol: str) -> List[Dict]:
        return await asyncio.to_thread(
            self.connector.cancel_all_orders, 
            symbol
        )
