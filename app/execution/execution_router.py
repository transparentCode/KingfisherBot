import logging
from typing import Dict, List
from app.execution.base_execution import BaseExecutionService
from app.execution.binance_execution import BinanceExecutionService
from app.execution.paper_execution import PaperExecutionService
from config.asset_indicator_config import ConfigurationManager

class ExecutionRouter:
    def __init__(self, binance_service: BinanceExecutionService, paper_service: PaperExecutionService):
        self.binance = binance_service
        self.paper = paper_service
        self.config_manager = ConfigurationManager()
        self.logger = logging.getLogger("app")

    def _get_service(self, symbol: str) -> BaseExecutionService:
        # Extract asset from symbol (e.g. BTCUSDT -> BTC)
        asset = symbol.replace("USDT", "") 
        
        # Check config
        # Access the dictionary directly as ConfigurationManager is a singleton
        asset_config = self.config_manager.asset_configs.get(asset)
        
        if asset_config:
            if getattr(asset_config, 'paper_mode', True):
                return self.paper
            else:
                return self.binance
        
        # Default to Paper if no config found
        return self.paper

    async def get_balance_for_symbol(self, symbol: str) -> float:
        """
        Get the appropriate balance (Live or Paper) for trading the given symbol.
        """
        service = self._get_service(symbol)
        return await service.get_balance("USDT")

    async def get_position(self, symbol: str) -> Dict:
        service = self._get_service(symbol)
        return await service.get_position(symbol)

    async def place_order(self, symbol: str, side: str, order_type: str, quantity: float, price: float = None, **kwargs) -> Dict:
        service = self._get_service(symbol)
        self.logger.info(f"Routing order for {symbol} to {type(service).__name__}")
        return await service.place_order(symbol, side, order_type, quantity, price, **kwargs)

    async def cancel_order(self, symbol: str, order_id: str) -> Dict:
        service = self._get_service(symbol)
        return await service.cancel_order(symbol, order_id)
        
    async def cancel_all_orders(self, symbol: str) -> List[Dict]:
        service = self._get_service(symbol)
        return await service.cancel_all_orders(symbol)
