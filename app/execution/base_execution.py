from abc import ABC, abstractmethod
from typing import Dict, List, Optional

class BaseExecutionService(ABC):
    """
    Abstract base class for execution services (Live vs Paper).
    """

    @abstractmethod
    async def get_balance(self, asset: str = "USDT") -> float:
        """Get available balance for the quote asset."""
        pass

    @abstractmethod
    async def get_position(self, symbol: str) -> Dict:
        """
        Get current position for a symbol.
        Returns dict with keys: symbol, positionAmt, entryPrice, unRealizedProfit, leverage
        """
        pass

    @abstractmethod
    async def get_all_positions(self) -> List[Dict]:
        """Get all open positions."""
        pass

    @abstractmethod
    async def place_order(self, symbol: str, side: str, order_type: str, quantity: float, price: float = None, **kwargs) -> Dict:
        """
        Place an order.
        Returns order response dict.
        """
        pass

    @abstractmethod
    async def cancel_order(self, symbol: str, order_id: str) -> Dict:
        """Cancel a specific order."""
        pass

    @abstractmethod
    async def cancel_all_orders(self, symbol: str) -> List[Dict]:
        """Cancel all open orders for a symbol."""
        pass
