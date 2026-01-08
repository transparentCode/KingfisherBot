from typing import Dict, Type, Optional
import logging
from app.strategy.base_strategy_interface import BaseStrategyInterface

class StrategyRegistry:
    """
    Singleton registry for trading strategies.
    """
    _instance = None
    _strategies: Dict[str, Type[BaseStrategyInterface]] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(StrategyRegistry, cls).__new__(cls)
            cls._instance.logger = logging.getLogger("app")
        return cls._instance

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a strategy class.
        @StrategyRegistry.register("MyStrategy")
        class MyStrategy(BaseStrategyInterface): ...
        """
        def decorator(strategy_class: Type[BaseStrategyInterface]):
            cls._strategies[name] = strategy_class
            return strategy_class
        return decorator

    @classmethod
    def register_strategy(cls, name: str, strategy_class: Type[BaseStrategyInterface]):
        """Manual registration"""
        cls._strategies[name] = strategy_class

    def get_strategy_class(self, name: str) -> Optional[Type[BaseStrategyInterface]]:
        return self._strategies.get(name)

    def get_all_strategies(self) -> Dict[str, Type[BaseStrategyInterface]]:
        return self._strategies

    @classmethod
    def register_default_strategies(cls):
        """Register built-in strategies."""
        from app.strategy.rsi_strategy import RSIStrategy
        from app.strategy.sma_confluence import SmaConfluence
        
        cls.register_strategy("RSIStrategy", RSIStrategy)
        cls.register_strategy("SmaConfluence", SmaConfluence)
        # Add aliases if needed
        cls.register_strategy("rsi_strategy", RSIStrategy)
        cls.register_strategy("sma_confluence", SmaConfluence)
