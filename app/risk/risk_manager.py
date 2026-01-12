import logging
import numpy as np
from typing import Dict, Optional, List
from app.risk.models import RiskProfile, TradePlan, TradingStyle, StopLossType, TakeProfitType, AssetRiskSettings
from app.execution.coin_clustering import ClusterManager

class RiskManager:
    """
    Central Risk Management Service.
    Calculates Position Sizing, Stop Losses, and Take Profits based on Risk Profiles.
    """
    
    def __init__(self, config_manager=None, cluster_manager: Optional[ClusterManager] = None):
        self.logger = logging.getLogger("app.risk")
        self.config_manager = config_manager
        self.cluster_manager = cluster_manager
        # Default Profiles (Can be loaded from config)
        self.profiles = {
            TradingStyle.SCALPING: RiskProfile(
                style=TradingStyle.SCALPING,
                max_risk_per_trade=0.005, # 0.5%
                risk_reward_ratio=1.5,
                sl_type=StopLossType.ATR,
                sl_atr_multiplier=1.0,
                tp_type=TakeProfitType.FIXED_RR,
                max_leverage=10.0
            ),
            TradingStyle.SWING: RiskProfile(
                style=TradingStyle.SWING,
                max_risk_per_trade=0.02, # 2%
                risk_reward_ratio=3.0,
                sl_type=StopLossType.ATR,
                sl_atr_multiplier=2.0,
                tp_type=TakeProfitType.MULTI_TIER,
                max_leverage=3.0
            )
        }
        # Per-Asset Settings (Symbol -> Settings)
        self.asset_settings: Dict[str, AssetRiskSettings] = {}
        # Global Default for unknown assets
        self.default_asset_settings = AssetRiskSettings(symbol="DEFAULT")

    def set_asset_settings(self, symbol: str, settings: AssetRiskSettings):
        self.asset_settings[symbol] = settings
    
    def set_default_asset_settings(self, settings: AssetRiskSettings):
        self.default_asset_settings = settings

    def check_cluster_risk(self, symbol: str, current_positions: List[str]) -> bool:
        """
        Delegates to ClusterManager to check if adding 'symbol' violates cluster limits.
        """
        if self.cluster_manager:
            return self.cluster_manager.check_risk_allowance(symbol, current_positions)
        return True # Default to allowed if no cluster manager

    def get_profile(self, style: TradingStyle) -> RiskProfile:
        return self.profiles.get(style, self.profiles[TradingStyle.SWING])

    def calculate_trade_plan(self, 
                             symbol: str, 
                             direction: str, 
                             entry_price: float, 
                             equity: float, 
                             atr: float, 
                             style: TradingStyle = TradingStyle.SWING,
                             confidence_score: float = 1.0,
                             rl_adjustments: Optional[Dict] = None) -> TradePlan:
        """
        Generates a complete TradePlan including Position Size, SL, and TPs.
        
        :param confidence_score: Multiplier derived from MTF analysis (e.g., 1.0 = Neutral, 1.5 = High Confluence)
        """
        profile = self.get_profile(style)
        
        # Resolve Asset Config (Handle Fallback)
        asset_config = self.asset_settings.get(symbol)
        if not asset_config:
            # Fallback to global default, but preserve the symbol name for logging
            asset_config = AssetRiskSettings(
                symbol=symbol,
                max_leverage=self.default_asset_settings.max_leverage,
                max_position_size_usd=self.default_asset_settings.max_position_size_usd,
                min_position_size_usd=self.default_asset_settings.min_position_size_usd,
                leverage_mode=self.default_asset_settings.leverage_mode
            )
        
        # --- 1. Dynamic Risk Adjustment (MTF & RL) ---
        # Base Risk
        risk_pct = profile.max_risk_per_trade
        
        # Apply MTF Confidence (e.g., if Daily+4H align, confidence might be 1.2)
        # We cap this to avoid reckless sizing (e.g., max 1.5x base risk)
        effective_confidence = min(confidence_score, 1.5)
        risk_pct *= effective_confidence
        
        # If RL is enabled, it can further override
        if profile.use_dynamic_adjustment and rl_adjustments:
            if 'sl_multiplier_factor' in rl_adjustments:
                profile.sl_atr_multiplier *= rl_adjustments['sl_multiplier_factor']
            if 'risk_factor' in rl_adjustments:
                risk_pct *= rl_adjustments['risk_factor']

        # --- 2. Calculate Stop Loss ---
        sl_price = self._calculate_stop_loss(direction, entry_price, atr, profile)
        sl_price = round(sl_price, 8) # Fix Floating Point Precision
        
        # Calculate Risk Per Share (Distance to SL)
        risk_per_share = abs(entry_price - sl_price)
        
        # Safety: Prevent division by zero or tiny float errors
        if risk_per_share < 1e-9:
            raise ValueError(f"Stop Loss ({sl_price}) is too close to Entry Price ({entry_price})")

        # --- 3. Calculate Position Size & Leverage ---
        # Total Account Risk = Equity * Effective Risk %
        total_risk_amount = equity * risk_pct
        
        # Raw Position Size (Units) = Total Risk / Risk Per Share
        position_size = total_risk_amount / risk_per_share
        position_size = round(position_size, 8)
        
        # Calculate Notional Value (USD Size)
        notional_value = position_size * entry_price
        
        # Check Asset Constraints (Max Position Size)
        if notional_value > asset_config.max_position_size_usd:
            notional_value = asset_config.max_position_size_usd
            position_size = notional_value / entry_price
            position_size = round(position_size, 8)
            total_risk_amount = position_size * risk_per_share # Recalculate risk amount
            self.logger.warning(f"Position capped by Max Size for {symbol}")

        # Calculate Required Leverage
        # Leverage = Notional Value / Equity
        required_leverage = notional_value / equity
        
        # Determine Max Allowed Leverage (Min of Style vs Asset)
        allowed_leverage = min(profile.max_leverage, asset_config.max_leverage)
        
        # Cap by Leverage
        if required_leverage > allowed_leverage:
            # We must reduce size to fit leverage
            # Max Notional = Equity * Allowed Leverage
            max_notional = equity * allowed_leverage
            position_size = max_notional / entry_price
            position_size = round(position_size, 8)
            total_risk_amount = position_size * risk_per_share
            required_leverage = allowed_leverage
            self.logger.info(f"Position capped by Max Leverage ({allowed_leverage}x) for {symbol}")

        # --- 4. Calculate Take Profits ---
        tp_prices, tp_quantities = self._calculate_take_profits(
            direction, entry_price, risk_per_share, position_size, profile
        )
        
        # Rounding for Exchange Precision
        tp_prices = [round(p, 8) for p in tp_prices]
        tp_quantities = [round(q, 8) for q in tp_quantities]

        # --- 5. Construct Plan ---
        return TradePlan(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            stop_loss_price=sl_price,
            risk_amount=total_risk_amount,
            position_size=position_size,
            leverage=required_leverage,
            take_profit_prices=tp_prices,
            take_profit_quantities=tp_quantities,
            risk_reward_ratio=profile.risk_reward_ratio,
            style=style,
            confidence_score=confidence_score,
            notes=f"ATR: {atr:.4f}, Risk: {risk_pct*100:.2f}% (Conf: {confidence_score}), Lev: {required_leverage:.2f}x"
        )

    def _calculate_stop_loss(self, direction: str, entry: float, atr: float, profile: RiskProfile) -> float:
        """Internal SL Logic"""
        if profile.sl_type == StopLossType.ATR:
            distance = atr * profile.sl_atr_multiplier
        elif profile.sl_type == StopLossType.FIXED_PERCENT:
            distance = entry * profile.sl_fixed_pct
        else:
            distance = atr * 1.5 # Fallback
            
        if direction == "LONG":
            return entry - distance
        else:
            return entry + distance

    def _calculate_take_profits(self, direction: str, entry: float, risk_per_share: float, 
                                total_size: float, profile: RiskProfile):
        """Internal TP Logic"""
        tp_prices = []
        tp_quantities = []
        
        if profile.tp_type == TakeProfitType.MULTI_TIER:
            remaining_qty = total_size
            num_tiers = len(profile.tp_tiers)
            
            # Split position into tiers based on R-multiples
            for i, rr_mult in enumerate(profile.tp_rr_multipliers):
                dist = risk_per_share * rr_mult
                price = (entry + dist) if direction == "LONG" else (entry - dist)
                tp_prices.append(price)
                
                # Quantity for this tier
                if i < num_tiers:
                    # DUST FIX: If this is the last defined tier, take all remaining size
                    # This ensures we don't leave 0.000001 BTC behind due to float math
                    if i == num_tiers - 1:
                        qty = remaining_qty
                    else:
                        qty = total_size * profile.tp_tiers[i]
                        remaining_qty -= qty
                else:
                    qty = 0 # Should not happen if config is correct
                tp_quantities.append(qty)
                
        else:
            # Single TP based on Risk Reward Ratio
            dist = risk_per_share * profile.risk_reward_ratio
            price = (entry + dist) if direction == "LONG" else (entry - dist)
            tp_prices.append(price)
            tp_quantities.append(total_size)
            
        return tp_prices, tp_quantities
