import logging
from flask import jsonify
from app.enums.AssetStatus import AssetStatus

logger = logging.getLogger(__name__)

def validate_asset_readiness(symbol: str):
    """
    Checks if the asset is ready in the MarketService.
    Returns a Flask JSON response tuple (response, status_code) if NOT ready.
    Returns None if ready.
    """
    from app.globals import market_service_instance
    
    if market_service_instance:
        if symbol in market_service_instance.asset_states:
            state = market_service_instance.asset_states[symbol]
            if state != AssetStatus.READY.value:
                return jsonify({
                    "success": False,
                    "error": f"Asset {symbol} is currently initializing ({state}). Please wait."
                }), 503
    else:
        logger.warning("Market Service instance not yet available in globals")
        
    return None

import pandas as pd
from typing import Optional, Tuple
from datetime import datetime

def validate_candle_request_params(
    start_dt_str: Optional[str], 
    end_dt_str: Optional[str]
) -> Tuple[Optional[datetime], Optional[datetime], Optional[Tuple]]:
    """
    Validates start/end datetime strings for candle data requests.
    
    Returns:
        (start_dt, end_dt, error_response)
        If error_response is present, return it immediately from the route.
    """
    if not start_dt_str or not end_dt_str:
        return None, None, (
            jsonify({
                "success": False,
                "error": "Missing required params: startDateTime and endDateTime"
            }),
            400,
        )

    try:
        # Use simple pandas parsing which handles ISO formats well
        # and convert to python datetime objects for DB compatibility
        start_dt = pd.to_datetime(start_dt_str, utc=True)
        end_dt = pd.to_datetime(end_dt_str, utc=True)
        
        return start_dt.to_pydatetime(), end_dt.to_pydatetime(), None
        
    except Exception:
        return None, None, (
            jsonify({"success": False, "error": "Invalid datetime format"}),
            400,
        )
