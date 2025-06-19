import logging

import asyncio
from flask import Blueprint, request, jsonify

from app.exchange.FyersConnector import FyersConnector
from app.services.market_service import MarketService

logger = logging.getLogger("app")

fyers_auth_bp = Blueprint('fyers_auth', __name__)

market_service_instance = MarketService()

def run_async(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

@fyers_auth_bp.get("/api/auth/fyers/url")
def get_fyers_auth_url():
    """Get Fyers authorization URL for the user to authenticate"""
    fyers_connector = FyersConnector()
    try:
        auth_url = fyers_connector.get_auth_url()
        return {
            "status": "success",
            "auth_url": auth_url
        }
    except Exception as e:
        logger.error(f"Error generating Fyers auth URL: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to generate authorization URL: {str(e)}"
        }, 500


@fyers_auth_bp.post("/api/auth/fyers/code")
def process_fyers_auth_code():
    """Process the auth code received from Fyers redirect"""
    fyers_connector = FyersConnector()

    data = request.json
    auth_code = data["auth_code"]

    if not auth_code:
        return jsonify({
            "status": "error",
            "message": "Auth code is required"
        }), 400

    try:
        # Authorize with the provided code
        fyers_accessor = fyers_connector.authorize(auth_code=auth_code)

        if fyers_connector.get_access_token():
            # Initialize the market service with the authenticated connector
            run_async(market_service_instance.initialize_after_auth())

            return {
                "status": "success",
                "message": "Authentication successful"
            }
        else:
            return {
                "status": "error",
                "message": "Failed to obtain access token"
            }, 400
    except Exception as e:
        logger.error(f"Error processing Fyers auth code: {str(e)}")
        return {
            "status": "error",
            "message": f"Authentication failed: {str(e)}"
        }, 500
