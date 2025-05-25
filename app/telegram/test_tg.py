#!/usr/bin/env python3

import os
import logging
from dotenv import load_dotenv
from app.telegram import TelegramClient

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("telegram_test")


def test_telegram_client():
    """Test the TelegramClient functionality"""
    load_dotenv()

    # Get the chat ID from environment variable
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not chat_id:
        logger.error("TELEGRAM_CHAT_ID environment variable not found!")
        return False

    # Create client instance
    client = TelegramClient()
    logger.info("TelegramClient initialized")

    # Test 1: Send a simple message
    logger.info("Test 1: Sending simple message...")
    result = client.send_message(
        message="🧪 This is a test message from TelegramClient",
        chat_id=chat_id
    )
    logger.info(f"Simple message send result: {result}")

    # Test 2: Send a template message (if template exists)
    logger.info("Test 2: Sending template message...")
    try:
        result = client.send_chart_alert(
            template_name="tg_alert_trendline.j2",
            chat_id=chat_id,
            chart_path="/path/to/trendline_chart.png",
            signal_type="BUY",
            timestamp="2023-11-28 14:30:00",
            stock_symbol="BTC/USDT",
            price="42000",
            strategy_name="Trendline Breakout",
            open="41800",
            high="42100",
            low="41750",
            close="42000",
            volume="1.2M",
            indicators="RSI: 65, MACD: Bullish",
            trendline_details="Support trendline breakout from 41500 level",
            risk_reward="1:3",
            targets="T1: 43000, T2: 44000",
            stop_loss="41200"
        )
        logger.info(f"Template message send result: {result}")
    except Exception as e:
        logger.error(f"Template test failed: {e}")

    # Test 3: Send a photo (create a sample image first)
    logger.info("Test 3: Sending photo...")
    try:
        # Create a simple test image using PIL if not available
        import numpy as np
        from PIL import Image

        # Create a simple test image
        test_image_path = "test_image.png"
        img = Image.new('RGB', (200, 200), color=(73, 109, 137))
        img.save(test_image_path)

        result = client.send_photo(
            photo_path=test_image_path,
            caption="Test photo from TelegramClient",
            chat_id=chat_id
        )
        logger.info(f"Photo send result: {result}")

        # Clean up
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
    except ImportError:
        logger.warning("PIL not installed, skipping photo test")
    except Exception as e:
        logger.error(f"Photo test failed: {e}")

    # Test 4: Send a document
    logger.info("Test 4: Sending document...")
    try:
        # Create a simple test document
        test_doc_path = "test_document.txt"
        with open(test_doc_path, 'w') as f:
            f.write("This is a test document for TelegramClient\n")

        result = client.send_document(
            file_path=test_doc_path,
            caption="Test document from TelegramClient",
            chat_id=chat_id
        )
        logger.info(f"Document send result: {result}")

        # Clean up
        if os.path.exists(test_doc_path):
            os.remove(test_doc_path)
    except Exception as e:
        logger.error(f"Document test failed: {e}")

    logger.info("All tests completed!")
    return True


if __name__ == "__main__":
    test_telegram_client()