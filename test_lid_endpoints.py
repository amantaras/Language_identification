"""
Test different endpoints for the language detection container
"""

import websocket
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def test_websocket_connection(url):
    """Test a simple WebSocket connection"""
    log.info(f"Testing WebSocket connection to: {url}")

    try:
        # Try to connect with a short timeout
        ws = websocket.create_connection(url, timeout=5)
        log.info("Connection successful!")
        ws.close()
        return True
    except Exception as e:
        log.error(f"Connection failed: {e}")
        return False


if __name__ == "__main__":
    # Test different endpoints for the language detection container
    endpoints = [
        "ws://localhost:5003/speech/universal/v2",
        "ws://localhost:5003/speech/recognition/conversation/cognitiveservices/v1",
        "ws://localhost:5003/speech/recognition/speech/cognitiveservices/v1",
        "ws://localhost:5003/speech/translation/cognitiveservices/v1",
        "ws://localhost:5003/speech/languageidentification/cognitiveservices/v1",
        "ws://localhost:5003/api/languageidentification/v1.0",
        "ws://localhost:5003/",
    ]

    for endpoint in endpoints:
        test_websocket_connection(endpoint)
