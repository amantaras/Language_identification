"""
Simple test to verify WebSocket connectivity to Speech containers
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
    # Test the language detection container
    test_websocket_connection("ws://localhost:5003/speech/universal/v2")

    # Also test the other speech containers
    test_websocket_connection(
        "ws://localhost:5004/speech/recognition/conversation/cognitiveservices/v1"
    )
    test_websocket_connection(
        "ws://localhost:5005/speech/recognition/conversation/cognitiveservices/v1"
    )
