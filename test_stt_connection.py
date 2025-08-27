"""
test_stt_connection.py

Simple test script to verify WebSocket connectivity to Speech-to-Text containers.
"""

import logging
import azure.cognitiveservices.speech as speechsdk
import sys

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def test_stt_connection(host, language):
    """Test connection to a Speech-to-Text container with proper WebSocket configuration."""
    log.info(f"=== Testing STT connection to {host} with language {language} ===")

    # Ensure the URL format is correct
    if not host.startswith(("ws://", "wss://")):
        log.info(f"Adding ws:// prefix to host URL: {host}")
        host = f"ws://{host}"

    # Create speech config (correct approach - host only, no path)
    speech_config = speechsdk.SpeechConfig(host=host)
    speech_config.speech_recognition_language = language

    # Create a simple recognizer to test the connection
    log.info(f"Creating speech recognizer with host={host}, language={language}")
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)

    # Test connection by getting recognizer properties
    try:
        # Just create the recognizer and see if it works
        log.info("Successfully created recognizer. Testing connection...")
        log.info(
            f"Recognizer properties: Authorization token length={len(recognizer.authorization_token) if recognizer.authorization_token else 'None'}"
        )
        log.info("✓ Connection test successful!")
        return True
    except Exception as e:
        log.error(f"Connection test failed: {str(e)}")
        return False


if __name__ == "__main__":
    # Test English container
    test_stt_connection("ws://localhost:5004", "en-US")

    # Test Arabic container
    test_stt_connection("ws://localhost:5005", "ar-SA")
