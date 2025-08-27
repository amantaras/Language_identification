"""
test_simple_connection.py

Test only the connection to make sure we can isolate the issue
"""

import azure.cognitiveservices.speech as speechsdk
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def test_container_connection(host, language):
    log.info(f"Testing connection to {host} with language {language}")

    # Follow your working_container_solution.py exactly:
    if not host.startswith(("ws://", "wss://")):
        host = f"ws://{host}"

    # Create speech config with just the host - NO path
    log.info(f"Creating speech config with host: {host}")
    speech_config = speechsdk.SpeechConfig(host=host)
    speech_config.speech_recognition_language = language

    try:
        log.info("Creating speech recognizer...")
        recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)
        log.info("Success!")
        return True
    except Exception as e:
        log.error(f"Failed: {str(e)}")
        return False


if __name__ == "__main__":
    log.info("=== TESTING BASIC CONNECTION TO STT CONTAINERS ===")

    # Test with direct host only - no path
    test_container_connection("ws://localhost:5004", "en-US")
    test_container_connection("ws://localhost:5005", "ar-SA")
