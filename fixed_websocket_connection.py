"""
fixed_websocket_connection.py

Demonstrates how to properly connect to Speech-to-Text containers with WebSocket.
This fixes the WS_OPEN_ERROR_UNDERLYING_IO_ERROR issue by using the correct endpoint path.
"""

import os
import logging
import json
import time
import azure.cognitiveservices.speech as speechsdk

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(
            "fixed_websocket_connection.txt", mode="w", encoding="utf-8"
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def create_stt_recognizer(
    host: str,
    language: str,
    audio_file: str = None,
    start_time_sec: float = None,
    duration_sec: float = None,
):
    """
    Creates a speech recognizer with the proper WebSocket endpoint path for containers.

    Args:
        host: Container host (e.g., 'localhost:5004')
        language: Language code (e.g., 'en-US')
        audio_file: Optional path to audio file
        start_time_sec: Optional start time in seconds
        duration_sec: Optional duration in seconds

    Returns:
        speechsdk.SpeechRecognizer: Configured speech recognizer
    """
    try:
        # 1. Ensure host has ws:// prefix
        if not host.startswith(("ws://", "wss://")):
            host = f"ws://{host}"

        # 2. THE FIX: Add the required endpoint path for Speech containers
        # This is the most critical part - Speech containers expect this specific path
        endpoint = f"{host}/speech/recognition/conversation/cognitiveservices/v1?language={language}"
        logger.info(f"Creating speech config with endpoint: {endpoint}")

        # 3. Create speech config with the full endpoint
        speech_config = speechsdk.SpeechConfig(host=endpoint)

        # 4. Set the recognition language
        speech_config.speech_recognition_language = language

        # 5. Create audio configuration
        if audio_file:
            if start_time_sec is not None and duration_sec is not None:
                logger.info(
                    f"Using audio segment: {start_time_sec:.2f}s to {start_time_sec + duration_sec:.2f}s"
                )
                audio_config = speechsdk.audio.AudioConfig(
                    filename=audio_file,
                    offset_in_seconds=start_time_sec,
                    duration_in_seconds=duration_sec,
                )
            else:
                audio_config = speechsdk.audio.AudioConfig(filename=audio_file)
        else:
            # Default to microphone if no file provided
            audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)

        # 6. Create the recognizer
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config, audio_config=audio_config
        )

        return recognizer

    except Exception as e:
        logger.error(f"Failed to create recognizer: {str(e)}")
        raise


def test_stt_containers():
    """Test connection to both STT containers with the fixed WebSocket path"""
    logger.info("=== TESTING FIXED SPEECH CONTAINER CONNECTIONS ===")

    # Define container hosts and languages
    containers = {"en-US": "ws://localhost:5004", "ar-SA": "ws://localhost:5005"}

    # Test each container connection
    success_count = 0

    for language, host in containers.items():
        logger.info(f"\nTesting connection to {language} container at {host}")

        try:
            # Create a recognizer to test connection
            recognizer = create_stt_recognizer(host, language)

            # Test with recognize_once to verify connection works
            logger.info(f"Testing recognition with {language} container...")

            # Define a test phrase for connection testing
            test_phrase = "This is a connection test."
            logger.info(f"Simulating recognition of '{test_phrase}'")

            # Add a connection event handler to confirm proper connection
            connected = [False]

            def connection_callback(evt):
                logger.info(f"Connection established with {language} container!")
                connected[0] = True

            def error_callback(evt):
                cancellation = speechsdk.CancellationDetails(evt)
                logger.error(f"Recognition canceled: {cancellation.reason}")
                logger.error(f"Error details: {cancellation.error_details}")

            # Connect the handlers
            recognizer.session_started.connect(connection_callback)
            recognizer.canceled.connect(error_callback)

            # Start recognition - we'll just test connection without actual recognition
            # By starting and stopping to verify connection works
            recognizer.start_continuous_recognition()

            # Wait for connection to establish
            start_time = time.time()
            timeout = 5  # seconds
            while not connected[0] and time.time() - start_time < timeout:
                time.sleep(0.1)

            # Stop recognition
            recognizer.stop_continuous_recognition()

            if connected[0]:
                logger.info(f"✓ Successfully connected to {language} container")
                success_count += 1
            else:
                logger.error(f"✗ Failed to connect to {language} container")

        except Exception as e:
            logger.error(f"Error testing connection: {str(e)}")

    # Log summary
    logger.info(
        f"\nConnection test summary: {success_count}/{len(containers)} successful"
    )

    if success_count == len(containers):
        logger.info("All container connections successful!")
        return True
    else:
        logger.error("Some container connections failed.")
        return False


def test_actual_transcription(
    audio_file="audio/samples/Arabic_english_mix_optimized.wav",
):
    """Test actual transcription with the fixed WebSocket connection"""
    if not os.path.exists(audio_file):
        logger.error(f"Audio file not found: {audio_file}")
        return False

    logger.info(f"Testing transcription with {audio_file}")

    # Test with English container first
    host = "ws://localhost:5004"
    language = "en-US"

    try:
        # Create recognizer with the first 5 seconds of audio
        recognizer = create_stt_recognizer(host, language, audio_file, 0.0, 5.0)

        # Perform recognition
        logger.info("Recognizing speech...")
        result = recognizer.recognize_once()

        # Process result
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            logger.info(f"✓ Transcription successful: {result.text}")
            return True
        elif result.reason == speechsdk.ResultReason.NoMatch:
            logger.warning("No speech could be recognized.")
            return False
        elif result.reason == speechsdk.ResultReason.Canceled:
            details = speechsdk.CancellationDetails.from_result(result)
            logger.error(f"✗ Transcription canceled: {details.reason}")
            logger.error(f"Error details: {details.error_details}")
            return False

    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
        return False


if __name__ == "__main__":
    logger.info("Starting fixed WebSocket connection tests")

    # Test container connections first
    if test_stt_containers():
        # If connections work, try actual transcription
        test_actual_transcription()

    logger.info("Tests completed")
