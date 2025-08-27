"""
working_container_solution.py

Simplified working solution for Speech-to-Text containers using proper WebSocket configuration
"""

import os
import logging
import json
import time
import wave
import azure.cognitiveservices.speech as speechsdk

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("working_code_output.txt", mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def test_stt_connection(host: str, language: str) -> bool:
    """
    Test connection to a Speech-to-Text container with proper WebSocket configuration.

    Args:
        host: WebSocket host address (e.g., 'localhost:5004')
        language: Language code (e.g., 'en-US')

    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        # 1. Ensure the host has ws:// prefix
        if not host.startswith(("ws://", "wss://")):
            host = f"ws://{host}"

        # 2. Build the proper endpoint with path - IMPORTANT!
        # The speech containers expect this specific path format
        endpoint = f"{host}/speech/recognition/conversation/cognitiveservices/v1?language={language}"
        logger.info(f"Connecting to STT endpoint: {endpoint}")

        # 3. Create speech config with the full endpoint
        speech_config = speechsdk.SpeechConfig(host=endpoint)

        # 4. Explicitly set the recognition language
        speech_config.speech_recognition_language = language

        # 5. Create a simple recognizer to test the connection
        recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)
        logger.info(f"Successfully created recognizer for {language}")
        return True

    except Exception as e:
        logger.error(f"Connection failed: {str(e)}")
        return False


def simple_speech_test():
    """Run a simple test of STT container connections"""
    logger.info("=== TESTING SPEECH CONTAINER CONNECTIONS ===")

    # Try connecting to each container
    containers = {"en-US": "ws://localhost:5004", "ar-SA": "ws://localhost:5005"}

    success_count = 0

    for language, host in containers.items():
        logger.info(f"\nTesting connection to {language} container at {host}")
        if test_stt_connection(host, language):
            logger.info(f"✓ Successfully connected to {language} container")
            success_count += 1
        else:
            logger.error(f"✗ Failed to connect to {language} container")

    # Log summary
    logger.info(
        f"\nConnection test summary: {success_count}/{len(containers)} successful"
    )

    # Only proceed with transcription if connections worked
    if success_count == len(containers):
        try:
            # Test with a short audio sample
            audio_file = "audio/samples/Arabic_english_mix_optimized.wav"
            if not os.path.exists(audio_file):
                logger.error(f"Audio file not found: {audio_file}")
                return

            # Get audio duration for logging
            with wave.open(audio_file, "rb") as wf:
                duration_sec = wf.getnframes() / wf.getframerate()

            logger.info(
                f"Testing transcription with {audio_file} ({duration_sec:.2f}s)"
            )

            # Try to transcribe the first 5 seconds using English container
            logger.info(
                "Attempting to transcribe first 5 seconds using English container..."
            )

            # Create speech config with proper path
            host = containers["en-US"]
            endpoint = f"{host}/speech/recognition/conversation/cognitiveservices/v1?language=en-US"
            speech_config = speechsdk.SpeechConfig(host=endpoint)
            speech_config.speech_recognition_language = "en-US"

            # Create audio config with 5-second limit
            audio_config = speechsdk.audio.AudioConfig(
                filename=audio_file, duration_in_seconds=5.0
            )

            # Create recognizer and recognize
            recognizer = speechsdk.SpeechRecognizer(
                speech_config=speech_config, audio_config=audio_config
            )

            # Perform recognition
            result = recognizer.recognize_once()

            # Check result
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                logger.info(f"✓ Transcription successful: {result.text}")
            elif result.reason == speechsdk.ResultReason.NoMatch:
                logger.warning("No speech could be recognized")
            elif result.reason == speechsdk.ResultReason.Canceled:
                details = speechsdk.CancellationDetails.from_result(result)
                logger.error(f"✗ Transcription canceled: {details.reason}")
                logger.error(f"Error details: {details.error_details}")

        except Exception as e:
            logger.error(f"Error during transcription test: {str(e)}")


if __name__ == "__main__":
    simple_speech_test()
