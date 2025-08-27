import os
import time
import logging
import azure.cognitiveservices.speech as speechsdk

# Set up logging
logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def test_basic_speech_recognition():
    # Use the Arabic_english_mix file
    input_file = "audio/samples/Arabic_english_mix.wav"

    log.info(f"Testing basic speech recognition with file: {input_file}")

    # Check if file exists
    if not os.path.exists(input_file):
        log.error(f"Audio file not found: {input_file}")
        return False

    # Use container configuration
    host = "localhost:5004"  # English STT container

    # Ensure the host has the ws:// prefix
    if not host.startswith(("ws://", "wss://")):
        host = f"ws://{host}"
        log.info(f"Using host with ws:// prefix: {host}")

    log.info(
        f"Using container at {host}"
    )  # Create speech config and audio config for basic recognition
    speech_config = speechsdk.SpeechConfig(host=host)
    audio_config = speechsdk.audio.AudioConfig(filename=input_file)

    # Set the speech recognition language to English
    speech_config.speech_recognition_language = "en-US"

    log.info("Creating SpeechRecognizer for basic test")
    try:
        speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config, audio_config=audio_config
        )
        log.info("Successfully created SpeechRecognizer")
    except Exception as e:
        log.error(f"Failed to create SpeechRecognizer: {e}")
        return False

    # Start recognition and get result
    log.info("Starting speech recognition")
    result = speech_recognizer.recognize_once()

    # Process the result
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        log.info(f"✅ Recognized text: {result.text}")
        return True
    elif result.reason == speechsdk.ResultReason.NoMatch:
        log.warning(f"No speech could be recognized: {result.no_match_details}")
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation = result.cancellation_details
        log.error(f"Speech recognition canceled: {cancellation.reason}")
        if cancellation.reason == speechsdk.CancellationReason.Error:
            log.error(f"Error details: {cancellation.error_details}")

    return False


if __name__ == "__main__":
    log.info("Starting basic speech recognition test")
    log.info(f"Azure Speech SDK version: {speechsdk.__version__}")
    success = test_basic_speech_recognition()
    if success:
        log.info("Test completed successfully")
    else:
        log.error("Test failed")
