import os
import time
import azure.cognitiveservices.speech as speechsdk
import sys
import logging

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("simple_test_output.txt", mode="w"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def test_direct_connection():
    # Set up the connection parameters
    host = "localhost:5003"
    audio_file = "audio/samples/Scenario1_20250323_S1056_New Postpaid Activation1.wav"
    languages = ["en-US", "ar-SA"]

    log.info(f"Testing direct connection to {host}")

    # Add ws:// prefix if not present
    if not host.startswith(("ws://", "wss://")):
        host = f"ws://{host}"
        log.info(f"Using host with ws:// prefix: {host}")

    # Create the speech config with direct host configuration
    try:
        speech_config = speechsdk.SpeechConfig(host=host)
        log.info("Created SpeechConfig successfully")
    except Exception as e:
        log.error(f"Failed to create SpeechConfig: {e}")
        return False

    # Set language detection mode
    try:
        speech_config.set_property(
            property_id=speechsdk.PropertyId.SpeechServiceConnection_LanguageIdMode,
            value="Continuous",
        )
        log.info("Set language detection mode to Continuous")
    except Exception as e:
        log.error(f"Failed to set language detection mode: {e}")
        return False

    # Verify audio file exists
    if not os.path.isfile(audio_file):
        log.error(f"Audio file not found: {audio_file}")
        return False

    log.info(f"Using audio file: {os.path.abspath(audio_file)}")

    # Create audio config
    try:
        audio_config = speechsdk.audio.AudioConfig(filename=audio_file)
        log.info("Created AudioConfig successfully")
    except Exception as e:
        log.error(f"Failed to create AudioConfig: {e}")
        return False

    # Create auto language detection config
    try:
        auto_detect = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
            languages=languages
        )
        log.info(
            f"Created AutoDetectSourceLanguageConfig for languages: {', '.join(languages)}"
        )
    except Exception as e:
        log.error(f"Failed to create AutoDetectSourceLanguageConfig: {e}")
        return False

    # Try creating a source language recognizer
    try:
        log.info("Creating SourceLanguageRecognizer...")
        recognizer = speechsdk.SourceLanguageRecognizer(
            speech_config=speech_config,
            auto_detect_source_language_config=auto_detect,
            audio_config=audio_config,
        )
        log.info("Successfully created SourceLanguageRecognizer")
    except Exception as e:
        log.error(f"Failed to create SourceLanguageRecognizer: {e}")
        log.error(f"Error type: {type(e).__name__}")
        log.error(f"Error details: {str(e)}")
        return False

    # Try starting recognition
    done = False
    events_received = False

    def recognized_cb(evt):
        nonlocal events_received
        events_received = True
        log.info(f"Recognized event received")
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            detected_lang = evt.result.properties.get(
                speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult
            )
            log.info(f"Detected language: {detected_lang}")

    def canceled_cb(evt):
        nonlocal done, events_received
        events_received = True
        log.error(
            f"Canceled: reason={evt.reason} code={evt.error_code} details={evt.error_details}"
        )
        done = True

    def stopped_cb(evt):
        nonlocal done, events_received
        events_received = True
        log.info("Session stopped event received")
        done = True

    def session_started_cb(evt):
        nonlocal events_received
        events_received = True
        log.info(f"Session started with ID: {evt.session_id}")

    # Connect callbacks
    recognizer.recognized.connect(recognized_cb)
    recognizer.canceled.connect(canceled_cb)
    recognizer.session_stopped.connect(stopped_cb)
    recognizer.session_started.connect(session_started_cb)

    # Start recognition
    try:
        log.info("Starting continuous recognition...")
        recognizer.start_continuous_recognition()
        log.info("Recognition started successfully")
    except Exception as e:
        log.error(f"Failed to start recognition: {e}")
        return False

    # Wait for results
    start_time = time.time()
    timeout_sec = 30

    while not done:
        elapsed = time.time() - start_time

        if elapsed > 5 and not events_received:
            log.warning("No events received after 5 seconds")

        if elapsed > timeout_sec:
            log.warning(f"Timeout reached after {timeout_sec} seconds")
            break

        time.sleep(0.5)

    # Stop recognition
    try:
        log.info("Stopping recognition...")
        recognizer.stop_continuous_recognition()
        log.info("Recognition stopped successfully")
    except Exception as e:
        log.error(f"Failed to stop recognition: {e}")

    return events_received


if __name__ == "__main__":
    log.info("Starting simple LID test")
    result = test_direct_connection()
    if result:
        log.info(
            "✅ Test successful - events were received from the language detection container"
        )
        sys.exit(0)
    else:
        log.error(
            "❌ Test failed - no events were received from the language detection container"
        )
        sys.exit(1)
