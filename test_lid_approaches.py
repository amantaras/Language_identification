"""
Comprehensive Language Identification Container Connection Test
"""

import os
import time
import logging
import azure.cognitiveservices.speech as speechsdk

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("lid_direct_test.txt", mode="w"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def test_speech_recognition_first():
    """Test basic speech recognition to validate audio and connection"""
    log.info("Testing basic speech recognition to verify audio...")

    # Use standard speech recognition to the English container
    host = "ws://localhost:5004"  # English STT container
    audio_file = "audio/samples/Arabic_english_mix_optimized.wav"

    try:
        speech_config = speechsdk.SpeechConfig(host=host)
        audio_config = speechsdk.audio.AudioConfig(filename=audio_file)
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config, audio_config=audio_config
        )

        log.info("Running speech recognition with English STT container...")
        result = recognizer.recognize_once()

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            log.info(f"Speech recognized: {result.text}")
            return True
        else:
            log.warning(f"Speech not recognized. Result: {result.reason}")
            return False

    except Exception as e:
        log.error(f"Speech recognition test failed: {e}")
        return False


def try_different_approaches():
    """Try different approaches to make language detection work"""

    # Test configurations to try
    configs = [
        {
            "name": "Default AtStart mode",
            "host": "ws://localhost:5003",
            "mode": "AtStart",
            "languages": ["en-US", "ar-SA"],
            "audio": "audio/samples/Arabic_english_mix_optimized.wav",
        },
        {
            "name": "Without ws:// prefix",
            "host": "localhost:5003",
            "mode": "AtStart",
            "languages": ["en-US", "ar-SA"],
            "audio": "audio/samples/Arabic_english_mix_optimized.wav",
        },
        {
            "name": "Single language only",
            "host": "ws://localhost:5003",
            "mode": "AtStart",
            "languages": ["en-US"],
            "audio": "audio/samples/Arabic_english_mix_optimized.wav",
        },
        {
            "name": "Alternative audio file",
            "host": "ws://localhost:5003",
            "mode": "AtStart",
            "languages": ["en-US", "ar-SA"],
            "audio": "audio/samples/Scenario1_20250323_S1056_New Postpaid Activation1.wav",
        },
    ]

    for i, config in enumerate(configs):
        log.info(f"\n=== TEST {i+1}: {config['name']} ===")
        success = test_language_detection(
            host=config["host"],
            mode=config["mode"],
            languages=config["languages"],
            audio_file=config["audio"],
        )

        if success:
            log.info(f"✅ TEST {i+1} SUCCEEDED!")
            return True
        else:
            log.info(f"❌ TEST {i+1} FAILED!")

    return False


def test_language_detection(host, mode, languages, audio_file):
    """Test language detection with specific configuration"""
    log.info(f"Testing with host: {host}")
    log.info(f"Language mode: {mode}")
    log.info(f"Languages: {languages}")
    log.info(f"Audio file: {os.path.abspath(audio_file)}")

    # Create speech config
    try:
        speech_config = speechsdk.SpeechConfig(host=host)
        log.info("Created SpeechConfig successfully")
    except Exception as e:
        log.error(f"Failed to create SpeechConfig: {e}")
        return False

    # Configure language detection mode
    try:
        speech_config.set_property(
            speechsdk.PropertyId.SpeechServiceConnection_LanguageIdMode, mode
        )
        log.info(f"Set language detection mode to {mode}")
    except Exception as e:
        log.error(f"Failed to set language detection mode: {e}")
        return False

    # Create audio config
    try:
        audio_config = speechsdk.audio.AudioConfig(filename=audio_file)
        log.info("Created AudioConfig successfully")
    except Exception as e:
        log.error(f"Failed to create AudioConfig: {e}")
        return False

    # Create language config
    try:
        auto_detect = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
            languages=languages
        )
        log.info(f"Created AutoDetectSourceLanguageConfig with languages: {languages}")
    except Exception as e:
        log.error(f"Failed to create AutoDetectSourceLanguageConfig: {e}")
        return False

    # Create recognizer and run recognition
    done = False
    success = False

    def recognized_cb(evt):
        nonlocal success
        detected_lang = evt.result.properties.get(
            speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult
        )
        log.info(f"DETECTED LANGUAGE: {detected_lang}")
        log.info(f"Recognized text: {evt.result.text}")
        success = True

    def canceled_cb(evt):
        nonlocal done
        log.warning(f"Recognition canceled: {evt.reason}")
        try:
            # Different SDK versions handle cancellation details differently
            error_details = getattr(evt, "error_details", None)
            if error_details:
                log.error(f"Error details: {error_details}")
        except Exception as e:
            log.error(f"Error accessing cancellation details: {e}")
        done = True

    def session_stopped_cb(evt):
        nonlocal done
        log.info("Session stopped")
        done = True

    try:
        # Create the recognizer
        recognizer = speechsdk.SourceLanguageRecognizer(
            speech_config=speech_config,
            auto_detect_source_language_config=auto_detect,
            audio_config=audio_config,
        )
        log.info("Created SourceLanguageRecognizer successfully")

        # Connect event handlers
        recognizer.recognized.connect(recognized_cb)
        recognizer.canceled.connect(canceled_cb)
        recognizer.session_stopped.connect(session_stopped_cb)

        # Start continuous recognition
        log.info("Starting continuous recognition...")
        recognizer.start_continuous_recognition()

        # Wait for results
        timeout = 15  # seconds
        start_time = time.time()

        while not done and (time.time() - start_time) < timeout:
            time.sleep(0.5)

        # Check if timeout occurred
        if (time.time() - start_time) >= timeout:
            log.warning("Recognition timed out")

        # Stop recognition
        log.info("Stopping recognition...")
        recognizer.stop_continuous_recognition()

        # Return success flag
        return success

    except Exception as e:
        log.error(f"Recognition failed with error: {e}")
        return False


if __name__ == "__main__":
    log.info("=== COMPREHENSIVE LANGUAGE IDENTIFICATION TEST ===")

    # First test if speech recognition works at all
    if test_speech_recognition_first():
        log.info("✅ Basic speech recognition works!")
        log.info("Testing language detection...")

        if try_different_approaches():
            log.info(
                "✅ SUCCESS! Language detection worked with one of the approaches."
            )
        else:
            log.error("❌ FAILED! None of the language detection approaches worked.")
    else:
        log.error(
            "❌ Basic speech recognition failed. Check container and audio setup first."
        )
