"""
Language Identification Container Connection Test
This script tests direct connectivity to the language identification container
using the most straightforward possible approach.
"""

import os
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


def test_simple_recognition():
    """Test the most basic possible language recognition"""
    # Configuration
    host = "ws://localhost:5003"
    audio_file = "audio/samples/Arabic_english_mix_optimized.wav"
    languages = ["en-US", "ar-SA"]

    log.info(f"Testing with host: {host}")
    log.info(f"Audio file: {os.path.abspath(audio_file)}")

    # Create speech config
    try:
        speech_config = speechsdk.SpeechConfig(host=host)
        log.info("Created SpeechConfig successfully")
    except Exception as e:
        log.error(f"Failed to create SpeechConfig: {e}")
        return False

    # Configure for language detection (at-start is simpler than continuous)
    try:
        # Use AtStart mode which is more reliable for testing
        speech_config.set_property(
            speechsdk.PropertyId.SpeechServiceConnection_LanguageIdMode, "AtStart"
        )
        log.info("Set language detection mode to AtStart")
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

    # Create recognizer
    try:
        # Use SourceLanguageRecognizer for container language ID
        recognizer = speechsdk.SourceLanguageRecognizer(
            speech_config=speech_config,
            auto_detect_source_language_config=auto_detect,
            audio_config=audio_config,
        )
        log.info("Created SourceLanguageRecognizer successfully")
    except Exception as e:
        log.error(f"Failed to create SourceLanguageRecognizer: {e}")
        return False

    # Test with basic recognition
    try:
        log.info("Starting recognition_once_async...")
        result_future = recognizer.recognize_once_async()
        log.info("Waiting for result...")
        result = result_future.get()

        log.info(f"Recognition result reason: {result.reason}")

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            # Get detected language
            detected_lang = result.properties.get(
                speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult
            )
            log.info(f"SUCCESS! Detected language: {detected_lang}")
            log.info(f"Recognized text: {result.text}")
            return True
        elif result.reason == speechsdk.ResultReason.NoMatch:
            log.warning("No speech could be recognized")
        elif result.reason == speechsdk.ResultReason.Canceled:
            log.warning("Recognition canceled")
            try:
                cancellation = speechsdk.CancellationDetails(result)
                log.error(f"Error code: {cancellation.error_code}")
                log.error(f"Error details: {cancellation.error_details}")
            except Exception as e:
                log.error(f"Failed to get cancellation details: {e}")
                log.error(
                    f"SDK version may not support CancellationDetails constructor with result argument"
                )
        else:
            log.warning(f"Recognition result reason: {result.reason}")

        return False
    except Exception as e:
        log.error(f"Recognition failed with error: {e}")
        return False


if __name__ == "__main__":
    log.info("=== LANGUAGE IDENTIFICATION DIRECT TEST ===")
    success = test_simple_recognition()

    if success:
        log.info("Test PASSED! Successfully detected language.")
    else:
        log.error("Test FAILED! Could not detect language.")
