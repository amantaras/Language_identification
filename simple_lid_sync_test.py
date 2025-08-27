"""
Simpler language detection test using sync recognition instead of continuous
This works around potential issues with continuous recognition in the container
"""

import os
import time
import json
import logging
import azure.cognitiveservices.speech as speechsdk

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("simple_lid_sync_test.txt", mode="w"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def test_language_detection_sync():
    """Test language detection with synchronous recognition instead of continuous"""
    log.info("Testing language detection with synchronous recognition...")

    # Configuration
    host = "ws://localhost:5003"  # LID container
    audio_file = "audio/samples/Arabic_english_mix_optimized.wav"
    languages = ["en-US", "ar-SA"]

    log.info(f"Using host: {host}")
    log.info(f"Languages: {languages}")
    log.info(f"Audio file: {os.path.abspath(audio_file)}")

    # Create speech config
    try:
        speech_config = speechsdk.SpeechConfig(host=host)
        log.info("Created SpeechConfig successfully")
    except Exception as e:
        log.error(f"Failed to create SpeechConfig: {e}")
        return False

    # Set language detection mode to AtStart (more reliable for simple tests)
    try:
        speech_config.set_property(
            speechsdk.PropertyId.SpeechServiceConnection_LanguageIdMode, "AtStart"
        )
        log.info(f"Set language detection mode to AtStart")
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

    try:
        # Create the recognizer
        recognizer = speechsdk.SourceLanguageRecognizer(
            speech_config=speech_config,
            auto_detect_source_language_config=auto_detect,
            audio_config=audio_config,
        )
        log.info("Created SourceLanguageRecognizer successfully")

        # Use synchronous recognition instead of continuous
        log.info("Starting synchronous recognition...")
        result = recognizer.recognize_once()

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            detected_lang = result.properties.get(
                speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult
            )
            log.info(f"DETECTED LANGUAGE: {detected_lang}")
            log.info(f"Recognized text: {result.text}")
            return True
        else:
            log.warning(f"Recognition failed with reason: {result.reason}")

            # Try to get cancellation details if available
            if result.reason == speechsdk.ResultReason.Canceled:
                try:
                    # For newer SDK versions
                    if hasattr(result, "cancellation_details"):
                        cancellation = result.cancellation_details
                        log.error(f"CANCELED: Reason={cancellation.reason}")

                        if cancellation.reason == speechsdk.CancellationReason.Error:
                            log.error(f"CANCELED: ErrorCode={cancellation.error_code}")
                            log.error(
                                f"CANCELED: ErrorDetails={cancellation.error_details}"
                            )
                    else:
                        # For older SDK versions
                        log.error("CANCELED: No detailed information available")
                except Exception as ex:
                    log.error(f"Error accessing cancellation details: {ex}")

            return False

    except Exception as e:
        log.error(f"Recognition failed with error: {e}")
        return False


if __name__ == "__main__":
    log.info("=== SIMPLE LANGUAGE IDENTIFICATION SYNC TEST ===")

    if test_language_detection_sync():
        log.info("✅ SUCCESS! Language detection worked with synchronous recognition.")
    else:
        log.error("❌ FAILED! Language detection with synchronous recognition failed.")
