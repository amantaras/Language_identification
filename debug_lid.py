import os
import time
import json
import argparse
import logging
from typing import Optional
import azure.cognitiveservices.speech as speechsdk


def setup_logging(verbose=False):
    """Configure logging based on verbosity level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(asctime)s %(message)s")
    return logging.getLogger(__name__)


def debug_container_connection(
    audio_file,
    lid_host,
    languages=["en-US", "ar-SA"],
    subscription_key=None,
    timeout_sec=60,
    logger=None,
):
    """Debug connection to speech container with detailed logging."""
    log = logger or logging.getLogger(__name__)
    log.info(f"Azure Speech SDK version: {speechsdk.__version__}")

    # Verify file exists
    if not os.path.isfile(audio_file):
        log.error(f"Audio file not found: {audio_file}")
        return False

    # Prepare host string
    if lid_host.startswith(("ws://", "wss://")):
        host_url = lid_host
    else:
        host_url = f"ws://{lid_host}"
    log.info(f"Using host URL: {host_url}")

    # Try with direct host config (for containers)
    try:
        # Set explicit connection properties
        speech_config = speechsdk.SpeechConfig(host=host_url)

        # Set additional properties that might help with connection
        speech_config.set_property(
            speechsdk.PropertyId.SpeechServiceConnection_LanguageIdMode, "Continuous"
        )

        # Set some container-specific properties
        speech_config.set_property(
            speechsdk.PropertyId.SpeechServiceConnection_TranslationToLanguages,
            ",".join(languages),
        )

        # Set optional properties
        if subscription_key:
            speech_config.set_property(
                speechsdk.PropertyId.SpeechServiceConnection_Key, subscription_key
            )
            log.info("Added subscription key")

        log.info("SpeechConfig initialized successfully")
    except Exception as e:
        log.error(f"Failed to create SpeechConfig: {e}")
        return False

    # Create audio config from file
    try:
        audio_config = speechsdk.audio.AudioConfig(filename=audio_file)
        log.info("AudioConfig created successfully")
    except Exception as e:
        log.error(f"Failed to create AudioConfig: {e}")
        return False

    # Create auto language detection config
    try:
        auto_detect_config = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
            languages=languages
        )
        log.info(f"AutoDetectSourceLanguageConfig created with languages: {languages}")
    except Exception as e:
        log.error(f"Failed to create AutoDetectSourceLanguageConfig: {e}")
        return False

    # First try creating just a speech recognizer (not language recognizer)
    # to debug if the issue is with the service connection or language detection specifically
    try:
        log.info("Trying to create a regular SpeechRecognizer first...")
        speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config, audio_config=audio_config
        )
        log.info("✓ SpeechRecognizer created successfully")

        # Try a basic recognition
        result = speech_recognizer.recognize_once()
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            log.info(f"Speech recognized: {result.text}")
        else:
            log.warning(f"No speech recognized. Result reason: {result.reason}")

        log.info("Regular speech recognition test completed")
    except Exception as e:
        log.error(f"Failed to create SpeechRecognizer: {e}")
        # Continue with language detection attempt anyway

    # Now attempt the source language recognizer
    try:
        log.info("Creating SourceLanguageRecognizer...")
        language_recognizer = speechsdk.SourceLanguageRecognizer(
            speech_config=speech_config,
            auto_detect_source_language_config=auto_detect_config,
            audio_config=audio_config,
        )
        log.info("✓ SourceLanguageRecognizer created successfully")
    except Exception as e:
        log.error(f"Failed to create SourceLanguageRecognizer: {e}")
        log.error(f"Error type: {type(e).__name__}")
        log.error(f"Error details: {str(e)}")
        return False

    # Setup event handlers
    done = False
    processed = False

    def recognized(evt):
        nonlocal processed
        log.info("Recognition result received")
        processed = True

        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            language = evt.result.properties.get(
                speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult
            )
            log.info(f"Detected language: {language}")
            json_result = evt.result.properties.get(
                speechsdk.PropertyId.SpeechServiceResponse_JsonResult
            )
            if json_result:
                try:
                    log.debug(f"Raw JSON result: {json_result}")
                    detail = json.loads(json_result)
                    log.info(f"Result details: {json.dumps(detail, indent=2)}")
                except Exception as e:
                    log.error(f"Failed to parse JSON result: {e}")

    def canceled(evt):
        nonlocal done
        log.error(f"Recognition canceled: reason={evt.reason}")
        log.error(f"Error code: {evt.error_code}")
        log.error(f"Error details: {evt.error_details}")
        done = True

    def session_started(evt):
        log.info(f"Session started: {evt}")

    def session_stopped(evt):
        nonlocal done
        log.info(f"Session stopped: {evt}")
        done = True

    # Connect callbacks
    language_recognizer.recognized.connect(recognized)
    language_recognizer.canceled.connect(canceled)
    language_recognizer.session_started.connect(session_started)
    language_recognizer.session_stopped.connect(session_stopped)

    # Start recognition
    try:
        log.info("Starting continuous language detection...")
        language_recognizer.start_continuous_recognition()
        log.info("Recognition started")
    except Exception as e:
        log.error(f"Failed to start recognition: {e}")
        return False

    # Wait for events
    start_time = time.time()

    while not done:
        elapsed = time.time() - start_time

        if elapsed > 10 and not processed:
            log.warning("No events received after 10 seconds")

        if elapsed > timeout_sec:
            log.warning(f"Timeout reached after {timeout_sec} seconds")
            break

        time.sleep(0.5)

    # Stop recognition
    try:
        log.info("Stopping recognition...")
        language_recognizer.stop_continuous_recognition()
        log.info("Recognition stopped")
    except Exception as e:
        log.error(f"Failed to stop recognition: {e}")

    return processed


def main():
    parser = argparse.ArgumentParser(description="Debug language detection")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["en-US", "ar-SA"],
        help="Languages to detect (default: en-US ar-SA)",
    )
    parser.add_argument(
        "--lid-host",
        default="localhost:5003",
        help="Language ID host (default: localhost:5003)",
    )
    parser.add_argument("--key", help="Optional subscription key")
    parser.add_argument(
        "--timeout-sec", type=int, default=60, help="Timeout in seconds (default: 60)"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()
    logger = setup_logging(args.verbose)

    success = debug_container_connection(
        audio_file=args.audio,
        lid_host=args.lid_host,
        languages=args.languages,
        subscription_key=args.key,
        timeout_sec=args.timeout_sec,
        logger=logger,
    )

    if success:
        logger.info("✓ Language detection succeeded - received events")
    else:
        logger.error("✗ Language detection failed - no events received")


if __name__ == "__main__":
    main()
