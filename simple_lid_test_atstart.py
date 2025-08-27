#!/usr/bin/env python
"""
Simple test script for language identification with container
"""
import os
import sys
import time
import json
import logging
import azure.cognitiveservices.speech as speechsdk

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("simple_lid_test.txt", mode="w"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def test_direct_connection():
    # Set up the connection parameters
    host = "localhost:5003"
    audio_file = "audio/samples/Arabic_english_mix_optimized.wav"
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

    # Set language detection mode - trying Continuous instead of continuous
    try:
        # Try these different configurations
        # speech_config.set_property(
        #     property_id=speechsdk.PropertyId.SpeechServiceConnection_LanguageIdMode,
        #     value="Continuous",
        # )
        # or
        speech_config.set_property(
            property_id=speechsdk.PropertyId.SpeechServiceConnection_LanguageIdMode,
            value="AtStart",
        )
        log.info("Set language detection mode to AtStart")
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

    # Create auto detect language config
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

    # Create source language recognizer
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
        log.error(f"Error details: {e}")
        return False

    # Track results
    events_received = False

    def recognized(evt):
        nonlocal events_received
        log.info("Recognized event received")
        events_received = True
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            detected_language = evt.result.properties.get(
                speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult
            )
            log.info(f"Detected language: {detected_language}")

            # Get detailed recognition result
            json_result = evt.result.properties.get(
                speechsdk.PropertyId.SpeechServiceResponse_JsonResult
            )
            if json_result:
                result_detail = json.loads(json_result)
                start = result_detail.get("Offset", 0)
                duration = result_detail.get("Duration", 0)
                log.info(
                    f"Segment: start={start/10000000:.2f}s, duration={duration/10000000:.2f}s"
                )

    def session_started(evt):
        log.info(f"Session started with ID: {evt.session_id}")

    def session_stopped(evt):
        log.info("Session stopped event received")

    # Connect event handlers
    recognizer.recognized.connect(recognized)
    recognizer.session_started.connect(session_started)
    recognizer.session_stopped.connect(session_stopped)

    # Start recognition
    log.info("Starting continuous recognition...")
    recognizer.start_continuous_recognition()

    # Wait for events
    start_time = time.time()
    max_wait_sec = 30
    while time.time() - start_time < max_wait_sec and not events_received:
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
