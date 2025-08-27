#!/usr/bin/env python
"""
Testing container-based language detection with continuous mode and optimized audio
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


def test_container_lid():
    """Test language detection with the container using continuous mode"""
    # Configuration
    container_url = "ws://localhost:5003"
    audio_file = "audio/samples/Arabic_english_mix_optimized.wav"
    languages = ["en-US", "ar-SA"]

    log.info(f"Testing language detection with container: {container_url}")
    log.info(f"Using audio file: {audio_file}")
    log.info(f"Candidate languages: {languages}")

    try:
        # Create speech config with container URL
        speech_config = speechsdk.SpeechConfig(host=container_url)

        # Set language detection mode to Continuous
        speech_config.set_property(
            property_id=speechsdk.PropertyId.SpeechServiceConnection_LanguageIdMode,
            value="Continuous",
        )

        # Create audio config
        audio_config = speechsdk.audio.AudioConfig(filename=audio_file)

        # Create auto-detect language config
        auto_detect_config = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
            languages=languages
        )

        # Create source language recognizer - required for containers
        recognizer = speechsdk.SourceLanguageRecognizer(
            speech_config=speech_config,
            auto_detect_source_language_config=auto_detect_config,
            audio_config=audio_config,
        )
        log.info("Successfully created SourceLanguageRecognizer")

        # Variables to track results
        done = False
        language_detected = False

        # Event callbacks
        def recognized_cb(evt):
            nonlocal language_detected
            log.info(f"Recognized event: {evt}")

            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                detected_lang = evt.result.properties.get(
                    speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult
                )
                log.info(f"Detected language: {detected_lang}")

                # Get detailed result for timestamps
                json_result = evt.result.properties.get(
                    speechsdk.PropertyId.SpeechServiceResponse_JsonResult
                )
                if json_result:
                    detail = json.loads(json_result)
                    start = detail.get("Offset", 0)
                    duration = detail.get("Duration", 0)
                    log.info(
                        f"Segment: start={start/10000000:.2f}s, duration={duration/10000000:.2f}s"
                    )

                language_detected = True

        def session_started_cb(evt):
            log.info(f"Session started: {evt.session_id}")

        def session_stopped_cb(evt):
            nonlocal done
            log.info(f"Session stopped: {evt.session_id}")
            done = True

        def canceled_cb(evt):
            nonlocal done
            log.error(f"Recognition canceled: {evt.reason}")
            if evt.reason == speechsdk.CancellationReason.Error:
                log.error(f"Error details: {evt.error_details}")
            done = True

        # Connect callbacks
        recognizer.recognized.connect(recognized_cb)
        recognizer.session_started.connect(session_started_cb)
        recognizer.session_stopped.connect(session_stopped_cb)
        recognizer.canceled.connect(canceled_cb)

        # Start continuous recognition
        log.info("Starting continuous recognition")
        recognizer.start_continuous_recognition()

        # Wait for results with timeout
        timeout_sec = 60  # Longer timeout for processing
        start_time = time.time()

        log.info("Waiting for recognition results...")
        while not done:
            elapsed = time.time() - start_time
            if elapsed > timeout_sec:
                log.warning(f"Timeout reached after {timeout_sec} seconds")
                break
            time.sleep(0.5)

        # Stop recognition
        log.info("Stopping recognition")
        recognizer.stop_continuous_recognition()

        if language_detected:
            log.info("✅ Success: Language detection events received")
            return True
        else:
            log.error("❌ Failed: No language detection events received")
            return False

    except Exception as e:
        log.error(f"Error in language detection test: {e}")
        import traceback

        log.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    log.info("Starting container-based language detection test")
    success = test_container_lid()
    if success:
        log.info("Test completed successfully")
        sys.exit(0)
    else:
        log.error("Test failed")
        sys.exit(1)
