import os
import time
import json
import logging
import azure.cognitiveservices.speech as speechsdk

# Set up logging
logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def test_language_detection():
    # Use the new mixed language audio file
    input_file = "audio/samples/Arabic_english_mix.wav"

    log.info(f"Testing language detection with file: {input_file}")

    # Check if file exists
    if not os.path.exists(input_file):
        log.error(f"Audio file not found: {input_file}")
        return False

    # Global variable for tracking completion
    done = False
    # Global variable for tracking language detection
    language_detected = False

    # Use EXACTLY the same configuration as the working example
    speech_config = speechsdk.SpeechConfig(host="ws://localhost:5003")
    audio_config = speechsdk.audio.AudioConfig(filename=input_file)

    auto_detect_source_language_config = (
        speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
            languages=["en-US", "ar-SA"]
        )
    )

    speech_config.set_property(
        property_id=speechsdk.PropertyId.SpeechServiceConnection_LanguageIdMode,
        value="Continuous",
    )

    log.info("Creating SourceLanguageRecognizer")
    try:
        source_language_recognizer = speechsdk.SourceLanguageRecognizer(
            speech_config=speech_config,
            auto_detect_source_language_config=auto_detect_source_language_config,
            audio_config=audio_config,
        )
        log.info("Successfully created SourceLanguageRecognizer")
    except Exception as e:
        log.error(f"Failed to create SourceLanguageRecognizer: {e}")
        return False

    def stop_cb(evt):
        """callback that signals to stop continuous recognition upon receiving an event"""
        log.info(f"CLOSING on {evt}")
        nonlocal done
        done = True

    def audio_recognized(evt):
        """callback that catches the recognized result of audio"""
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            if (
                evt.result.properties.get(
                    speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult
                )
                is None
            ):
                log.warning("Unable to detect any language")
            else:
                detected_src_lang = evt.result.properties[
                    speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult
                ]
                json_result = evt.result.properties[
                    speechsdk.PropertyId.SpeechServiceResponse_JsonResult
                ]
                detail_result = json.loads(json_result)
                start_offset = detail_result["Offset"]
                duration = detail_result["Duration"]
                if duration >= 0:
                    end_offset = duration + start_offset
                else:
                    end_offset = 0

                # Convert time units to seconds for readability
                start_sec = start_offset / 10000000
                duration_sec = duration / 10000000
                end_sec = end_offset / 10000000

                log.info(f"Detected language = {detected_src_lang}")
                log.info(
                    f"Start = {start_sec:.2f}s, End = {end_sec:.2f}s, Duration = {duration_sec:.2f}s"
                )

                nonlocal language_detected
                language_detected = True

    # Connect callbacks to the events fired by the speech recognizer
    source_language_recognizer.recognized.connect(audio_recognized)
    source_language_recognizer.session_started.connect(
        lambda evt: log.info(f"SESSION STARTED: {evt}")
    )
    source_language_recognizer.session_stopped.connect(
        lambda evt: log.info(f"SESSION STOPPED {evt}")
    )
    source_language_recognizer.canceled.connect(
        lambda evt: log.error(f"CANCELED {evt}")
    )

    # Stop continuous recognition on either session stopped or canceled events
    source_language_recognizer.session_stopped.connect(stop_cb)
    source_language_recognizer.canceled.connect(stop_cb)

    # Start continuous speech recognition
    log.info("Starting continuous recognition")
    source_language_recognizer.start_continuous_recognition()

    start_time = time.time()
    timeout_sec = 60  # Give it a longer timeout

    # Wait for recognition to complete or timeout
    log.info("Waiting for recognition to complete...")
    while not done:
        elapsed = time.time() - start_time
        if elapsed > timeout_sec:
            log.warning(f"Timeout reached after {timeout_sec} seconds")
            break

        # Log status every 10 seconds
        if elapsed % 10 < 0.5:
            log.info(f"Still processing... ({elapsed:.1f}s elapsed)")

        time.sleep(0.5)

    log.info("Stopping continuous recognition")
    source_language_recognizer.stop_continuous_recognition()

    if language_detected:
        log.info("✅ Success: Language was detected in the audio")
        return True
    else:
        log.error("❌ No language was detected in the audio")
        return False


if __name__ == "__main__":
    log.info("Starting language detection test using exact working example pattern")
    success = test_language_detection()
    if success:
        log.info("Test completed successfully")
    else:
        log.error("Test failed")
