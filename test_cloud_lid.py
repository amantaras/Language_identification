import azure.cognitiveservices.speech as speechsdk
import logging
import json
import time
import os
import sys

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Parameters
SUBSCRIPTION_KEY = "f82cdb5c583346368f7d32af5c22d50d"
REGION = "westus"  # Or other region
AUDIO_FILE = "audio/samples/Arabic_english_mix.wav"
LANGUAGES = ["en-US", "ar-SA"]

log.info(f"Testing Azure cloud service using Speech SDK {speechsdk.__version__}")

# Create speech config for cloud service
try:
    speech_config = speechsdk.SpeechConfig(subscription=SUBSCRIPTION_KEY, region=REGION)
    speech_config.set_property(
        property_id=speechsdk.PropertyId.SpeechServiceConnection_LanguageIdMode,
        value="Continuous",
    )
    log.info("Created SpeechConfig for cloud service")
except Exception as e:
    log.error(f"Failed to create SpeechConfig: {e}")
    sys.exit(1)

# Create audio config
try:
    audio_config = speechsdk.audio.AudioConfig(filename=AUDIO_FILE)
    log.info(f"Created AudioConfig using file: {AUDIO_FILE}")
except Exception as e:
    log.error(f"Failed to create AudioConfig: {e}")
    sys.exit(1)

# Create auto language detection config
try:
    auto_detect_config = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
        languages=LANGUAGES
    )
    log.info(f"Created AutoDetectSourceLanguageConfig with languages: {LANGUAGES}")
except Exception as e:
    log.error(f"Failed to create AutoDetectSourceLanguageConfig: {e}")
    sys.exit(1)

# Create source language recognizer
try:
    language_recognizer = speechsdk.SourceLanguageRecognizer(
        speech_config=speech_config,
        auto_detect_source_language_config=auto_detect_config,
        audio_config=audio_config,
    )
    log.info("Created SourceLanguageRecognizer")
except Exception as e:
    log.error(f"Failed to create SourceLanguageRecognizer: {e}")
    sys.exit(1)

# Event handlers
done = False
event_received = False


def recognized_cb(evt):
    global event_received
    event_received = True
    log.info("Recognition result received")

    if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
        language = evt.result.properties.get(
            speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult
        )
        log.info(f"Detected language: {language}")

        # Get detailed result
        json_result = evt.result.properties.get(
            speechsdk.PropertyId.SpeechServiceResponse_JsonResult
        )
        if json_result:
            try:
                detail = json.loads(json_result)
                log.info(
                    f"Offset: {detail.get('Offset')}, Duration: {detail.get('Duration')}"
                )
            except:
                pass


def canceled_cb(evt):
    global done
    log.error(f"Recognition canceled: {evt.reason}")
    log.error(f"Error code: {evt.error_code}")
    log.error(f"Error details: {evt.error_details}")
    done = True


def session_started_cb(evt):
    log.info(f"Session started")


def session_stopped_cb(evt):
    global done
    log.info(f"Session stopped")
    done = True


# Connect event handlers
language_recognizer.recognized.connect(recognized_cb)
language_recognizer.canceled.connect(canceled_cb)
language_recognizer.session_started.connect(session_started_cb)
language_recognizer.session_stopped.connect(session_stopped_cb)

# Start recognition
log.info("Starting continuous language identification...")
language_recognizer.start_continuous_recognition()

# Wait for events
timeout_sec = 60
start_time = time.time()

while not done:
    elapsed = time.time() - start_time

    if elapsed > 10 and not event_received:
        log.warning("No events received after 10 seconds")

    if elapsed > timeout_sec:
        log.warning(f"Timeout reached after {timeout_sec} seconds")
        break

    time.sleep(0.5)

# Stop recognition
language_recognizer.stop_continuous_recognition()

if event_received:
    log.info("✓ Test succeeded - events were received from the cloud service")
    sys.exit(0)
else:
    log.error("✗ Test failed - no events were received from the cloud service")
    sys.exit(1)
