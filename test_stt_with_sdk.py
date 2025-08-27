"""
test_stt_with_sdk.py

Test transcription using the Speech SDK directly instead of REST.
"""

import os
import logging
import time
import azure.cognitiveservices.speech as speechsdk
import argparse

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("test_stt_with_sdk.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def transcribe_with_sdk(audio_file, language, host):
    """
    Transcribe audio using the Speech SDK connected to container.
    """
    log.info(f"Transcribing {audio_file} in {language} using {host}")

    # Ensure the host has ws:// prefix
    if not host.startswith(("ws://", "wss://")):
        host = f"ws://{host}"

    # Create speech config with just the host - NO path
    log.info(f"Creating speech config with host: {host}")
    speech_config = speechsdk.SpeechConfig(host=host)
    speech_config.speech_recognition_language = language

    # Create audio config from file
    audio_config = speechsdk.audio.AudioConfig(filename=audio_file)

    # Create recognizer
    log.info("Creating speech recognizer...")
    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, audio_config=audio_config
    )

    # Start recognition with timeout
    log.info("Starting recognition...")
    done = False
    result = None

    def on_recognized(evt):
        nonlocal result, done
        result = evt.result
        done = True
        log.info("Recognition completed")

    def on_canceled(evt):
        nonlocal done
        log.warning(f"Recognition canceled: {evt.result.cancellation_details.reason}")
        done = True

    recognizer.recognized.connect(on_recognized)
    recognizer.canceled.connect(on_canceled)
    recognizer.session_stopped.connect(lambda evt: setattr(locals(), "done", True))

    recognizer.start_continuous_recognition()

    # Wait for result with timeout
    start_time = time.time()
    timeout = 10  # 10 seconds timeout
    while not done and time.time() - start_time < timeout:
        time.sleep(0.1)

    recognizer.stop_continuous_recognition()

    if not result:
        log.error("Recognition timed out or failed")
        return ""

    # Process result
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        log.info(f"Recognition successful: {result.text}")
        return result.text
    elif result.reason == speechsdk.ResultReason.NoMatch:
        log.warning(f"No speech could be recognized: {result.no_match_details}")
        return ""
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation = result.cancellation_details
        log.error(f"Recognition canceled: {cancellation.reason}")
        if cancellation.reason == speechsdk.CancellationReason.Error:
            log.error(f"Error details: {cancellation.error_details}")
        return ""


def main():
    parser = argparse.ArgumentParser(description="Test STT with Speech SDK")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--language", required=True, help="Language code (e.g., en-US)")
    parser.add_argument(
        "--host", required=True, help="STT container host (e.g., ws://localhost:5004)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.audio):
        log.error(f"Audio file not found: {args.audio}")
        return

    log.info("=== TESTING STT WITH SPEECH SDK ===")
    transcribe_with_sdk(args.audio, args.language, args.host)


if __name__ == "__main__":
    main()
