import os
import time
import json
import logging
import azure.cognitiveservices.speech as speechsdk

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s] %(asctime)s %(message)s",
    handlers=[
        logging.FileHandler("container_connection_test.txt", mode="w"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def test_connection(host, audio_file, languages=["en-US", "ar-SA"]):
    """Test connection to Speech container with different host formats."""
    log.info(f"Azure Speech SDK version: {speechsdk.__version__}")
    log.info(f"Testing connection to: {host}")
    log.info(f"Audio file: {os.path.abspath(audio_file)}")
    log.info(f"Languages: {languages}")

    # Check if the file exists
    if not os.path.isfile(audio_file):
        log.error(f"Audio file not found: {audio_file}")
        return False

    # Try both with and without ws:// prefix
    hosts_to_try = []

    # If host includes a port but no protocol, try both with and without ws://
    if ":" in host and not host.startswith(("ws://", "wss://")):
        hosts_to_try.append(f"ws://{host}")
        hosts_to_try.append(host)
    else:
        # Otherwise just use the provided host
        hosts_to_try.append(host)

    # Log the hosts we'll try
    log.info(f"Will try the following host configurations: {hosts_to_try}")

    success = False

    for current_host in hosts_to_try:
        log.info(f"Trying host: {current_host}")

        try:
            # Create speech config
            speech_config = speechsdk.SpeechConfig(host=current_host)
            log.info("✓ Created SpeechConfig successfully")

            # Set language detection mode to continuous
            speech_config.set_property(
                speechsdk.PropertyId.SpeechServiceConnection_LanguageIdMode,
                "Continuous",
            )
            log.info("✓ Set language detection mode to Continuous")

            # Create audio config
            audio_config = speechsdk.audio.AudioConfig(filename=audio_file)
            log.info("✓ Created AudioConfig successfully")

            # Create auto language detection config
            auto_detect = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
                languages=languages
            )
            log.info(
                f"✓ Created AutoDetectSourceLanguageConfig with languages: {languages}"
            )

            # Create source language recognizer (for language detection)
            recognizer = speechsdk.SourceLanguageRecognizer(
                speech_config=speech_config,
                auto_detect_source_language_config=auto_detect,
                audio_config=audio_config,
            )
            log.info("✓ Created SourceLanguageRecognizer successfully")

            # Test with recognize_once first for quick validation
            log.info("Testing with recognize_once()...")
            result = recognizer.recognize_once()

            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                lang = result.properties.get(
                    speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult
                )
                log.info(f"✓ Success! Detected language: {lang}, text: '{result.text}'")
                success = True

                # Try to get the JSON result for detailed inspection
                json_result = result.properties.get(
                    speechsdk.PropertyId.SpeechServiceResponse_JsonResult
                )
                if json_result:
                    try:
                        detail = json.loads(json_result)
                        log.info(f"JSON result details: {json.dumps(detail, indent=2)}")
                    except:
                        log.warning("Could not parse JSON result")

                return True  # Success!
            else:
                log.warning(f"× recognize_once failed with reason: {result.reason}")
                if result.reason == speechsdk.ResultReason.Canceled:
                    cancellation = speechsdk.CancellationDetails.from_result(result)
                    log.error(f"  Error code: {cancellation.error_code}")
                    log.error(f"  Error details: {cancellation.error_details}")
                    log.error(
                        f"  Did you start the container with the correct parameters?"
                    )

        except Exception as e:
            log.error(f"× Failed with host {current_host}: {e}")

    if not success:
        log.error("All connection attempts failed")

    return success


if __name__ == "__main__":
    # Try both test files
    host = "localhost:5003"
    audio_files = [
        "audio/samples/Arabic_english_mix_optimized.wav",
        "audio/samples/Scenario1_20250323_S1056_New Postpaid Activation1.wav",
    ]

    log.info("=== CONTAINER CONNECTION TEST ===")
    success = False

    # Try both audio files
    for audio_file in audio_files:
        log.info(f"\n\nTesting with {audio_file}")
        if test_connection(host, audio_file):
            success = True
            log.info(f"Success with {audio_file}!")
            break
        else:
            log.warning(f"Test failed with {audio_file}")

    # If all standard tests failed, try an alternate port configuration
    if not success:
        log.info("\n\nTrying alternate port mapping: 5000")
        if test_connection("localhost:5000", audio_files[0]):
            log.info(
                "Success with port 5000! Container is listening on port 5000, not 5003."
            )
            success = True

    if success:
        log.info("\n✅ CONTAINER CONNECTION SUCCESSFUL")
    else:
        log.error("\n❌ ALL CONNECTION ATTEMPTS FAILED")
