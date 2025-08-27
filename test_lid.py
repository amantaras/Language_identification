import os
import json
import argparse
import logging
import time
import azure.cognitiveservices.speech as speechsdk


def setup_logging(verbose=False):
    """Configure logging based on verbosity level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")
    return logging.getLogger(__name__)


def detect_language_segments(
    audio_file, languages, lid_host, output_file, timeout_sec=30, logger=None
):
    """Detect language segments using the exact working implementation from your code."""
    log = logger or logging.getLogger(__name__)

    log.info(f"Detecting language segments in: {audio_file}")

    # Keep this pattern exactly as it is in your working implementation
    log.debug(f"Using host URL: {lid_host}")

    # Add ws:// prefix if not present for container connectivity
    if not lid_host.startswith(("ws://", "wss://")):
        lid_host = f"ws://{lid_host}"
        log.debug(f"Added ws:// prefix. Using host URL: {lid_host}")

    # Use direct SpeechConfig constructor
    speech_config = speechsdk.SpeechConfig(host=lid_host)

    # Set continuous language detection mode
    speech_config.set_property(
        property_id=speechsdk.PropertyId.SpeechServiceConnection_LanguageIdMode,
        value="Continuous",
    )

    audio_config = speechsdk.audio.AudioConfig(filename=audio_file)

    # Create auto language detection config
    auto_detect = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
        languages=languages
    )

    # Create source language recognizer
    log.debug("Creating SourceLanguageRecognizer...")
    recognizer = speechsdk.SourceLanguageRecognizer(
        speech_config=speech_config,
        auto_detect_source_language_config=auto_detect,
        audio_config=audio_config,
    )

    # Setup for collecting segments
    segments = []
    done = False

    def recognized(evt):
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            detected_lang = evt.result.properties.get(
                speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult
            )
            if detected_lang:
                log.debug(f"Detected language: {detected_lang}")
                json_result = evt.result.properties.get(
                    speechsdk.PropertyId.SpeechServiceResponse_JsonResult
                )
                if json_result:
                    detail = json.loads(json_result)
                    start = detail.get("Offset", 0)
                    duration = detail.get("Duration", 0)
                    end_offset = start + duration if duration >= 0 else start

                    # Convert from HNS (100-nanosecond units) to seconds for display
                    start_sec = start / 10000000
                    duration_sec = duration / 10000000

                    log.debug(
                        f"Segment from {start_sec:.2f}s to {start_sec + duration_sec:.2f}s"
                    )

                    segments.append(
                        {
                            "Id": len(segments),
                            "Language": detected_lang,
                            "StartTimeHns": start,
                            "EndTimeHns": end_offset,
                            "DurationHns": duration,
                            "IsSkipped": False,
                        }
                    )

    def stop_cb(evt):
        nonlocal done
        log.debug("Recognition stopped/canceled event received")
        done = True

    # Connect callbacks
    recognizer.recognized.connect(recognized)
    recognizer.session_stopped.connect(stop_cb)
    recognizer.canceled.connect(stop_cb)

    # Start continuous recognition
    log.info("Starting continuous language identification...")
    recognizer.start_continuous_recognition()

    # Wait for recognition to complete
    start_time = time.time()
    while not done:
        if timeout_sec is not None and (time.time() - start_time) > timeout_sec:
            log.warning(f"Timeout reached after {timeout_sec} seconds")
            recognizer.stop_continuous_recognition()
            break
        time.sleep(0.5)

    # Write segments to file
    result = {
        "AudioFile": audio_file,
        "SegmentCount": len(segments),
        "Segments": segments,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    log.info(f"Wrote {len(segments)} language segments to {output_file}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Simple language detection test")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument(
        "--languages",
        nargs="+",
        required=True,
        help="List of languages to detect (e.g., en-US ar-SA)",
    )
    parser.add_argument(
        "--lid-host",
        default="ws://localhost:5003",
        help="Language ID container host (default: ws://localhost:5003)",
    )
    parser.add_argument(
        "--output", default="segments.json", help="Path to save segments JSON"
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=30,
        help="Timeout for language detection in seconds",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()
    logger = setup_logging(args.verbose)

    detect_language_segments(
        audio_file=args.audio,
        languages=args.languages,
        lid_host=args.lid_host,
        output_file=args.output,
        timeout_sec=args.timeout_sec,
        logger=logger,
    )


if __name__ == "__main__":
    main()
