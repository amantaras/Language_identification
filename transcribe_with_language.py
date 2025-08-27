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
    audio_file,
    languages,
    lid_host,
    output_file,
    timeout_sec=30,
    min_segment_sec=0,
    logger=None,
):
    """Detect language segments in the audio file."""
    log = logger or logging.getLogger(__name__)

    log.info(f"Detecting language segments in: {audio_file}")

    # Configure speech recognition for language detection
    log.info(f"Connecting to language identification service at {lid_host}")

    try:
        # Ensure the URL format is correct (keep ws:// protocol for Speech SDK)
        if not lid_host.startswith(("ws://", "wss://")):
            log.warning(f"Adding ws:// prefix to host URL: {lid_host}")
            lid_host = f"ws://{lid_host}"

        speech_config = speechsdk.SpeechConfig(host=lid_host)

        # Set continuous language detection mode
        speech_config.set_property(
            property_id=speechsdk.PropertyId.SpeechServiceConnection_LanguageIdMode,
            value="Continuous",
        )

        log.debug(
            f"Using {len(languages)} languages for detection: {', '.join(languages)}"
        )
        audio_config = speechsdk.audio.AudioConfig(filename=audio_file)

        # Create auto language detection config
        auto_detect = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
            languages=languages
        )

        # Create source language recognizer with error handling for common issues
        recognizer = speechsdk.SourceLanguageRecognizer(
            speech_config=speech_config,
            auto_detect_source_language_config=auto_detect,
            audio_config=audio_config,
        )

        # Add explicit error handler
        def error_handler(evt):
            cancellation = speechsdk.CancellationDetails(evt)
            log.error(f"Speech recognition canceled: {cancellation.reason}")
            if cancellation.reason == speechsdk.CancellationReason.Error:
                log.error(f"Error details: {cancellation.error_details}")
                log.error(f"Error code: {cancellation.error_code}")

        recognizer.canceled.connect(error_handler)

    except Exception as e:
        log.error(f"Error creating language recognizer: {str(e)}")
        log.error(
            "This may be due to container connectivity issues or incorrect host format"
        )
        log.error("Suggestions:")
        log.error("1. Ensure the language container is running (docker ps)")
        log.error("2. Check if the container exposes the correct port (5003)")
        log.error("3. Try using 'ws://127.0.0.1:5003' instead of 'ws://localhost:5003'")
        log.error("4. Check container logs: docker logs speech-lid")
        raise

    # Setup for collecting segments
    segments = []
    done = False

    def recognized(evt):
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            detected_lang = evt.result.properties.get(
                speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult
            )
            if detected_lang:
                json_result = evt.result.properties.get(
                    speechsdk.PropertyId.SpeechServiceResponse_JsonResult
                )
                if json_result:
                    detail = json.loads(json_result)
                    start = detail.get("Offset", 0)
                    duration = detail.get("Duration", 0)
                    end_offset = start + duration if duration >= 0 else start

                    # Convert from HNS (100-nanosecond units) to seconds
                    start_sec = start / 10000000
                    duration_sec = duration / 10000000

                    # Check minimum segment duration
                    if min_segment_sec <= 0 or duration_sec >= min_segment_sec:
                        log.debug(
                            f"Detected {detected_lang} from {start_sec:.2f}s to {start_sec + duration_sec:.2f}s"
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
                    else:
                        log.debug(
                            f"Skipping short segment {detected_lang} ({duration_sec:.2f}s < {min_segment_sec:.2f}s)"
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


def transcribe_segment(
    audio_file, language, start_time, duration, stt_host_map, logger=None
):
    """Transcribe a specific segment of audio in the detected language."""
    log = logger or logging.getLogger(__name__)

    # Get the appropriate STT host for this language
    if language not in stt_host_map:
        log.warning(
            "No STT container configured for language {}. Skipping segment.".format(
                language
            )
        )
        return None

    stt_host = stt_host_map[language]
    log.info("Transcribing {} segment using {}".format(language, stt_host))

    # Convert from HNS to seconds if needed
    if start_time > 1000000:  # Likely in HNS units
        start_sec = start_time / 10000000
        duration_sec = duration / 10000000
    else:  # Already in seconds
        start_sec = start_time
        duration_sec = duration

    # Configure speech recognition
    speech_config = speechsdk.SpeechConfig(host=stt_host)
    speech_config.speech_recognition_language = language

    # Configure audio with offset and duration
    audio_config = speechsdk.audio.AudioConfig(
        filename=audio_file,
        offset_in_seconds=start_sec,
        duration_in_seconds=duration_sec,
    )

    # Create recognizer
    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, audio_config=audio_config
    )

    # Perform recognition
    log.debug(
        "Recognizing speech from {:.2f}s to {:.2f}s".format(
            start_sec, start_sec + duration_sec
        )
    )
    result = recognizer.recognize_once()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        transcript = result.text
        log.debug("Transcribed: {}".format(transcript))
        return {
            "language": language,
            "start_time": start_time,
            "duration": duration,
            "start_sec": start_sec,
            "duration_sec": duration_sec,
            "transcript": transcript,
        }
    elif result.reason == speechsdk.ResultReason.NoMatch:
        log.warning("No speech recognized in this segment")
        return {
            "language": language,
            "start_time": start_time,
            "duration": duration,
            "start_sec": start_sec,
            "duration_sec": duration_sec,
            "transcript": "",
            "note": "No speech recognized",
        }
    elif result.reason == speechsdk.ResultReason.Canceled:
        details = speechsdk.CancellationDetails(result)
        log.error("Recognition canceled: {}".format(details.reason))
        log.error("Error details: {}".format(details.error_details))
        return {
            "language": language,
            "start_time": start_time,
            "duration": duration,
            "start_sec": start_sec,
            "duration_sec": duration_sec,
            "transcript": "",
            "error": details.error_details,
        }


def process_audio_file(args, logger):
    """Process an audio file: detect languages and transcribe segments."""
    # Step 1: Detect language segments
    if not os.path.exists(args.segments) or args.force_detection:
        logger.info(f"Performing language detection on {args.audio}")
        segments_result = detect_language_segments(
            audio_file=args.audio,
            languages=args.languages,
            lid_host=args.lid_host,
            output_file=args.segments,
            timeout_sec=args.timeout_sec,
            min_segment_sec=args.min_segment_sec,
            logger=logger,
        )
    else:
        logger.info(f"Using existing language segments from {args.segments}")
        with open(args.segments, "r", encoding="utf-8") as f:
            segments_result = json.load(f)

    # Step 2: Create STT host mapping
    stt_host_map = {}
    for mapping in args.stt_map:
        if "=" in mapping:
            lang, host = mapping.split("=", 1)
            stt_host_map[lang] = host

    if not stt_host_map:
        logger.error("No STT host mappings provided. Cannot transcribe segments.")
        return

    logger.info("STT host mapping: {}".format(stt_host_map))

    # Step 3: Transcribe each segment
    transcribed_segments = []

    for segment in segments_result.get("Segments", []):
        language = segment.get("Language")
        start_time = segment.get("StartTimeHns")
        duration = segment.get("DurationHns")

        # Skip segments marked as skipped
        if segment.get("IsSkipped", False):
            logger.info(
                "Skipping segment marked as skipped: {:.2f}s to {:.2f}s".format(
                    start_time / 10000000, (start_time + duration) / 10000000
                )
            )
            continue

        result = transcribe_segment(
            audio_file=args.audio,
            language=language,
            start_time=start_time,
            duration=duration,
            stt_host_map=stt_host_map,
            logger=logger,
        )

        if result:
            transcribed_segments.append(result)

    # Step 4: Create final transcript
    final_result = {
        "audio_file": args.audio,
        "segment_count": len(transcribed_segments),
        "transcribed_segments": transcribed_segments,
        "full_transcript": " ".join(
            [
                f"[{seg['language']}] {seg.get('transcript', '')}"
                for seg in sorted(transcribed_segments, key=lambda x: x["start_time"])
                if seg.get("transcript")
            ]
        ),
    }

    # Write final transcript to output file
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(final_result, f, indent=2)

    logger.info(f"Wrote complete transcript to {args.output}")
    logger.info(f"Full transcript: {final_result['full_transcript']}")

    return final_result


def main():
    parser = argparse.ArgumentParser(description="Language-aware speech transcription")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument(
        "--languages",
        nargs="+",
        required=True,
        help="List of languages to detect (e.g., en-US es-ES)",
    )
    parser.add_argument(
        "--lid-host",
        default="ws://localhost:5003",
        help="Language ID container host (default: ws://localhost:5003)",
    )
    parser.add_argument(
        "--stt-map",
        nargs="+",
        required=True,
        help="STT host mapping, format: lang=host (e.g., en-US=ws://localhost:5000)",
    )
    parser.add_argument(
        "--segments",
        default="segments.json",
        help="Path to save/load language segments JSON",
    )
    parser.add_argument(
        "--output", default="transcript.json", help="Path to save final transcript JSON"
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=30,
        help="Timeout for language detection in seconds",
    )
    parser.add_argument(
        "--min-segment-sec",
        type=float,
        default=0,
        help="Minimum segment duration in seconds",
    )
    parser.add_argument(
        "--force-detection",
        action="store_true",
        help="Force language detection even if segments file exists",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()
    logger = setup_logging(args.verbose)

    process_audio_file(args, logger)


if __name__ == "__main__":
    main()
