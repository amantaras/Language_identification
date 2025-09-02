import os
import json
import argparse
import logging
import azure.cognitiveservices.speech as speechsdk
from .language_detect import detect_languages


def setup_logging(verbose=False):
    """Configure logging based on verbosity level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")
    return logging.getLogger(__name__)


def transcribe_segment(
    audio_file, language, start_time, duration, stt_host_map, logger=None
):
    """Transcribe a specific segment of audio in the detected language."""
    log = logger or logging.getLogger(__name__)

    # Get the appropriate STT host for this language
    if language not in stt_host_map:
        log.warning(
            f"No STT container configured for language {language}. Skipping segment."
        )
        return None

    stt_host = stt_host_map[language]
    log.info(f"Transcribing {language} segment using {stt_host}")

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
        f"Recognizing speech from {start_sec:.2f}s to {start_sec + duration_sec:.2f}s"
    )
    result = recognizer.recognize_once()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        transcript = result.text
        log.debug(f"Transcribed: {transcript}")
        return {
            "language": language,
            "start_time": start_time,
            "duration": duration,
            "start_sec": start_sec,
            "duration_sec": duration_sec,
            "transcript": transcript,
        }
    elif result.reason == speechsdk.ResultReason.NoMatch:
        log.warning(f"No speech recognized in this segment")
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
        log.error(f"Recognition canceled: {details.reason}")
        log.error(f"Error details: {details.error_details}")
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
    # Step 1: Detect language segments if needed
    if not os.path.exists(args.segments) or args.force_detection:
        logger.info(f"Performing language detection on {args.audio}")

        # Use existing detect_languages function
        detect_languages(
            audio_file=args.audio,
            lid_host=args.lid_host,
            languages=args.languages,
            out_segments=args.segments,
            timeout_sec=args.timeout_sec,
            min_segment_sec=args.min_segment_sec,
            logger=logger,
        )
    else:
        logger.info(f"Using existing language segments from {args.segments}")

    # Load segments file
    with open(args.segments, "r", encoding="utf-8") as f:
        segments_data = json.load(f)

    # Step 2: Create STT host mapping
    stt_host_map = {}
    for mapping in args.stt_map:
        if "=" in mapping:
            lang, host = mapping.split("=", 1)
            stt_host_map[lang] = host

    if not stt_host_map:
        logger.error("No STT host mappings provided. Cannot transcribe segments.")
        return

    logger.info(f"STT host mapping: {stt_host_map}")

    # Step 3: Transcribe each segment
    transcribed_segments = []

    for segment in segments_data.get("Segments", []):
        language = segment.get("Language")
        start_time = segment.get("StartTimeHns")
        duration = segment.get("DurationHns")

        # Skip segments marked as skipped
        if segment.get("IsSkipped", False):
            logger.info(
                f"Skipping segment marked as skipped: {start_time/10000000:.2f}s to {(start_time + duration)/10000000:.2f}s"
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
