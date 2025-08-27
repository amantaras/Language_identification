#!/usr/bin/env python
"""
Test full transcription with language detection
"""
import os
import sys
import json
import logging
import argparse
from app.transcribe_with_language import process_audio_file

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("test_output.txt", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def parse_args():
    # Create a mock args object that simulates command line arguments
    parser = argparse.ArgumentParser(
        description="Test transcription with language detection"
    )
    parser.add_argument(
        "--audio",
        default="audio/samples/Scenario1_20250323_S1056_New Postpaid Activation1.mp3",
        help="Path to audio file",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["en-US", "ar-SA"],
        help="List of languages to detect",
    )
    parser.add_argument(
        "--lid-host", default="localhost:5003", help="Language ID container host"
    )
    parser.add_argument(
        "--stt-map",
        nargs="+",
        default=["en-US=localhost:5004", "ar-SA=localhost:5005"],
        help="STT host mapping",
    )
    parser.add_argument(
        "--segments",
        default="test_segments.json",
        help="Path to save/load language segments JSON",
    )
    parser.add_argument(
        "--output",
        default="test_transcript.json",
        help="Path to save final transcript JSON",
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
    parser.add_argument(
        "--verbose", action="store_true", default=True, help="Enable verbose logging"
    )

    return parser.parse_args()


def main():
    log.info("Starting test for full transcription with language detection")

    # Parse arguments
    args = parse_args()

    # Verify audio file exists
    if not os.path.exists(args.audio):
        log.error(f"Audio file not found: {args.audio}")
        return False

    # Check if containers are available
    log.info(f"Using Language ID container at: {args.lid_host}")
    for mapping in args.stt_map:
        lang, host = mapping.split("=", 1)
        log.info(f"Using STT container for {lang} at: {host}")

    try:
        # Process the audio file
        log.info(f"Processing audio file: {args.audio}")
        result = process_audio_file(args, log)

        # Verify output file was created
        if os.path.exists(args.output):
            with open(args.output, "r") as f:
                transcript_data = json.load(f)

            log.info(f"Transcript file created: {args.output}")
            log.info(
                f"Number of segments transcribed: {transcript_data.get('segment_count', 0)}"
            )
            log.info(f"Full transcript: {transcript_data.get('full_transcript', '')}")

            return True
        else:
            log.error(f"Output file not created: {args.output}")
            return False

    except Exception as e:
        log.error(f"Error in transcription process: {e}")
        import traceback

        log.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = main()
    if success:
        log.info("Test completed successfully")
    else:
        log.error("Test failed")
        sys.exit(1)
