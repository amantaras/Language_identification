#!/usr/bin/env python
"""
Test for the updated language_detect.py module with automatic audio conversion
"""
import os
import logging
from app.language_detect import detect_languages

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("fixed_module_test.txt", mode="w"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def test_detect_languages():
    # Test parameters
    audio_file = "audio/samples/Scenario1_20250323_S1056_New Postpaid Activation1.mp3"
    lid_host = "localhost:5003"
    languages = ["en-US", "ar-SA"]
    out_segments = "fixed_module_segments.json"

    log.info(f"Testing detect_languages with audio file: {audio_file}")

    try:
        # Run the language detection with automatic conversion
        result = detect_languages(
            audio_file=audio_file,
            lid_host=lid_host,
            languages=languages,
            out_segments=out_segments,
            timeout_sec=60,
            logger=log,
        )

        log.info(f"✅ Language detection successful!")
        log.info(f"Segments written to: {result.segments_json_path}")

        # Verify that the segments file was created
        if os.path.exists(result.segments_json_path):
            log.info(
                f"Segments file exists: {os.path.getsize(result.segments_json_path)} bytes"
            )
            return True
        else:
            log.error(f"Segments file not found: {result.segments_json_path}")
            return False

    except Exception as e:
        log.error(f"❌ Error in detect_languages: {e}")
        return False


if __name__ == "__main__":
    log.info("Starting test of fixed language detection module")
    if test_detect_languages():
        log.info("✅ Test completed successfully")
    else:
        log.error("❌ Test failed")
