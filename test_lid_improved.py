"""
Test script to verify the improved language detection implementation
"""

import os
import logging
import json
import socket
from app.language_detect_improved import detect_languages, detect_language_sync

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("lid_improved_test.txt", mode="w"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def check_container_status(host_str):
    """Check if the container is running and accessible"""
    # Strip the ws:// prefix if present
    host = host_str.replace("ws://", "").replace("wss://", "")
    # Split into host and port
    parts = host.split(":")
    host = parts[0]
    port = int(parts[1]) if len(parts) > 1 else 5000

    log.info(f"Checking container at {host}:{port}...")

    try:
        # Try to connect to the container
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        result = sock.connect_ex((host, port))
        sock.close()

        if result == 0:
            log.info(f"[SUCCESS] Container at {host}:{port} is running and accessible")

            # Additional check - try an HTTP request to see if the container is responding
            try:
                import requests

                response = requests.get(f"http://{host}:{port}/", timeout=3)
                log.debug(f"HTTP response status: {response.status_code}")

                if response.status_code >= 200 and response.status_code < 300:
                    log.info("[SUCCESS] Container is responding to HTTP requests")
                else:
                    log.warning(
                        f"[WARNING] Container responded with status {response.status_code}"
                    )
            except Exception as e:
                log.warning(f"[WARNING] Could not make HTTP request to container: {e}")
                log.info(
                    "This is expected if the container only accepts WebSocket connections"
                )

            return True
        else:
            log.error(
                f"[ERROR] Container at {host}:{port} is NOT accessible (error code: {result})"
            )
            log.error("Make sure the container is running with: docker ps -a")
            log.error(
                "You might need to start the containers with: ./scripts/run-containers-simple.ps1"
            )
            return False
    except Exception as e:
        log.error(f"[ERROR] Error checking container status: {e}")
        return False


def diagnose_container_issue(host_str):
    """Perform a comprehensive diagnosis of container issues"""
    host = host_str.replace("ws://", "").replace("wss://", "")
    parts = host.split(":")
    host = parts[0]
    port = int(parts[1]) if len(parts) > 1 else 5000

    log.info("\n===== CONTAINER DIAGNOSTIC REPORT =====")

    # 1. Check basic connectivity
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        result = sock.connect_ex((host, port))
        sock.close()

        if result == 0:
            log.info("[DIAGNOSTIC] TCP port is open")
        else:
            log.error(f"[DIAGNOSTIC] TCP port is closed (error code: {result})")
            log.error(
                "[RECOMMENDATION] Start the container with: ./scripts/run-containers-simple.ps1"
            )
            return
    except Exception as e:
        log.error(f"[DIAGNOSTIC] TCP connectivity check error: {e}")
        return

    # 2. Check container logs
    log.info(
        "[DIAGNOSTIC] Use 'docker logs speech-lid' to check container logs for errors"
    )

    # 3. Check container state
    log.info(
        "[DIAGNOSTIC] Use 'docker inspect speech-lid' to check the container state"
    )

    # 4. Suggest solution
    log.info(
        "[RECOMMENDATION] If the container is running but not responding properly:"
    )
    log.info("  1. Stop the container: ./scripts/stop-containers.ps1")
    log.info("  2. Start it again: ./scripts/run-containers-simple.ps1")
    log.info("  3. Verify your API key and billing endpoint in .env")
    log.info("=================================\n")


def test_improved_lid():
    """Test the improved language detection implementation"""
    log.info("=== TESTING IMPROVED LANGUAGE DETECTION ===")

    # Configuration
    host = "ws://localhost:5003"  # LID container
    audio_file = "audio/samples/Arabic_english_mix_optimized.wav"
    languages = ["en-US", "ar-SA"]
    output_segments = "test_segments_improved.json"

    log.info(f"Using host: {host}")
    log.info(f"Languages: {languages}")
    log.info(f"Audio file: {os.path.abspath(audio_file)}")
    log.info(f"Output segments: {os.path.abspath(output_segments)}")

    # First check if the container is running
    if not check_container_status(host):
        log.error(
            "Please make sure the language detection container is running before testing."
        )
        log.info("You can run the containers with: ./scripts/run-containers-simple.ps1")
        log.info("Continuing with tests anyway, but they will likely fail...")

    # First test the simpler sync method
    log.info("\nTesting synchronous language detection...")
    detected_lang = detect_language_sync(audio_file, host, languages, logger=log)

    if detected_lang:
        log.info(f"[SUCCESS] SYNC TEST SUCCEEDED! Detected language: {detected_lang}")
    else:
        log.error("[ERROR] SYNC TEST FAILED! No language detected.")
        # Run diagnostics since we had a failure
        diagnose_container_issue(host)

    # Now test the full detection with segments output
    log.info("\nTesting full language detection with segments...")

    # Force the use of continuous mode by modifying detect_languages call
    # Set a longer timeout to ensure we process the entire audio file
    result = detect_languages(
        audio_file=audio_file,
        lid_host=host,
        languages=languages,
        out_segments=output_segments,
        timeout_sec=60,  # Longer timeout to process the entire file
        logger=log,
        # Skip the synchronous detection by passing a special flag
        _force_continuous=True,
    )

    if result and os.path.exists(result.segments_json_path):
        log.info(
            f"[SUCCESS] FULL TEST SUCCEEDED! Created segments file: {result.segments_json_path}"
        )

        # Display the segments
        try:
            with open(result.segments_json_path, "r") as f:
                segments = json.load(f)
                log.info(f"Segments: {json.dumps(segments, indent=2)}")
        except Exception as e:
            log.error(f"Error reading segments file: {e}")

        return True
    else:
        log.error("[ERROR] FULL TEST FAILED! No segments file created.")
        return False


if __name__ == "__main__":
    test_improved_lid()
