"""
test_direct_http.py

Test direct HTTP connection to Speech containers.
"""

import os
import json
import requests
import logging
import time
import wave

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def get_audio_duration(wav_path):
    """Get the duration of a wav file in seconds."""
    with wave.open(wav_path, "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        duration = frames / float(rate)
    return duration


def transcribe_with_rest(host, language, audio_file):
    """
    Use direct HTTP POST to transcribe audio using the Speech container REST API.

    Args:
        host: The container host address, e.g., "ws://localhost:5004" (will be converted to HTTP)
        language: The language code, e.g., "en-US"
        audio_file: Path to the audio file
    """
    # Convert WebSocket URL to HTTP
    if host.startswith("ws://"):
        host = "http://" + host[5:]
    elif host.startswith("wss://"):
        host = "https://" + host[6:]
    elif not host.startswith(("http://", "https://")):
        host = "http://" + host

    # Ensure no trailing slash
    host = host.rstrip("/")

    # The REST API endpoint for speech recognition
    url = f"{host}/speech/recognition/conversation/cognitiveservices/v1"

    # Parameters
    params = {
        "language": language,
        "format": "detailed",  # Get detailed output
    }

    # Headers
    headers = {
        "Content-Type": "audio/wav",
        "Accept": "application/json",
    }

    log.info(f"Sending audio to {url}")
    log.info(f"Audio file: {audio_file} ({get_audio_duration(audio_file):.2f} seconds)")

    try:
        # Open and read the audio file
        with open(audio_file, "rb") as f:
            audio_data = f.read()

        # Send the POST request
        start_time = time.time()
        response = requests.post(
            url,
            params=params,
            headers=headers,
            data=audio_data,
            timeout=30,  # 30 second timeout
        )
        elapsed = time.time() - start_time

        # Process the response
        if response.status_code == 200:
            try:
                result = response.json()
                text = result.get("DisplayText", "")
                log.info(f"Transcription completed in {elapsed:.2f}s: {text}")
                log.info(f"Full response: {json.dumps(result, indent=2)}")
                return text
            except json.JSONDecodeError:
                log.error(f"Invalid JSON response: {response.text[:200]}")
        else:
            log.error(f"Error {response.status_code}: {response.text[:200]}")

    except requests.exceptions.RequestException as e:
        log.error(f"Request failed: {str(e)}")

    return None


if __name__ == "__main__":
    # Test with English container
    audio_file = "audio/samples/Arabic_english_mix_optimized.wav"
    if not os.path.exists(audio_file):
        log.error(f"Audio file not found: {audio_file}")
    else:
        log.info("=== TESTING DIRECT HTTP CONNECTION TO SPEECH CONTAINERS ===")

        # Try English container
        log.info("\n=== English Container ===")
        transcribe_with_rest("http://localhost:5004", "en-US", audio_file)

        # Try Arabic container
        log.info("\n=== Arabic Container ===")
        transcribe_with_rest("http://localhost:5005", "ar-SA", audio_file)
