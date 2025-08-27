"""
working_container_solution.py

Implementation using WebSocket connection pattern that's proven to work with Speech containers.
"""

import os
import wave
import tempfile
import uuid
import logging
import time
import json
import azure.cognitiveservices.speech as speechsdk

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(
            "working_container_solution.log", mode="w", encoding="utf-8"
        ),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def extract_wav_segment(src_wav, start_sec=0, duration_sec=5):
    """Extract a segment from a WAV file and save to a temp file."""
    tmp = os.path.join(
        tempfile.gettempdir(),
        f"segment_{uuid.uuid4().hex}_{int(start_sec*1000)}_{int(duration_sec*1000)}.wav",
    )

    with wave.open(src_wav, "rb") as src:
        n_channels = src.getnchannels()
        sampwidth = src.getsampwidth()
        framerate = src.getframerate()

        # Calculate frames
        start_frame = int(start_sec * framerate)
        n_frames = int(duration_sec * framerate)

        # Ensure we don't read past the end
        total_frames = src.getnframes()
        if start_frame >= total_frames:
            log.warning(f"Start position {start_sec}s is beyond end of file")
            return None

        if start_frame + n_frames > total_frames:
            n_frames = total_frames - start_frame
            log.warning(f"Adjusted duration to {n_frames/framerate:.2f}s")

        # Read the frames
        src.setpos(start_frame)
        frames = src.readframes(n_frames)

        # Write to the temp file
        with wave.open(tmp, "wb") as dst:
            dst.setnchannels(n_channels)
            dst.setsampwidth(sampwidth)
            dst.setframerate(framerate)
            dst.writeframes(frames)

    log.info(f"Created segment file: {tmp}")
    return tmp


def transcribe_segment(audio_file, language, host):
    """Transcribe a short audio segment using the Speech SDK."""
    log.info(f"Transcribing {audio_file} with {language} on {host}")

    # Ensure the host has ws:// prefix
    if not host.startswith(("ws://", "wss://")):
        host = f"ws://{host}"

    # Create speech config - host only, no path
    speech_config = speechsdk.SpeechConfig(host=host)
    speech_config.speech_recognition_language = language

    # Create audio config
    audio_config = speechsdk.audio.AudioConfig(filename=audio_file)

    # Create recognizer
    log.info(f"Creating recognizer with host={host}, language={language}")
    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, audio_config=audio_config
    )

    # Use synchronous recognition for simplicity
    log.info("Starting recognition...")
    result = recognizer.recognize_once()

    # Process result
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        log.info(f"RECOGNIZED: {result.text}")
        return result.text
    elif result.reason == speechsdk.ResultReason.NoMatch:
        log.warning(f"NOMATCH: {result.no_match_details.reason}")
        return ""
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation = result.cancellation_details
        log.error(f"CANCELED: {cancellation.reason}")
        if cancellation.reason == speechsdk.CancellationReason.Error:
            log.error(f"Error details: {cancellation.error_details}")
        return ""


def process_audio(audio_file, segments):
    """Process audio file in segments using container STT."""
    if not os.path.exists(audio_file):
        log.error(f"Audio file not found: {audio_file}")
        return

    log.info(f"Processing audio file: {audio_file}")
    log.info(f"Total segments to process: {len(segments)}")

    results = []

    for i, seg in enumerate(segments):
        log.info(
            f"\n=== Segment {i+1}/{len(segments)}: {seg['language']} at {seg['start']}s ==="
        )

        # Extract segment
        segment_file = extract_wav_segment(
            audio_file, start_sec=seg["start"], duration_sec=seg["duration"]
        )

        if not segment_file:
            log.error("Failed to extract segment")
            continue

        try:
            # Transcribe segment
            text = transcribe_segment(segment_file, seg["language"], seg["host"])

            if text:
                results.append(
                    {
                        "segment": i + 1,
                        "start": seg["start"],
                        "duration": seg["duration"],
                        "language": seg["language"],
                        "text": text,
                    }
                )

        finally:
            # Clean up temp file
            try:
                if segment_file and os.path.exists(segment_file):
                    os.remove(segment_file)
            except Exception as e:
                log.warning(f"Failed to remove temp file: {e}")

    # Print results
    log.info("\n=== TRANSCRIPTION RESULTS ===")
    for res in results:
        log.info(
            f"[{res['segment']}] {res['language']} ({res['start']}-{res['start']+res['duration']}s): {res['text']}"
        )

    return results


if __name__ == "__main__":
    audio_file = "audio/samples/Arabic_english_mix_optimized.wav"

    # Define known language segments (normally these would come from LID)
    segments = [
        {"start": 0, "duration": 5, "language": "en-US", "host": "ws://localhost:5004"},
        {
            "start": 10,
            "duration": 5,
            "language": "ar-SA",
            "host": "ws://localhost:5005",
        },
        {
            "start": 20,
            "duration": 5,
            "language": "en-US",
            "host": "ws://localhost:5004",
        },
    ]

    log.info("=== PROCESSING AUDIO WITH CONTAINER STT ===")
    results = process_audio(audio_file, segments)

    if results:
        log.info("Processing completed successfully.")
    else:
        log.error("Processing failed or no results were generated.")
