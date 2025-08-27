"""
Advanced test script for multilingual speech recognition using persistent container connections.
This approach establishes connections to all speech containers at startup and reuses them for all segments.
"""

import os
import logging
import json
import time
import uuid
import wave
import tempfile
from typing import Dict, List, Optional, Any
import azure.cognitiveservices.speech as speechsdk
from app.language_detect_improved import detect_languages

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("advanced_transcription_test.txt", mode="w"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


class ContainerConnection:
    """Manages a persistent connection to a speech container."""

    def __init__(self, host: str, language: Optional[str] = None):
        """
        Initialize a connection to a speech container.

        Args:
            host: The WebSocket host address (ws://hostname:port)
            language: The language code for STT containers (None for LID)
        """
        self.host = host
        self.language = language
        self.speech_config = None
        self.recognizer = None

        # Initialize connection
        self._initialize()

    def _initialize(self):
        """Initialize the speech configuration and recognizer."""
        log.info(
            f"Initializing connection to {self.host} for language: {self.language or 'Language Detection'}"
        )

        try:
            # Create speech config
            self.speech_config = speechsdk.SpeechConfig(host=self.host)

            # Set language for STT containers
            if self.language:
                self.speech_config.speech_recognition_language = self.language

            # Set timeout
            self.speech_config.set_property(
                speechsdk.PropertyId.Speech_SessionTimeout, "10000"  # 10 seconds
            )

            log.info(f"Successfully initialized connection to {self.host}")
            return True

        except Exception as e:
            log.error(f"Failed to initialize connection to {self.host}: {e}")
            return False

    def transcribe_segment(
        self, audio_file: str, start_sec: float, duration_sec: float
    ) -> dict:
        """
        Transcribe a segment of audio using this connection.

        Args:
            audio_file: Path to the full audio file
            start_sec: Start time in seconds
            duration_sec: Duration in seconds

        Returns:
            A dictionary with transcription results
        """
        # Create a temporary file for the segment
        temp_dir = tempfile.gettempdir()
        unique_id = str(uuid.uuid4())[:8]  # Use unique ID to avoid conflicts
        temp_file = os.path.join(
            temp_dir,
            f"segment_{self.language or 'lid'}_{int(start_sec)}_{int(duration_sec)}_{unique_id}.wav",
        )

        try:
            # Extract segment to temporary file
            with wave.open(audio_file, "rb") as wf:
                # Get parameters
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                frame_rate = wf.getframerate()

                # Calculate frame positions
                start_frame = int(start_sec * frame_rate)
                num_frames = int(duration_sec * frame_rate)

                # Set position to start frame
                wf.setpos(start_frame)

                # Read frames for the segment
                frames = wf.readframes(num_frames)

                # Create a new file for the segment
                with wave.open(temp_file, "wb") as segment_file:
                    segment_file.setnchannels(channels)
                    segment_file.setsampwidth(sample_width)
                    segment_file.setframerate(frame_rate)
                    segment_file.writeframes(frames)

            log.info(f"Created temporary segment file: {temp_file}")

            # Create audio config for the segment
            audio_config = speechsdk.audio.AudioConfig(filename=temp_file)

            # Create recognizer
            recognizer = speechsdk.SpeechRecognizer(
                speech_config=self.speech_config, audio_config=audio_config
            )

            # Perform recognition
            log.debug(f"Recognizing speech from segment file using {self.host}")
            result = recognizer.recognize_once()

            # Process result
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                transcript = result.text
                log.info(f"Transcribed: {transcript}")
                return {"status": "success", "transcript": transcript}
            elif result.reason == speechsdk.ResultReason.NoMatch:
                log.warning("No speech recognized in this segment")
                return {"status": "nomatch", "note": "No speech recognized"}
            elif result.reason == speechsdk.ResultReason.Canceled:
                details = speechsdk.CancellationDetails(result)
                log.error(f"Recognition canceled: {details.reason}")
                log.error(f"Error details: {details.error_details}")
                return {"status": "error", "error": details.error_details}

        except Exception as e:
            log.error(f"Error processing segment: {e}")
            return {"status": "error", "error": str(e)}

        finally:
            # Clean up temporary file with retry
            self._cleanup_temp_file(temp_file)

    def _cleanup_temp_file(self, temp_file: str):
        """Clean up a temporary file with retry."""
        for retry in range(3):
            try:
                if os.path.exists(temp_file):
                    # Give Windows a moment to release file handles
                    time.sleep(0.5)
                    os.remove(temp_file)
                    log.debug(f"Removed temporary file: {temp_file}")
                    break
            except Exception as e:
                log.warning(
                    f"Failed to remove temporary file (attempt {retry+1}/3): {e}"
                )
                if retry == 2:  # Last attempt
                    log.warning(
                        "Could not remove temporary file after multiple attempts. It will be left for system cleanup."
                    )


def advanced_transcribe_with_language():
    """Test advanced multilingual transcription with persistent connections."""
    log.info("=== TESTING ADVANCED MULTILINGUAL TRANSCRIPTION ===")

    # Configuration
    lid_host = "ws://localhost:5003"  # LID container
    audio_file = "audio/samples/Arabic_english_mix_optimized.wav"
    languages = ["en-US", "ar-SA"]
    segments_file = "advanced_segments.json"
    transcript_file = "advanced_transcript.json"

    # STT host mapping
    stt_host_map = {
        "en-us": "ws://localhost:5004",  # English STT container
        "ar-sa": "ws://localhost:5005",  # Arabic STT container
    }

    log.info(f"Using language detection host: {lid_host}")
    log.info(f"Languages: {languages}")
    log.info(f"Audio file: {os.path.abspath(audio_file)}")
    log.info(f"Output segments: {os.path.abspath(segments_file)}")
    log.info(f"Output transcript: {os.path.abspath(transcript_file)}")
    log.info(f"STT host mapping: {stt_host_map}")

    # Step 1: Detect language segments
    log.info("\nStep 1: Detecting language segments...")
    result = detect_languages(
        audio_file=audio_file,
        lid_host=lid_host,
        languages=languages,
        out_segments=segments_file,
        timeout_sec=60,  # Longer timeout to process the entire file
        logger=log,
        # Force continuous mode to detect multiple languages
        _force_continuous=True,
    )

    # Step 2: Load the segments file
    log.info("\nStep 2: Loading language segments...")
    with open(segments_file, "r", encoding="utf-8") as f:
        segments_data = json.load(f)

    log.info(f"Found {len(segments_data['segments'])} language segments")

    # Step 3: Initialize persistent connections
    log.info("\nStep 3: Initializing persistent connections to speech containers...")
    connections = {}
    for lang, host in stt_host_map.items():
        connections[lang] = ContainerConnection(host=host, language=lang)

    # Step 4: Transcribe each segment using persistent connections
    log.info("\nStep 4: Transcribing each language segment...")
    transcribed_segments = []

    for segment in segments_data["segments"]:
        language = segment["language"]
        start_time = segment["start_hns"]
        end_time = segment["end_hns"]
        duration = end_time - start_time

        # Convert from HNS to seconds
        start_sec = start_time / 10000000
        duration_sec = duration / 10000000

        log.info(
            f"Processing segment: {language} from {start_sec:.2f}s to {start_sec + duration_sec:.2f}s"
        )

        # Get connection for this language
        if language in connections:
            connection = connections[language]

            # Transcribe the segment
            result = connection.transcribe_segment(
                audio_file=audio_file, start_sec=start_sec, duration_sec=duration_sec
            )

            # Create segment record
            segment_record = {
                "language": language,
                "start_time": start_time,
                "duration": duration,
                "start_sec": start_sec,
                "duration_sec": duration_sec,
            }

            # Add transcription result
            if result["status"] == "success":
                segment_record["transcript"] = result["transcript"]
            elif result["status"] == "nomatch":
                segment_record["transcript"] = ""
                segment_record["note"] = "No speech recognized"
            else:
                segment_record["transcript"] = ""
                segment_record["error"] = result.get("error", "Unknown error")

            transcribed_segments.append(segment_record)
        else:
            log.warning(
                f"No connection available for language {language}. Skipping segment."
            )

    # Step 5: Create final transcript
    log.info("\nStep 5: Creating final transcript...")
    final_result = {
        "audio_file": audio_file,
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
    with open(transcript_file, "w", encoding="utf-8") as f:
        json.dump(final_result, f, indent=2)

    log.info(f"Wrote complete transcript to {transcript_file}")
    log.info(f"Full transcript: {final_result['full_transcript']}")


if __name__ == "__main__":
    advanced_transcribe_with_language()
