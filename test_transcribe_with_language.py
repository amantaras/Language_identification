"""
Test script to demonstrate the complete language detection and transcription flow
using the improved language detection.
"""

import os
import logging
import json
import wave
import tempfile
from app.language_detect_improved import detect_languages

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("transcribe_with_language_test.txt", mode="w"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


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

    log.info(
        f"Processing segment from {start_sec:.2f}s to {start_sec + duration_sec:.2f}s"
    )

    # For SDK compatibility, let's extract the segment to a temporary file
    import uuid

    temp_dir = tempfile.gettempdir()
    # Use a unique identifier to avoid conflicts with existing files
    unique_id = str(uuid.uuid4())[:8]
    temp_file = os.path.join(
        temp_dir,
        f"segment_{language}_{int(start_sec)}_{int(duration_sec)}_{unique_id}.wav",
    )

    try:
        # Open the original audio file
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

        # Configure speech recognition with the segment file
        import azure.cognitiveservices.speech as speechsdk

        speech_config = speechsdk.SpeechConfig(host=stt_host)
        speech_config.speech_recognition_language = language

        # Set a timeout for the connection to avoid hanging
        speech_config.set_property(
            speechsdk.PropertyId.Speech_SessionTimeout, "5000"
        )  # 5 seconds

        # Use the segment file directly without offset
        audio_config = speechsdk.audio.AudioConfig(
            filename=temp_file
        )  # Create recognizer
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config, audio_config=audio_config
        )

        # Perform recognition
        log.debug("Recognizing speech from segment file")
        result = recognizer.recognize_once()

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            transcript = result.text
            log.info(f"Transcribed: {transcript}")
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

    except Exception as e:
        log.error(f"Error processing segment: {e}")
        return {
            "language": language,
            "start_time": start_time,
            "duration": duration,
            "start_sec": start_sec,
            "duration_sec": duration_sec,
            "transcript": "",
            "error": str(e),
        }
    finally:
        # Clean up temporary file with a retry mechanism
        # Sometimes Windows holds file handles even after they're closed
        import time

        for retry in range(3):
            try:
                if os.path.exists(temp_file):
                    # Give Windows a moment to release the file handle
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
                        f"Could not remove temporary file after multiple attempts. It will be left for system cleanup."
                    )

    # Perform recognition
    log.debug(
        f"Recognizing speech from {start_sec:.2f}s to {start_sec + duration_sec:.2f}s"
    )
    result = recognizer.recognize_once()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        transcript = result.text
        log.info(f"Transcribed: {transcript}")
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


def test_transcribe_with_language():
    """Test the complete language detection and transcription flow."""
    log.info("=== TESTING COMPLETE LANGUAGE DETECTION AND TRANSCRIPTION ===")

    # Configuration
    lid_host = "ws://localhost:5003"  # LID container
    audio_file = "audio/samples/Arabic_english_mix_optimized.wav"
    languages = ["en-US", "ar-SA"]
    segments_file = "test_segments_improved.json"
    transcript_file = "test_transcript.json"

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

    # Step 3: Transcribe each segment
    log.info("\nStep 3: Transcribing each language segment...")
    transcribed_segments = []

    for segment in segments_data["segments"]:
        language = segment["language"]
        start_time = segment["start_hns"]
        end_time = segment["end_hns"]
        duration = end_time - start_time

        log.info(
            f"Processing segment: {language} from {start_time/10000000:.2f}s to {end_time/10000000:.2f}s"
        )

        result = transcribe_segment(
            audio_file=audio_file,
            language=language,
            start_time=start_time,
            duration=duration,
            stt_host_map=stt_host_map,
            logger=log,
        )

        if result:
            transcribed_segments.append(result)

    # Step 4: Create final transcript
    log.info("\nStep 4: Creating final transcript...")
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
    test_transcribe_with_language()
