import time
import json
import logging
import os
from typing import List, Optional
import azure.cognitiveservices.speech as speechsdk
from .segmentation import SegmentBuilder, HNS_PER_SECOND

# Note: This module performs a pass over the audio file to detect language switches.

# Import the audio conversion function
try:
    from pydub import AudioSegment

    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False


class LanguageDetectionResult:
    def __init__(self, segments_json_path: str):
        self.segments_json_path = segments_json_path


def convert_audio_to_wav(audio_path, logger=None):
    """Convert audio file to WAV format suitable for Speech SDK if not already WAV"""
    log = logger or logging.getLogger(__name__)

    # Check if file already has .wav extension with the right case
    if audio_path.lower().endswith(".wav"):
        log.debug(f"File already in WAV format: {audio_path}")
        return audio_path

    # Check if pydub is available
    if not PYDUB_AVAILABLE:
        log.warning(
            "pydub not available for audio conversion. Using original format which may cause errors."
        )
        return audio_path

    # Create output filename
    wav_path = os.path.splitext(audio_path)[0] + ".wav"
    log.info(f"Converting {audio_path} to {wav_path}")

    try:
        # Detect file type based on extension
        file_ext = os.path.splitext(audio_path)[1].lower()

        # Load audio file based on format
        if file_ext == ".mp3":
            audio = AudioSegment.from_mp3(audio_path)
        elif file_ext in [".ogg", ".oga"]:
            audio = AudioSegment.from_ogg(audio_path)
        elif file_ext == ".flac":
            audio = AudioSegment.from_file(audio_path, "flac")
        elif file_ext in [".wav", ".wave"]:
            # Should be caught by the earlier check, but just in case
            return audio_path
        else:
            # Try to load based on extension or fallback to autodetect
            try:
                audio = AudioSegment.from_file(audio_path, format=file_ext[1:])
            except Exception:
                log.warning(f"Unknown format {file_ext}, attempting to autodetect")
                audio = AudioSegment.from_file(audio_path)

        # Convert to mono if stereo
        if audio.channels > 1:
            log.info("Converting stereo to mono")
            audio = audio.set_channels(1)

        # Set sample rate to 16kHz if needed
        if audio.frame_rate != 16000:
            log.info(f"Converting sample rate from {audio.frame_rate}Hz to 16000Hz")
            audio = audio.set_frame_rate(16000)

        # Export as WAV with PCM format
        audio.export(wav_path, format="wav")
        log.info(f"Successfully converted to {wav_path}")

        return wav_path

    except Exception as e:
        log.error(f"Error converting audio: {e}")
        log.warning("Using original format which may cause errors.")
        return audio_path


def detect_languages(
    audio_file: str,
    lid_host: str,
    languages: List[str],
    out_segments: str,
    timeout_sec: Optional[float] = None,
    min_segment_sec: float = 0.0,
    logger: Optional[logging.Logger] = None,
) -> LanguageDetectionResult:
    """Perform continuous language identification and emit segments JSON.

    timeout_sec: Optional overall timeout to abort recognition loop.
    min_segment_sec: Drop segments shorter than this duration.
    """
    log = logger or logging.getLogger(__name__)

    # Convert audio to WAV format if necessary
    processed_audio_file = convert_audio_to_wav(audio_file, logger=log)

    # Keep track of whether we're using a converted file
    using_converted = processed_audio_file != audio_file
    if using_converted:
        log.info(f"Using converted audio file: {processed_audio_file}")

    # For container deployment, use direct host configuration which is more reliable
    log.debug(f"Connecting to LID container at: {lid_host}")

    # Ensure the host has the ws:// prefix required for container communication
    if not lid_host.startswith(("ws://", "wss://")):
        lid_host = f"ws://{lid_host}"
        log.debug(f"Added ws:// prefix. Using host URL: {lid_host}")

    # Create the speech config with the direct host method
    try:
        speech_config = speechsdk.SpeechConfig(host=lid_host)
        log.debug(f"Successfully created SpeechConfig with host: {lid_host}")
    except Exception as e:
        log.error(f"Failed to create SpeechConfig: {e}")
        raise

    # Add API key if provided (useful for hosted LID services)
    # In container scenarios, this is handled by the container configuration

    # Set the required continuous language detection mode
    speech_config.set_property(
        property_id=speechsdk.PropertyId.SpeechServiceConnection_LanguageIdMode,
        value="Continuous",
    )

    audio_config = speechsdk.audio.AudioConfig(filename=processed_audio_file)

    # Microsoft docs confirm: Use SourceLanguageRecognizer for container language ID
    auto_detect = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
        languages=languages
    )

    recognizer = speechsdk.SourceLanguageRecognizer(
        speech_config=speech_config,
        auto_detect_source_language_config=auto_detect,
        audio_config=audio_config,
    )

    min_hns = int(min_segment_sec * HNS_PER_SECOND)
    builder = SegmentBuilder(min_duration_hns=min_hns, logger=log)
    done = False
    last_end = 0

    def recognized(evt: speechsdk.SpeechRecognitionEventArgs):
        nonlocal last_end
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            detected = evt.result.properties.get(
                speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult
            )
            if detected:
                json_result = evt.result.properties.get(
                    speechsdk.PropertyId.SpeechServiceResponse_JsonResult
                )
                if json_result:
                    detail = json.loads(json_result)
                    start = detail.get("Offset", 0)
                    duration = detail.get("Duration", 0)
                    end_offset = start + duration if duration >= 0 else start
                    log.debug(f"LID event lang={detected} start={start} dur={duration}")
                    builder.on_detection(detected, start, end_offset)
                    last_end = max(last_end, end_offset)

    def canceled(evt: speechsdk.SpeechRecognitionCanceledEventArgs):
        nonlocal done
        log.error(
            f"Canceled: reason={evt.reason} error_code={evt.error_code} error_details={evt.error_details}"
        )
        done = True

    def stop_cb(evt):
        nonlocal done
        log.debug("Recognition stopped/canceled event received")
        done = True

    recognizer.recognized.connect(recognized)
    recognizer.session_stopped.connect(stop_cb)
    recognizer.canceled.connect(canceled)

    log.info("Starting continuous language identification")
    recognizer.start_continuous_recognition()
    start_time = time.time()
    while not done:
        if timeout_sec is not None and (time.time() - start_time) > timeout_sec:
            log.warning("Timeout reached; stopping recognition")
            break
        time.sleep(0.5)
    recognizer.stop_continuous_recognition()

    builder.finalize(final_end_hns=last_end)

    # Always use the original audio file path for the JSON output
    # This ensures that downstream processes refer to the original file
    builder.to_json(out_segments, audio_file)
    log.info(f"Wrote segments to {out_segments}")

    return LanguageDetectionResult(out_segments)
