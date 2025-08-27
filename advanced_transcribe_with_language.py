"""
multilingual_transcriber.py

Multilingual speech recognition with persistent container connections.
- Detects language segments (via LID container) and routes segments to the correct STT container.
- Uses canonical BCP-47 casing, mapping variants to supported targets.
- Reuses container SpeechConfig and performs robust recognition with fallback.
- Parallelizes segment transcription with a small thread pool.
"""

import os
import logging
import json
import time
import uuid
import wave
import tempfile
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Any

import azure.cognitiveservices.speech as speechsdk
from app.language_detect_improved import detect_languages


# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(
            "advanced_transcription_test.txt", mode="w", encoding="utf-8"
        ),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# -----------------------------
# Language utils
# -----------------------------
LANG_NORMALIZE_RE = re.compile(r"[^A-Za-z\-]")


def normalize_lang(code: str) -> str:
    """Return canonical BCP-47 casing, e.g., 'en-US' from 'en-us' / 'EN_us' / 'en'."""
    c = LANG_NORMALIZE_RE.sub("", (code or "").strip())
    if not c:
        return ""
    parts = c.replace("_", "-").split("-")
    if len(parts) == 1:
        return parts[0].lower()
    return f"{parts[0].lower()}-{parts[1].upper()}"


SUPPORTED_LANGS = {"en-US", "ar-SA"}


def collapse_to_supported_lang(lang: str) -> Optional[str]:
    """
    Map arbitrary BCP-47 to one of the supported container languages.
    Example: 'en', 'en-GB' -> 'en-US'; 'ar', 'ar-EG' -> 'ar-SA'
    """
    if not lang:
        return None
    base = lang.split("-")[0].lower()
    if base == "en":
        return "en-US"
    if base == "ar":
        return "ar-SA"
    lang_norm = normalize_lang(lang)
    return lang_norm if lang_norm in SUPPORTED_LANGS else None


# -----------------------------
# Container connection
# -----------------------------
class ContainerConnection:
    """Manages a persistent connection (SpeechConfig) to a speech container."""

    def __init__(self, host: str, language: Optional[str] = None):
        """
        Args:
            host: The WebSocket host address (e.g., 'ws://localhost:5004' or 'wss://...').
            language: Language code for STT containers (None for LID). Canonical casing will be applied.
        """
        self.host = host
        self.language = normalize_lang(language) if language else None
        self.speech_config: Optional[speechsdk.SpeechConfig] = None
        self.is_ready: bool = False
        self._initialize()

    def _initialize(self) -> bool:
        """Initialize the speech configuration."""
        log.info(
            f"Initializing connection to {self.host} for language: {self.language or 'Language Detection'}"
        )
        try:
            cfg = speechsdk.SpeechConfig(host=self.host)

            if self.language:
                cfg.speech_recognition_language = self.language

            # NOTE: do NOT set non-existent PropertyIds (e.g., Speech_SessionTimeout).
            # If you need segmentation/timeout tuning, use supported properties like:
            #  - Speech_SegmentationSilenceTimeoutMs
            #  - SpeechServiceConnection_InitialSilenceTimeoutMs
            # Only set them if you really need to.

            self.speech_config = cfg
            self.is_ready = True
            log.info(f"Successfully initialized connection to {self.host}")
            return True
        except Exception as e:
            self.is_ready = False
            log.error(f"Failed to initialize connection to {self.host}: {e}")
            return False

    # ---------- recognition helpers ----------

    def _recognize_continuous(
        self, audio_config: speechsdk.audio.AudioConfig, expected_duration_sec: float
    ) -> Dict[str, Any]:
        """
        Run continuous recognition to collect multiple utterances for a segment.
        Returns dict: {"status": "success"/"nomatch"/"error", "transcript"|"note"|"error"}.
        """
        try:
            recognizer = speechsdk.SpeechRecognizer(
                speech_config=self.speech_config, audio_config=audio_config
            )
        except Exception as e:
            return {"status": "error", "error": f"Recognizer init failed: {e}"}

        done = threading.Event()
        transcript_parts: List[str] = []
        err: List[Optional[str]] = [None]

        def on_recognized(evt):
            try:
                if (
                    evt.result.reason == speechsdk.ResultReason.RecognizedSpeech
                    and evt.result.text
                ):
                    transcript_parts.append(evt.result.text)
            except Exception as ex:
                err[0] = f"recognized handler error: {ex}"
                done.set()

        def on_canceled(evt):
            try:
                details = getattr(evt, "error_details", None)
                reason = getattr(evt, "reason", None)
                err[0] = f"Canceled ({reason}): {details}"
            finally:
                done.set()

        def on_stopped(_):
            done.set()

        recognizer.recognized.connect(on_recognized)
        recognizer.canceled.connect(on_canceled)
        recognizer.session_stopped.connect(on_stopped)

        try:
            recognizer.start_continuous_recognition()
            # Allow buffer beyond audio length
            wait_timeout = max(20.0, min(180.0, expected_duration_sec * 1.5 + 5.0))
            done.wait(wait_timeout)
        finally:
            try:
                recognizer.stop_continuous_recognition()
            except Exception:
                pass

        if err[0]:
            return {"status": "error", "error": err[0]}

        text = " ".join(transcript_parts).strip()
        if not text:
            return {"status": "nomatch", "note": "No speech recognized"}
        return {"status": "success", "transcript": text}

    def _recognize_once(
        self, audio_config: speechsdk.audio.AudioConfig
    ) -> Dict[str, Any]:
        """Single-utterance recognition fallback."""
        try:
            recognizer = speechsdk.SpeechRecognizer(
                speech_config=self.speech_config, audio_config=audio_config
            )
            res = recognizer.recognize_once()
        except Exception as e:
            return {"status": "error", "error": f"recognize_once failed to start: {e}"}

        if res.reason == speechsdk.ResultReason.RecognizedSpeech:
            return {"status": "success", "transcript": res.text}
        if res.reason == speechsdk.ResultReason.NoMatch:
            return {"status": "nomatch", "note": "No speech recognized"}
        if res.reason == speechsdk.ResultReason.Canceled:
            try:
                details = speechsdk.CancellationDetails.from_result(res)
                return {
                    "status": "error",
                    "error": f"Canceled ({details.reason}): {details.error_details}",
                }
            except Exception:
                return {
                    "status": "error",
                    "error": "Canceled (unknown): Unknown details",
                }
        return {"status": "error", "error": f"Unexpected result: {res.reason}"}

    # ---------- public API ----------

    def transcribe_segment(
        self, audio_file: str, start_sec: float, duration_sec: float
    ) -> Dict[str, Any]:
        """
        Transcribe a segment of audio using this connection.

        Args:
            audio_file: Path to the full audio file (wav)
            start_sec: Start time (seconds)
            duration_sec: Duration (seconds)
        """
        if not self.is_ready or not self.speech_config:
            return {"status": "error", "error": "Connection not initialized"}

        temp_dir = tempfile.gettempdir()
        unique_id = str(uuid.uuid4())[:8]
        temp_file = os.path.join(
            temp_dir,
            f"segment_{self.language or 'lid'}_{int(start_sec)}_{int(duration_sec)}_{unique_id}.wav",
        )

        try:
            # Extract segment to a temporary file (bounds-safe)
            with wave.open(audio_file, "rb") as wf:
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                frame_rate = wf.getframerate()
                total_frames = wf.getnframes()

                start_frame = max(0, int(start_sec * frame_rate))
                if start_frame >= total_frames:
                    return {"status": "nomatch", "note": "Segment starts after EOF"}

                num_frames_req = int(duration_sec * frame_rate)
                num_frames = max(0, min(num_frames_req, total_frames - start_frame))

                wf.setpos(start_frame)
                frames = wf.readframes(num_frames)

            with wave.open(temp_file, "wb") as segment_file:
                segment_file.setnchannels(channels)
                segment_file.setsampwidth(sample_width)
                segment_file.setframerate(frame_rate)
                segment_file.writeframes(frames)

            log.info(f"Created temporary segment file: {temp_file}")

            audio_config = speechsdk.audio.AudioConfig(filename=temp_file)

            # Try continuous first (captures multiple utterances), then fallback
            result = self._recognize_continuous(
                audio_config, expected_duration_sec=duration_sec
            )
            if result["status"] == "nomatch" or result["status"] == "error":
                log.debug(
                    f"Continuous recognition returned {result['status']}; trying recognize_once fallback..."
                )
                result = self._recognize_once(audio_config)

            if result["status"] == "success":
                log.info(f"Transcribed: {result['transcript']}")
            elif result["status"] == "nomatch":
                log.warning("No speech recognized in this segment")
            else:
                log.error(f"Recognition error: {result.get('error')}")

            return result

        except Exception as e:
            log.error(f"Error processing segment: {e}")
            return {"status": "error", "error": str(e)}

        finally:
            # Clean up temporary file with retry (Windows file handles can linger)
            for retry in range(3):
                try:
                    if os.path.exists(temp_file):
                        time.sleep(0.5)
                        os.remove(temp_file)
                        log.debug(f"Removed temporary file: {temp_file}")
                        break
                except Exception as ex:
                    log.warning(
                        f"Failed to remove temporary file (attempt {retry+1}/3): {ex}"
                    )
                    if retry == 2:
                        log.warning("Leaving temp file for system cleanup.")


# -----------------------------
# Orchestration
# -----------------------------
def advanced_transcribe_with_language():
    """End-to-end multilingual transcription with persistent connections."""
    log.info("=== TESTING ADVANCED MULTILINGUAL TRANSCRIPTION ===")

    # ---- Configuration (override with env vars if desired) ----
    lid_host = os.getenv(
        "LID_HOST", "ws://localhost:5003"
    )  # Language detection container
    audio_file = os.getenv(
        "AUDIO_FILE", "audio/samples/Arabic_english_mix_optimized.wav"
    )
    segments_file = os.getenv("SEGMENTS_FILE", "advanced_segments.json")
    transcript_file = os.getenv("TRANSCRIPT_FILE", "advanced_transcript.json")

    # Supported STT containers by canonical language key
    stt_host_map: Dict[str, str] = {
        "en-US": os.getenv("EN_US_STT_HOST", "ws://localhost:5004"),
        "ar-SA": os.getenv("AR_SA_STT_HOST", "ws://localhost:5005"),
    }

    # Languages to consider in LID
    languages = [normalize_lang("en-US"), normalize_lang("ar-SA")]

    log.info(f"Using language detection host: {lid_host}")
    log.info(f"Languages: {languages}")
    log.info(f"Audio file: {os.path.abspath(audio_file)}")
    log.info(f"Output segments: {os.path.abspath(segments_file)}")
    log.info(f"Output transcript: {os.path.abspath(transcript_file)}")
    log.info(f"STT host mapping: {stt_host_map}")

    # Guard inputs
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"Audio file not found: {os.path.abspath(audio_file)}")

    # ---- Step 1: Detect language segments (continuous mode for multi-language audio)
    log.info("\nStep 1: Detecting language segments...")
    _ = detect_languages(
        audio_file=audio_file,
        lid_host=lid_host,
        languages=languages,
        out_segments=segments_file,
        timeout_sec=60,
        logger=log,
        _force_continuous=True,
    )

    # ---- Step 2: Load segments
    log.info("\nStep 2: Loading language segments...")
    with open(segments_file, "r", encoding="utf-8") as f:
        segments_data = json.load(f)

    segments: List[Dict[str, Any]] = segments_data.get("segments", [])
    log.info(f"Found {len(segments)} language segments")

    # ---- Step 3: Persistent connections to STT containers
    log.info("\nStep 3: Initializing persistent connections to speech containers...")
    connections: Dict[str, ContainerConnection] = {
        lang_key: ContainerConnection(host=host, language=lang_key)
        for lang_key, host in stt_host_map.items()
    }

    # Remove any failed connections to avoid cancellations with None-details
    connections = {k: v for k, v in connections.items() if v.is_ready}
    if not connections:
        log.error("All STT connections failed to initialize. Aborting transcription.")
        final_result = {
            "audio_file": audio_file,
            "segment_count": 0,
            "transcribed_segments": [],
            "full_transcript": "",
        }
        with open(transcript_file, "w", encoding="utf-8") as f:
            json.dump(final_result, f, indent=2, ensure_ascii=False)
        return

    # ---- Step 4: Transcribe each segment (parallel, modest concurrency)
    log.info("\nStep 4: Transcribing each language segment...")

    def transcribe_one(segment: Dict[str, Any]) -> Dict[str, Any]:
        raw_language = normalize_lang(segment.get("language", ""))
        language = collapse_to_supported_lang(raw_language)

        start_time_hns = int(segment.get("start_hns", 0))
        end_time_hns = int(segment.get("end_hns", 0))
        duration_hns = max(0, end_time_hns - start_time_hns)

        start_sec = start_time_hns / 10_000_000.0  # HNS (100ns) -> seconds
        duration_sec = duration_hns / 10_000_000.0

        log.info(
            f"Processing segment: {raw_language} → {language} from {start_sec:.2f}s to {(start_sec + duration_sec):.2f}s"
        )

        rec: Dict[str, Any] = {
            "language": language or raw_language,
            "start_time": start_time_hns,
            "duration": duration_hns,
            "start_sec": start_sec,
            "duration_sec": duration_sec,
        }

        if duration_sec < 0.25:
            rec["transcript"] = ""
            rec["note"] = "Skipped very short segment"
            return rec

        if not language or language not in connections:
            log.warning(
                f"No connection for language '{raw_language}' → '{language}'. Skipping segment."
            )
            rec["transcript"] = ""
            rec["note"] = "Unsupported language"
            return rec

        result_seg = connections[language].transcribe_segment(
            audio_file=audio_file, start_sec=start_sec, duration_sec=duration_sec
        )

        if result_seg["status"] == "success":
            rec["transcript"] = result_seg["transcript"]
        elif result_seg["status"] == "nomatch":
            rec["transcript"] = ""
            rec["note"] = "No speech recognized"
        else:
            rec["transcript"] = ""
            rec["error"] = result_seg.get("error", "Unknown error")
        return rec

    transcribed_segments: List[Dict[str, Any]] = []
    max_workers = min(
        4, max(1, (os.cpu_count() or 2) // 2)
    )  # keep containers responsive
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(transcribe_one, seg) for seg in segments]
        for fut in as_completed(futures):
            try:
                transcribed_segments.append(fut.result())
            except Exception as e:
                log.error(f"Segment transcription failed: {e}")

    # ---- Step 5: Assemble final transcript
    log.info("\nStep 5: Creating final transcript...")
    transcribed_segments_sorted = sorted(
        transcribed_segments, key=lambda x: x["start_time"]
    )
    full_transcript = " ".join(
        [
            f"[{seg['language']}] {seg.get('transcript', '').strip()}"
            for seg in transcribed_segments_sorted
            if seg.get("transcript")
        ]
    ).strip()

    final_result = {
        "audio_file": audio_file,
        "segment_count": len(transcribed_segments_sorted),
        "transcribed_segments": transcribed_segments_sorted,
        "full_transcript": full_transcript,
    }

    try:
        with open(transcript_file, "w", encoding="utf-8") as f:
            json.dump(final_result, f, indent=2, ensure_ascii=False)
        log.info(f"Wrote complete transcript to {transcript_file}")
    except Exception as e:
        log.error(f"Failed to write transcript file: {e}")

    log.info(f"Full transcript: {final_result['full_transcript']}")
    log.debug(json.dumps(final_result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    advanced_transcribe_with_language()
