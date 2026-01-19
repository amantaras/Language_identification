"""
disconnected_language_detector.py — LIVE LID + LIVE CLOUD STT

Overview
--------
This module implements a real-time, streaming pipeline for mixed-language audio:
1. Language Identification (LID) via Azure Speech CONTAINER (WebSocket or HTTP)
2. Speech-to-Text (STT) via Azure Speech CLOUD SDK

Architecture
------------
┌─────────────────┐    WebSocket/HTTP    ┌───────────────────┐
│   Audio File    │ ──────────────────► │  LID Container    │
│   (WAV)         │                      │  (Language ID)    │
└─────────────────┘                      └────────┬──────────┘
                                                  │ Language events
                                                  ▼
                                         ┌───────────────────┐
                                         │  Event Router     │
                                         │  (on_lid_segment) │
                                         └────────┬──────────┘
                                                  │ Start/Stop
                                                  ▼
┌─────────────────┐    Push Stream       ┌───────────────────┐
│   Audio Chunks  │ ──────────────────► │  Cloud STT        │
│   (16kHz PCM)   │                      │  (per-language)   │
└─────────────────┘                      └───────────────────┘

Environment Variables
---------------------
LID Container Billing:
    - SPEECH_BILLING_ENDPOINT: Billing endpoint for container
    - SPEECH_API_KEY: API key for container billing

Cloud STT Authentication (choose one mode):
    - Region mode: SPEECH_KEY/AZURE_SPEECH_KEY + SPEECH_REGION/AZURE_SPEECH_REGION
    - Endpoint mode: SPEECH_KEY/AZURE_SPEECH_KEY + SPEECH_ENDPOINT/AZURE_SPEECH_ENDPOINT

LID Protocol:
    - LID_PROTOCOL: Connection protocol (ws, websocket, wss, http, https)

Usage Example
-------------
python disconnected_language_detector.py \\
    --audio "audio/samples/mixed_language.wav" \\
    --languages en-US ar-SA \\
    --lid-host localhost:5003 \\
    --segments segments.json \\
    --output transcript.json \\
    --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import threading
import wave
from typing import Dict, List, Optional, Tuple

# NOTE: audioop is deprecated in Python 3.13; OK on Python 3.12.
import audioop

import azure.cognitiveservices.speech as speechsdk


# =============================================================================
# ENVIRONMENT CONFIGURATION
# =============================================================================

def _load_dotenv(path: str = ".env", log: Optional[logging.Logger] = None) -> None:
    """
    Load environment variables from a .env file.
    
    Only sets variables that are not already defined in the environment,
    allowing real environment variables to take precedence.
    
    Args:
        path: Path to the .env file (default: ".env")
        log: Optional logger for debug output
    
    File Format:
        KEY=VALUE
        # Comments are ignored
        QUOTED_VALUE="with spaces"
    """
    if not os.path.isfile(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                if key in os.environ:
                    continue  # Don't override existing env vars
                val = val.strip().strip('"').strip("'")
                os.environ[key] = val
        if log:
            log.debug("Loaded variables from .env")
    except Exception as e:  # noqa: BLE001
        if log:
            log.debug(f"Failed loading .env: {e}")


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(verbose: bool) -> logging.Logger:
    """
    Configure logging for the application.
    
    Args:
        verbose: If True, enables DEBUG level logging; otherwise INFO level
    
    Returns:
        Configured logger instance
    
    Outputs to:
        - Console (stdout)
        - File: fixed_module_test.txt (overwritten each run)
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("fixed_module_test.txt", mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("lid_cloud_stt_live")


# =============================================================================
# TIME & LANGUAGE CONVERSION UTILITIES
# =============================================================================

# Azure Speech SDK uses 100-nanosecond units (HNS = Hundred NanoSeconds)
HNS_PER_SEC = 10_000_000


def hns_to_sec(hns: int) -> float:
    """Convert hundred-nanosecond units to seconds."""
    return hns / HNS_PER_SEC


def sec_to_hns(sec: float) -> int:
    """Convert seconds to hundred-nanosecond units."""
    return int(sec * HNS_PER_SEC)


def canon_lang(lang: str) -> str:
    """
    Canonicalize a language code to BCP-47 format.
    
    Examples:
        "en_us" -> "en-US"
        "AR-SA" -> "ar-SA"
        "en" -> "en"
    
    Args:
        lang: Raw language code string
    
    Returns:
        Canonicalized language code (e.g., "en-US", "ar-SA")
    """
    if not lang:
        return ""
    parts = lang.replace("_", "-").split("-")
    if len(parts) == 1:
        return parts[0].lower()
    return f"{parts[0].lower()}-{parts[1].upper()}"


def collapse_supported(code: str) -> str:
    """
    Map language codes to supported Azure STT locales.
    
    This function normalizes variant codes to the primary supported locale.
    For example, "en-GB" would be mapped to "en-US" since that's what
    we're configured to support.
    
    Args:
        code: Language code to normalize
    
    Returns:
        Supported locale code (e.g., "en-US" or "ar-SA")
    """
    if not code:
        return ""
    low = code.lower()
    if low.startswith("en"):
        return "en-US"
    if low.startswith("ar"):
        return "ar-SA"
    return canon_lang(code)


# =============================================================================
# WAV FILE UTILITIES
# =============================================================================

def _wav_props(path: str) -> Tuple[int, int, int, int]:
    """
    Read WAV file properties.
    
    Args:
        path: Path to WAV file
    
    Returns:
        Tuple of (channels, sample_width_bytes, sample_rate, total_frames)
    """
    with wave.open(path, "rb") as wf:
        return wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes()


def _read_frames(path: str, start_frame: int, frames: int) -> bytes:
    """
    Read a chunk of audio frames from a WAV file.
    
    Args:
        path: Path to WAV file
        start_frame: Starting frame position
        frames: Number of frames to read
    
    Returns:
        Raw audio bytes
    """
    with wave.open(path, "rb") as wf:
        wf.setpos(start_frame)
        return wf.readframes(frames)


# =============================================================================
# LID STREAM - LANGUAGE IDENTIFICATION VIA CONTAINER
# =============================================================================

class LIDStream:
    """
    Continuous Language Identification (LID) stream using Azure Speech container.
    
    This class connects to a local Azure Speech LID container via WebSocket or HTTP
    and streams audio for real-time language detection. As languages are detected,
    it emits events via callbacks.
    
    Architecture:
        Audio File → LID Container → Language Events → Callbacks
    
    Events Emitted:
        - on_segment(language, start_hns, end_hns): Called for each detected language span
        - on_done(): Called when LID session ends (audio complete or stopped)
    
    Protocol Selection (via LID_PROTOCOL env var):
        - ws/websocket: WebSocket (recommended for real-time)
        - wss: WebSocket Secure
        - http: HTTP REST
        - https: HTTPS REST
    
    Args:
        audio_file: Path to WAV audio file
        languages: List of candidate language codes (e.g., ["en-US", "ar-SA"])
        lid_host: Container host:port (e.g., "localhost:5003")
        min_segment_sec: Minimum segment duration for logging (0 = log all)
        log: Logger instance
        on_segment: Callback for language detection events
        on_done: Callback when LID completes
    
    Example:
        lid = LIDStream(
            audio_file="audio.wav",
            languages=["en-US", "ar-SA"],
            lid_host="localhost:5003",
            on_segment=handle_language,
            on_done=handle_complete
        )
        lid.start()
    """

    def __init__(
        self,
        audio_file: str,
        languages: List[str],
        lid_host: str,
        min_segment_sec: float,
        log: logging.Logger,
        on_segment: callable,
        on_done: callable,
    ):
        # -----------------------------------------------------------------
        # Determine connection protocol from environment
        # Supported: ws, websocket, wss, http, https
        # -----------------------------------------------------------------
        lid_protocol = os.getenv("LID_PROTOCOL", "http").strip().lower()
        if not lid_host.startswith(("ws://", "wss://", "http://", "https://")):
            if lid_protocol in ("ws", "websocket", "wss"):
                lid_host = "ws://" + lid_host
            else:
                lid_host = "http://" + lid_host
        log.info(f"LID protocol: {lid_protocol}, host: {lid_host}")
        
        self.audio_file = audio_file
        self.languages = [canon_lang(l) for l in languages]
        self.lid_host = lid_host
        self.min_segment_hns = sec_to_hns(min_segment_sec) if min_segment_sec > 0 else 0
        self.log = log
        self.on_segment = on_segment
        self.on_done = on_done

        # -----------------------------------------------------------------
        # Configure Azure Speech SDK for LID container
        # Note: Use host= parameter only (no path like /speech/)
        # -----------------------------------------------------------------
        self.speech_config = speechsdk.SpeechConfig(host=self.lid_host)
        self.speech_config.set_property(
            speechsdk.PropertyId.SpeechServiceConnection_LanguageIdMode, "Continuous"
        )
        
        # Configure audio input from file
        self.audio_config = speechsdk.audio.AudioConfig(filename=self.audio_file)
        
        # Set up auto-detect for specified languages
        self.auto_detect = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
            languages=self.languages
        )
        
        # -----------------------------------------------------------------
        # Create the SourceLanguageRecognizer for continuous LID
        # -----------------------------------------------------------------
        self.recognizer = speechsdk.SourceLanguageRecognizer(
            speech_config=self.speech_config,
            auto_detect_source_language_config=self.auto_detect,
            audio_config=self.audio_config,
        )
        
        # -----------------------------------------------------------------
        # Connect event handlers
        # -----------------------------------------------------------------
        # RECOGNIZED: Fired when a language segment is confidently detected
        self.recognizer.recognized.connect(self._on_recognized)
        
        # SESSION_STOPPED: Fired when the recognition session ends normally
        self.recognizer.session_stopped.connect(self._on_stopped)
        
        # CANCELED: Fired on errors or when recognition is canceled
        self.recognizer.canceled.connect(self._on_stopped)

    def start(self):
        """Start continuous language identification."""
        self.log.info("Starting continuous LID…")
        self.recognizer.start_continuous_recognition()

    def _on_recognized(self, evt):
        """
        Event handler: Language recognition result received.
        
        This is called each time the LID container detects a language span.
        Parses the JSON result to extract timing and language info.
        
        Event Properties:
            evt.result.reason: Should be RecognizedSpeech for valid detections
            evt.result.properties: Contains language and timing JSON
        """
        if evt.result.reason != speechsdk.ResultReason.RecognizedSpeech:
            return
            
        # Extract detected language from result properties
        detected = evt.result.properties.get(
            speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult
        )
        
        # Get raw JSON response for timing information
        raw = evt.result.properties.get(
            speechsdk.PropertyId.SpeechServiceResponse_JsonResult
        )
        
        if not detected or not raw:
            return
            
        try:
            js = json.loads(raw)
            start_hns = int(js.get("Offset", 0))
            dur_hns = int(js.get("Duration", 0))
            
            if dur_hns <= 0:
                return
                
            end_hns = start_hns + dur_hns
            
            # Log based on segment duration
            if self.min_segment_hns and dur_hns < self.min_segment_hns:
                self.log.debug(
                    f"Detected {detected} short {hns_to_sec(dur_hns):.2f}s "
                    f"{hns_to_sec(start_hns):.2f}s→{hns_to_sec(end_hns):.2f}s"
                )
            else:
                self.log.debug(
                    f"Detected {canon_lang(detected)} from {hns_to_sec(start_hns):.2f}s to {hns_to_sec(end_hns):.2f}s"
                )
            
            # Emit segment event to callback
            self.on_segment(canon_lang(detected), start_hns, end_hns)
            
        except Exception as e:
            self.log.debug(f"LID JSON parse error: {e}")

    def _on_stopped(self, _):
        """
        Event handler: LID session stopped or canceled.
        
        Called when:
            - Audio file processing completes
            - Recognition is explicitly stopped
            - An error occurs
        
        Cleans up the recognizer and notifies via on_done callback.
        """
        self.log.debug("LID session stopped")
        try:
            self.recognizer.stop_continuous_recognition()
        except Exception:
            pass
        self.on_done()


# =============================================================================
# AZURE CLOUD STT - SPEECH-TO-TEXT CONFIGURATION
# =============================================================================

class AzureCloudSTT:
    """
    Azure Cloud Speech-to-Text configuration manager.
    
    This class handles authentication and configuration for Azure's cloud-based
    STT service. It supports two authentication modes:
    
    Authentication Modes:
        1. Region Mode: Use subscription key + region name
           - SPEECH_KEY/AZURE_SPEECH_KEY + SPEECH_REGION/AZURE_SPEECH_REGION
           - Example: key + "eastus2"
        
        2. Endpoint Mode: Use subscription key + full endpoint URL
           - SPEECH_KEY/AZURE_SPEECH_KEY + SPEECH_ENDPOINT/AZURE_SPEECH_ENDPOINT
           - Example: key + "https://eastus2.stt.speech.microsoft.com/speech/..."
    
    Note:
        If an endpoint is provided without "/speech/" path and a region is also
        available, the class will prefer region mode for better stability.
    
    Args:
        log: Logger instance for status messages
    
    Raises:
        ValueError: If required credentials are missing
    
    Example:
        cloud = AzureCloudSTT(log=logger)
        config = cloud.speech_config("en-US")
    """
    
    def __init__(self, log: logging.Logger):
        self.log = log
        
        # -----------------------------------------------------------------
        # Load credentials from environment (support multiple var names)
        # -----------------------------------------------------------------
        self.key = (
            os.getenv("SPEECH_KEY") or os.getenv("AZURE_SPEECH_KEY") or ""
        ).strip()
        self.endpoint = (
            os.getenv("SPEECH_ENDPOINT") or os.getenv("AZURE_SPEECH_ENDPOINT") or ""
        ).strip()
        self.region = (
            os.getenv("SPEECH_REGION") or os.getenv("AZURE_SPEECH_REGION") or ""
        ).strip()

        # Validate: Key is always required
        if not self.key:
            raise ValueError(
                "Missing SPEECH_KEY/AZURE_SPEECH_KEY in env for Azure STT."
            )

        # -----------------------------------------------------------------
        # Determine authentication mode
        # Prefer region mode if endpoint doesn't have /speech/ path
        # -----------------------------------------------------------------
        if self.endpoint and "/speech/" not in self.endpoint and self.region:
            self.log.warning(
                "Endpoint provided without /speech/ path; using REGION mode for stability."
            )
            self.endpoint = ""

        if self.endpoint:
            self.mode = "endpoint"
        elif self.region:
            self.mode = "region"
        else:
            raise ValueError(
                "Provide either SPEECH_ENDPOINT/AZURE_SPEECH_ENDPOINT or SPEECH_REGION/AZURE_SPEECH_REGION."
            )

        self.log.info(f"Azure STT mode: {self.mode.upper()} from environment.")

    def speech_config(self, language: str) -> speechsdk.SpeechConfig:
        """
        Create a SpeechConfig for the specified language.
        
        Configures:
            - Authentication (endpoint or region mode)
            - Recognition language
            - Output format (detailed for confidence scores)
            - Silence timeouts for better segment detection
        
        Args:
            language: Language code (e.g., "en-US", "ar-SA")
        
        Returns:
            Configured SpeechConfig ready for recognizer creation
        """
        lang = canon_lang(language)
        
        # Create config based on authentication mode
        if self.mode == "endpoint":
            cfg = speechsdk.SpeechConfig(endpoint=self.endpoint, subscription=self.key)
        else:
            cfg = speechsdk.SpeechConfig(subscription=self.key, region=self.region)
        
        # Set recognition language
        cfg.speech_recognition_language = lang
        
        # Enable detailed output for confidence scores
        cfg.output_format = speechsdk.OutputFormat.Detailed
        
        # -----------------------------------------------------------------
        # Configure silence detection timeouts
        # These help detect end-of-speech in mixed-language scenarios
        # -----------------------------------------------------------------
        cfg.set_property(
            speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs, "1200"
        )
        cfg.set_property(
            speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, "2000"
        )
        
        return cfg


# =============================================================================
# STREAMING TRANSCRIBER - REAL-TIME STT FOR A SINGLE LANGUAGE
# =============================================================================

class StreamingTranscriber:
    """
    Real-time streaming transcriber for a single language segment.
    
    This class handles streaming audio to Azure Cloud STT while a specific
    language is being spoken. It uses PushAudioInputStream to feed audio
    chunks and collects both partial (interim) and final recognition results.
    
    Workflow:
        1. Created when LID detects a new language
        2. start() begins recognition and audio streaming
        3. set_latest_end_hns() extends the audio window as LID continues
        4. stop() finalizes and returns the transcript
    
    Audio Processing:
        - Reads from source WAV file
        - Converts to 16kHz mono PCM (required by Azure Speech)
        - Pushes chunks at regular intervals
    
    Threading:
        - Writer loop runs in daemon thread
        - Pushes audio chunks while monitoring stop flag
        - Includes tail overlap to avoid cutting words at boundaries
    
    Args:
        audio_file: Path to source WAV file
        language: Language code for this segment (e.g., "en-US")
        start_hns: Starting position in hundred-nanoseconds
        cloud_stt: AzureCloudSTT instance for configuration
        log: Logger instance
        overlap_ms: Extra audio to include after stop (default: 200ms)
        chunk_ms: Audio chunk size for streaming (default: 40ms)
    
    Events Handled:
        - RECOGNIZING: Partial/interim results (logged as ~)
        - RECOGNIZED: Final results (logged as ✔)
        - CANCELED: Errors or cancellation (logged as warning)
        - SESSION_STOPPED: Recognition session ended
    
    Example:
        transcriber = StreamingTranscriber(
            audio_file="audio.wav",
            language="en-US",
            start_hns=0,
            cloud_stt=cloud_client,
            log=logger
        )
        transcriber.start(initial_end_hns=10000000)
        # ... LID continues detecting ...
        transcriber.set_latest_end_hns(50000000)
        # ... language switches ...
        result = transcriber.stop(final_end_hns=55000000)
    """

    def __init__(
        self,
        audio_file: str,
        language: str,
        start_hns: int,
        cloud_stt: AzureCloudSTT,
        log: logging.Logger,
        overlap_ms: int = 200,
        chunk_ms: int = 40,
    ):
        self.audio_file = audio_file
        self.language = collapse_supported(language)
        self.start_hns = start_hns
        self.cloud = cloud_stt
        self.log = log
        self.overlap_hns = overlap_ms * 10_000  # Convert ms to HNS
        self.chunk_ms = chunk_ms

        # Get source audio properties
        ch, width, rate, nframes = _wav_props(audio_file)
        self.src_channels = ch
        self.src_width = width
        self.src_rate = rate
        self.total_frames = nframes

        # Threading synchronization
        self._latest_end_hns = start_hns
        self._lock = threading.Lock()
        self._stop_flag = threading.Event()      # Signal to stop writer loop
        self._stopped_event = threading.Event()  # Writer loop has stopped

        # -----------------------------------------------------------------
        # Configure Azure Speech recognizer with push stream
        # -----------------------------------------------------------------
        cfg = self.cloud.speech_config(self.language)
        
        # Azure requires 16kHz, 16-bit, mono PCM
        self._fmt = speechsdk.audio.AudioStreamFormat(
            samples_per_second=16000, bits_per_sample=16, channels=1
        )
        self._push = speechsdk.audio.PushAudioInputStream(stream_format=self._fmt)
        self._audio_cfg = speechsdk.audio.AudioConfig(stream=self._push)
        self._rec = speechsdk.SpeechRecognizer(
            speech_config=cfg, audio_config=self._audio_cfg
        )

        # Result storage
        self.partials: List[Dict] = []  # Interim results
        self.finals: List[Dict] = []     # Final confirmed results
        self._session_stopped = threading.Event()

        # -----------------------------------------------------------------
        # Connect recognition event handlers
        # -----------------------------------------------------------------
        # RECOGNIZING: Fired for partial/interim results during speech
        self._rec.recognizing.connect(self._on_recognizing)
        
        # RECOGNIZED: Fired when a phrase is confidently recognized
        self._rec.recognized.connect(self._on_recognized)
        
        # SESSION_STOPPED: Fired when recognition session ends
        self._rec.session_stopped.connect(self._on_session_stopped)
        
        # CANCELED: Fired on errors or explicit cancellation
        self._rec.canceled.connect(self._on_session_stopped)

        # Audio writer thread (pushes chunks to Azure)
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)

    def log_prefix(self) -> str:
        """Get log prefix showing language (e.g., "[STT en-US]")."""
        return f"[STT {self.language}]"

    def start(self, initial_end_hns: int):
        """
        Start streaming transcription.
        
        Args:
            initial_end_hns: Initial end position (will be extended by LID)
        """
        with self._lock:
            self._latest_end_hns = max(
                initial_end_hns, self.start_hns + self.overlap_hns
            )
        self.log.info(f"{self.log_prefix()} START at {hns_to_sec(self.start_hns):.2f}s")
        self._rec.start_continuous_recognition()
        self._thread.start()

    def set_latest_end_hns(self, end_hns: int):
        """
        Extend the audio streaming window.
        
        Called by the main loop as LID continues detecting the same language.
        
        Args:
            end_hns: New end position (only updates if larger than current)
        """
        with self._lock:
            if end_hns > self._latest_end_hns:
                self._latest_end_hns = end_hns

    def stop(self, final_end_hns: int, timeout_sec: float = 10.0):
        """
        Stop transcription and return results.
        
        Adds overlap padding, signals the writer to stop, waits for completion,
        then returns the combined transcript.
        
        Args:
            final_end_hns: Final audio position from LID
            timeout_sec: Max time to wait for pending results
        
        Returns:
            Dict with language, timing, transcript, and detailed finals
        """
        # Add tail overlap to avoid cutting words
        final_end_hns += self.overlap_hns
        self.set_latest_end_hns(final_end_hns)
        
        # Signal writer loop to stop
        self._stop_flag.set()
        self._stopped_event.wait(timeout=timeout_sec)
        
        # Stop the recognizer
        try:
            self._rec.stop_continuous_recognition()
        except Exception:
            pass
        self._session_stopped.wait(timeout=timeout_sec)
        
        # Combine all final results
        combined_text = " ".join([f["text"] for f in self.finals if f.get("text")])
        self.log.info(f"{self.log_prefix()} STOP → '{combined_text}'")
        
        return {
            "language": self.language,
            "start_hns": self.start_hns,
            "end_hns": final_end_hns,
            "transcript": combined_text,
            "finals": self.finals,
        }

    # -------------------------------------------------------------------------
    # Recognition Event Handlers
    # -------------------------------------------------------------------------

    def _on_recognizing(self, evt):
        """
        Event handler: Partial/interim recognition result.
        
        These are unstable hypotheses that may change. Logged with ~ prefix.
        Useful for showing real-time feedback but not stored as final text.
        """
        if evt and evt.result and evt.result.text:
            t = evt.result.text
            ts = time.time()
            self.partials.append({"t": ts, "text": t})
            self.log.debug(f"{self.log_prefix()} ~ {t[:120]}")

    def _on_recognized(self, evt):
        """
        Event handler: Final recognition result.
        
        This is a stable, confirmed transcription. Logged with ✔ prefix.
        
        Result Reasons:
            - RecognizedSpeech: Successful transcription
            - NoMatch: Speech detected but not recognized
            - Canceled: Error or explicit cancellation
        """
        if not evt or not evt.result:
            return
            
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            txt = evt.result.text or ""
            ts = time.time()
            self.finals.append({"t": ts, "text": txt})
            if txt.strip():
                self.log.info(f"{self.log_prefix()} ✔ {txt}")
                
        elif evt.result.reason == speechsdk.ResultReason.NoMatch:
            self.log.debug(f"{self.log_prefix()} (NoMatch)")
            
        elif evt.result.reason == speechsdk.ResultReason.Canceled:
            try:
                det = speechsdk.CancellationDetails.from_result(evt.result)
                self.log.warning(
                    f"{self.log_prefix()} CANCELED: {det.reason} — {det.error_details}"
                )
            except Exception:
                self.log.warning(f"{self.log_prefix()} CANCELED (unknown)")

    def _on_session_stopped(self, _):
        """Event handler: Recognition session has stopped."""
        self._session_stopped.set()

    # -------------------------------------------------------------------------
    # Audio Writer Thread
    # -------------------------------------------------------------------------

    def _writer_loop(self):
        """
        Background thread that reads audio and pushes to Azure.
        
        This loop:
            1. Reads chunks from the WAV file
            2. Converts to 16kHz mono PCM if needed
            3. Pushes to the PushAudioInputStream
            4. Paces itself to match real-time audio rate
            5. Stops when signaled and end position reached
        """
        start_sec = hns_to_sec(self.start_hns)
        chunk_src_frames = max(1, int(self.src_rate * (self.chunk_ms / 1000.0)))
        start_frame = int(start_sec * self.src_rate)
        
        # Handle case where start is beyond audio end
        if start_frame >= self.total_frames:
            try:
                self._push.close()
            except Exception:
                pass
            self._stopped_event.set()
            return

        state = None  # audioop.ratecv state for resampling
        cursor_frame = start_frame

        try:
            while True:
                # Get current streaming limit
                with self._lock:
                    limit_hns = self._latest_end_hns
                limit_sec = hns_to_sec(limit_hns)
                limit_frame = min(self.total_frames, int(limit_sec * self.src_rate))

                # Wait if we've caught up to the limit
                if cursor_frame >= limit_frame:
                    if self._stop_flag.is_set():
                        break  # Stop signaled and we've reached the end
                    time.sleep(0.02)  # Brief pause before checking again
                    continue

                # Read and process audio chunk
                remaining = limit_frame - cursor_frame
                to_read = min(chunk_src_frames, remaining)
                raw = _read_frames(self.audio_file, cursor_frame, to_read)
                cursor_frame += to_read
                
                if not raw:
                    break

                # -------------------------------------------------------------
                # Convert audio to Azure's required format (16kHz mono PCM16)
                # -------------------------------------------------------------
                
                # Convert sample width to 16-bit if needed
                if self.src_width != 2:
                    raw = audioop.lin2lin(raw, self.src_width, 2)
                
                # Convert to mono if stereo
                if self.src_channels > 1:
                    raw = audioop.tomono(raw, 2, 0.5, 0.5)
                
                # Resample to 16kHz if needed
                if self.src_rate != 16000:
                    raw, state = audioop.ratecv(raw, 2, 1, self.src_rate, 16000, state)

                # Push to Azure
                self._push.write(raw)
                
                # Pace to approximate real-time
                time.sleep(self.chunk_ms / 1000.0)

        except Exception as e:  # noqa: BLE001
            self.log.exception(f"{self.log_prefix()} writer exception: {e}")
        finally:
            # Close the push stream to signal end of audio
            try:
                self._push.close()
            except Exception:
                pass
            self._stopped_event.set()


# =============================================================================
# MAIN PROCESSING - ORCHESTRATES LID + STT PIPELINE
# =============================================================================

def _merge_segments_from_events(
    events: List[Tuple[str, int, int]], max_gap_ms: int = 200
) -> List[Dict]:
    """
    Merge consecutive LID events of the same language into segments.
    
    Raw LID events come in small chunks (e.g., 1 second each). This function
    combines consecutive events of the same language into larger segments,
    allowing small gaps between them.
    
    Args:
        events: List of (language, start_hns, end_hns) tuples
        max_gap_ms: Maximum gap (in ms) between events to still merge them
    
    Returns:
        List of merged segment dictionaries with Language, StartTimeHns, etc.
    
    Example:
        Input events:
            [("en-US", 0, 1s), ("en-US", 1s, 2s), ("ar-SA", 2s, 3s)]
        Output:
            [{"Language": "en-US", ...duration 2s...}, {"Language": "ar-SA", ...duration 1s...}]
    """
    if not events:
        return []
        
    # Sort by start time
    events = sorted(events, key=lambda x: x[1])  # (lang, start, end)
    
    out: List[Dict] = []
    cur_lang, cur_start, cur_end = events[0][0], events[0][1], events[0][2]
    
    for lang, s, e in events[1:]:
        # Same language and small gap → extend current segment
        if lang == cur_lang and (s - cur_end) <= max_gap_ms * 10_000:
            cur_end = max(cur_end, e)
        else:
            # Different language or large gap → finalize current, start new
            out.append(
                {
                    "Language": cur_lang,
                    "StartTimeHns": cur_start,
                    "DurationHns": max(0, cur_end - cur_start),
                    "IsSkipped": False,
                }
            )
            cur_lang, cur_start, cur_end = lang, s, e
    
    # Don't forget the last segment
    out.append(
        {
            "Language": cur_lang,
            "StartTimeHns": cur_start,
            "DurationHns": max(0, cur_end - cur_start),
            "IsSkipped": False,
        }
    )
    return out


def process_audio_file(args, log: logging.Logger):
    """
    Main processing function: orchestrates LID + STT for an audio file.
    
    This is the heart of the pipeline. It:
        1. Sets up the Azure Cloud STT client
        2. Creates an LID stream connected to the container
        3. Handles language detection events in real-time
        4. Starts/stops STT recognizers as languages switch
        5. Collects and merges all results
        6. Writes output files
    
    Flow:
        Audio → LID Container → Language Events → STT Recognizers → Transcript
    
    State Management:
        - raw_events: All individual LID detection events
        - merged_segments: Consolidated language segments
        - segment_results: Per-segment transcription results
        - active: Currently running StreamingTranscriber (or None)
    
    Args:
        args: Parsed command-line arguments
        log: Logger instance
    
    Returns:
        Final payload dict with segments and full transcript
    """
    # Initialize Azure Cloud STT client
    cloud_client = AzureCloudSTT(log=log)

    # -------------------------------------------------------------------------
    # State tracking for LID events
    # -------------------------------------------------------------------------
    raw_events: List[Tuple[str, int, int]] = []  # All raw LID events
    merged_segments: List[Dict] = []              # Consolidated segments
    lid_done = threading.Event()                  # Signals when LID finishes

    # -------------------------------------------------------------------------
    # State tracking for STT transcription
    # -------------------------------------------------------------------------
    active: Optional[StreamingTranscriber] = None  # Current transcriber
    current_lang: Optional[str] = None             # Language being transcribed
    current_start_hns: Optional[int] = None        # Start of current segment
    latest_end_hns: int = 0                        # Latest end position from LID
    lock = threading.Lock()                        # Thread safety for state

    # Accumulate results as we complete each language segment
    segment_results: List[Dict] = []

    def _write_output_if_needed():
        """
        Helper: Write intermediate output if --flush-after-stop is enabled.
        
        This allows monitoring progress on very long files.
        """
        if not args.flush_after_stop:
            return
            
        # Build transcript from completed segments so far
        full = " ".join(
            f"[{r['language']}] {r['transcript']}".strip()
            for r in segment_results
            if r.get("transcript")
        )
        payload = {
            "audio_file": args.audio,
            "segment_count": len(merged_segments),
            "segments": merged_segments,
            "recognized_results": segment_results,
            "full_transcript": full,
        }
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception as e:
            log.warning(f"Failed to flush output: {e}")

    # -------------------------------------------------------------------------
    # LID Event Callback: Called for each language detection
    # -------------------------------------------------------------------------
    def on_lid_segment(lang: str, start_hns: int, end_hns: int):
        """
        Callback: Handle a language detection event from LID container.
        
        This is the core routing logic. It decides whether to:
            - Start a new transcriber (first language detected)
            - Extend the current transcriber (same language continues)
            - Stop current and start new (language switched)
        
        Args:
            lang: Detected language code (e.g., "en-US")
            start_hns: Segment start time in hundred-nanoseconds
            end_hns: Segment end time in hundred-nanoseconds
        """
        nonlocal active, current_lang, current_start_hns, latest_end_hns
        
        # Record raw event for audit/merging
        raw_events.append((lang, start_hns, end_hns))

        with lock:
            # -----------------------------------------------------------------
            # Case 1: First language detected → Start first transcriber
            # -----------------------------------------------------------------
            if current_lang is None:
                current_lang = lang
                current_start_hns = start_hns
                latest_end_hns = end_hns
                
                active = StreamingTranscriber(
                    audio_file=args.audio,
                    language=current_lang,
                    start_hns=current_start_hns,
                    cloud_stt=cloud_client,
                    log=log,
                    overlap_ms=args.tail_overlap_ms,
                    chunk_ms=args.chunk_ms,
                )
                active.start(initial_end_hns=latest_end_hns)
                return

            # -----------------------------------------------------------------
            # Case 2: Same language continues → Extend streaming window
            # -----------------------------------------------------------------
            if lang == current_lang:
                latest_end_hns = max(latest_end_hns, end_hns)
                if active:
                    active.set_latest_end_hns(latest_end_hns)
                return

            # -----------------------------------------------------------------
            # Case 3: Language switched → Stop current, start new transcriber
            # -----------------------------------------------------------------
            
            # Stop current transcriber and collect its result
            if active:
                try:
                    result = active.stop(final_end_hns=latest_end_hns)
                except Exception:
                    log.exception("Error stopping active transcriber")
                    result = {
                        "language": current_lang,
                        "start_hns": current_start_hns,
                        "end_hns": latest_end_hns,
                        "transcript": "",
                    }
                segment_results.append(result)
                
            # Record merged segment for the completed language
            merged_segments.append(
                {
                    "Language": current_lang,
                    "StartTimeHns": current_start_hns,
                    "DurationHns": max(0, latest_end_hns - current_start_hns),
                    "IsSkipped": False,
                }
            )
            _write_output_if_needed()

            # Start new transcriber for the new language
            current_lang = lang
            current_start_hns = start_hns
            latest_end_hns = end_hns
            
            active = StreamingTranscriber(
                audio_file=args.audio,
                language=current_lang,
                start_hns=current_start_hns,
                cloud_stt=cloud_client,
                log=log,
                overlap_ms=args.tail_overlap_ms,
                chunk_ms=args.chunk_ms,
            )
            active.start(initial_end_hns=latest_end_hns)

    # -------------------------------------------------------------------------
    # LID Done Callback: Called when LID finishes
    # -------------------------------------------------------------------------
    def on_lid_done():
        """Callback: LID processing has completed."""
        lid_done.set()

    # -------------------------------------------------------------------------
    # Create and start LID stream
    # -------------------------------------------------------------------------
    lid = LIDStream(
        audio_file=args.audio,
        languages=args.languages,
        lid_host=args.lid_host,
        min_segment_sec=args.min_segment_sec,
        log=log,
        on_segment=on_lid_segment,
        on_done=on_lid_done,
    )
    lid.start()

    # -------------------------------------------------------------------------
    # Wait for LID to complete (or timeout)
    # -------------------------------------------------------------------------
    t0 = time.time()
    while not lid_done.is_set():
        if args.timeout_sec and (time.time() - t0) > args.timeout_sec:
            log.warning(f"LID timeout after {args.timeout_sec}s — stopping.")
            try:
                lid.recognizer.stop_continuous_recognition()
            except Exception:
                pass
            break
        time.sleep(0.1)

    # -------------------------------------------------------------------------
    # Finalize: Stop last active transcriber
    # -------------------------------------------------------------------------
    if active and current_lang is not None and current_start_hns is not None:
        try:
            last_result = active.stop(final_end_hns=latest_end_hns)
        except Exception:
            log.exception("Error stopping final transcriber")
            last_result = {
                "language": current_lang,
                "start_hns": current_start_hns,
                "end_hns": latest_end_hns,
                "transcript": "",
            }
        segment_results.append(last_result)
        merged_segments.append(
            {
                "Language": current_lang,
                "StartTimeHns": current_start_hns,
                "DurationHns": max(0, latest_end_hns - current_start_hns),
                "IsSkipped": False,
            }
        )

    # -------------------------------------------------------------------------
    # Write LID segments file (audit/reference)
    # -------------------------------------------------------------------------
    segments_json = _merge_segments_from_events(raw_events)
    try:
        with open(args.segments, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "AudioFile": args.audio,
                    "SegmentCount": len(segments_json),
                    "Segments": segments_json,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        log.info(f"Wrote {len(segments_json)} language segments to {args.segments}")
    except Exception as e:
        log.warning(f"Failed writing segments file: {e}")

    # -------------------------------------------------------------------------
    # Build and write final transcript
    # -------------------------------------------------------------------------
    full = " ".join(
        f"[{r['language']}] {r['transcript']}".strip()
        for r in segment_results
        if r.get("transcript")
    )

    final_payload = {
        "audio_file": args.audio,
        "segment_count": len(merged_segments),
        "segments": merged_segments,
        "recognized_results": segment_results,  # Ordered by time
        "full_transcript": full,
    }

    try:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(final_payload, f, indent=2, ensure_ascii=False)
        log.info(f"Wrote complete transcript to {args.output}")
        log.info(f"Full transcript: {final_payload['full_transcript']}")
    except Exception as e:
        log.error(f"Failed writing output: {e}")

    return final_payload


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

def main():
    """
    Entry point: Parse arguments and run the LID + STT pipeline.
    
    Usage:
        python disconnected_language_detector.py \\
            --audio "audio/sample.wav" \\
            --languages en-US ar-SA \\
            --lid-host localhost:5003 \\
            --verbose
    """
    parser = argparse.ArgumentParser(
        description="LID (container) + live per-language STT (cloud)"
    )
    
    # -------------------------------------------------------------------------
    # Required arguments
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--audio", 
        required=True, 
        help="Path to WAV audio file"
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        required=True,
        help="Languages for LID (e.g., en-US ar-SA)",
    )
    
    # -------------------------------------------------------------------------
    # Connection settings
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--lid-host",
        default="http://localhost:5000",
        help="LID container host (http://host:port, NO path)",
    )
    
    # -------------------------------------------------------------------------
    # Output files
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--segments",
        default="fixed_module_segments.json",
        help="Where to save merged LID segments JSON (for audit)",
    )
    parser.add_argument(
        "--output",
        default="fixed_module_transcript.json",
        help="Where to save transcript JSON",
    )
    
    # -------------------------------------------------------------------------
    # Timing and behavior
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--timeout-sec", 
        type=float, 
        default=60.0, 
        help="LID timeout seconds"
    )
    parser.add_argument(
        "--min-segment-sec",
        type=float,
        default=0.0,
        help="LID will emit everything; values below this are only flagged in logs",
    )
    parser.add_argument(
        "--force-detection",
        action="store_true",
        help="(kept for compatibility; unused)",
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Verbose logging"
    )
    
    # -------------------------------------------------------------------------
    # Audio streaming tuning
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--chunk-ms", 
        type=int, 
        default=40, 
        help="Audio chunk size pushed to STT (ms)"
    )
    parser.add_argument(
        "--tail-overlap-ms",
        type=int,
        default=200,
        help="Tail overlap added when stopping a language (reduces word clipping)",
    )
    parser.add_argument(
        "--flush-after-stop",
        action="store_true",
        help="Write --output after each language stop (incremental saves)",
    )

    args = parser.parse_args()
    
    # Set up logging
    log = setup_logging(args.verbose)
    
    # Load environment variables from .env (if present)
    _load_dotenv(log=log)

    # Run the main processing pipeline
    process_audio_file(args, log)


if __name__ == "__main__":
    main()
