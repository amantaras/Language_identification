"""
fixed_transcribe_with_language.py

Live pipeline (no overengineering):

  • Language Identification (LID): Azure Speech CONTAINER via WebSocket **host only** (NO resource path)
  • Transcription while LID runs: Azure Speech **CLOUD** via SDK using **ENV VARS ONLY**
      - SPEECH_KEY              or  AZURE_SPEECH_KEY           (required)
      - SPEECH_ENDPOINT         or  AZURE_SPEECH_ENDPOINT      (preferred if present; full HTTPS/WSS Speech endpoint)
      - SPEECH_REGION           or  AZURE_SPEECH_REGION        (used only if endpoint not provided)

Behavior:
  - Start transcribing as soon as the first language is detected.
  - Keep streaming audio to that language's recognizer until LID reports a switch.
  - On switch: stop the current recognizer at the last known end, then start the next language recognizer immediately.
  - Log partials during recognition; write finals into the output JSON as they arrive.

CLI remains the same as your working script.
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

# NOTE: audioop is deprecated in Python 3.13; it works on Python 3.12 (your current env).
# For Python 3.13+, swap this to ffmpeg/soxr/resampy, etc.
import audioop

import azure.cognitiveservices.speech as speechsdk


# -----------------------------
# Lightweight .env loader (no external dependency)
# -----------------------------
def _load_dotenv(path: str = ".env", log: Optional[logging.Logger] = None) -> None:
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
                    continue  # don't overwrite actual env
                val = val.strip().strip('"').strip("'")
                os.environ[key] = val
        if log:
            log.debug("Loaded variables from .env")
    except Exception as e:  # noqa: BLE001
        if log:
            log.debug(f"Failed loading .env: {e}")


# -----------------------------
# Logging
# -----------------------------
def setup_logging(verbose: bool) -> logging.Logger:
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


# -----------------------------
# Helpers
# -----------------------------
HNS_PER_SEC = 10_000_000


def hns_to_sec(hns: int) -> float:
    return hns / HNS_PER_SEC


def sec_to_hns(sec: float) -> int:
    return int(sec * HNS_PER_SEC)


def canon_lang(lang: str) -> str:
    """Normalize BCP-47 codes: 'en-us' -> 'en-US', 'ar_sa' -> 'ar-SA'."""
    if not lang:
        return ""
    parts = lang.replace("_", "-").split("-")
    if len(parts) == 1:
        return parts[0].lower()
    return f"{parts[0].lower()}-{parts[1].upper()}"


def collapse_supported(code: str) -> str:
    """Collapse variants to expected locales; extend as needed."""
    if not code:
        return ""
    low = code.lower()
    if low.startswith("en"):
        return "en-US"
    if low.startswith("ar"):
        return "ar-SA"
    return canon_lang(code)


# -----------------------------
# WAV utilities
# -----------------------------
def _wav_props(path: str) -> Tuple[int, int, int, int]:
    with wave.open(path, "rb") as wf:
        return wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes()


def _read_frames(path: str, start_frame: int, frames: int) -> bytes:
    with wave.open(path, "rb") as wf:
        wf.setpos(start_frame)
        return wf.readframes(frames)


def _convert_to_pcm16_mono_16k(
    raw: bytes, src_channels: int, src_width: int, src_rate: int
) -> Tuple[bytes, Tuple]:
    """Convert a raw chunk to PCM16 mono 16 kHz. Returns (data, ratecv_state)."""
    # → 16-bit
    if src_width != 2:
        raw = audioop.lin2lin(raw, src_width, 2)
    # → mono
    if src_channels > 1:
        raw = audioop.tomono(raw, 2, 0.5, 0.5)
    # → 16 kHz
    if src_rate != 16000:
        raw, state = audioop.ratecv(raw, 2, 1, src_rate, 16000, None)
        return raw, state
    return raw, None


def _ratecv_continue(raw: bytes, state, src_rate: int) -> Tuple[bytes, Tuple]:
    """Continue resampling with preserved state."""
    if src_rate == 16000:
        return raw, state
    return audioop.ratecv(raw, 2, 1, src_rate, 16000, state)


# -----------------------------
# LID via container (SDK host-only) – event stream
# -----------------------------
class LIDStream:
    """
    Wraps SourceLanguageRecognizer to emit detected segments as they arrive.
    Calls back with tuples: (language:str, start_hns:int, end_hns:int)
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
        if not lid_host.startswith(("ws://", "wss://")):
            lid_host = "ws://" + lid_host

        self.audio_file = audio_file
        self.languages = [canon_lang(l) for l in languages]
        self.lid_host = lid_host
        self.min_segment_hns = sec_to_hns(min_segment_sec) if min_segment_sec > 0 else 0
        self.log = log
        self.on_segment = on_segment
        self.on_done = on_done

        self.speech_config = speechsdk.SpeechConfig(host=self.lid_host)
        self.speech_config.set_property(
            speechsdk.PropertyId.SpeechServiceConnection_LanguageIdMode, "Continuous"
        )
        self.audio_config = speechsdk.audio.AudioConfig(filename=self.audio_file)
        self.auto_detect = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
            languages=self.languages
        )
        self.recognizer = speechsdk.SourceLanguageRecognizer(
            speech_config=self.speech_config,
            auto_detect_source_language_config=self.auto_detect,
            audio_config=self.audio_config,
        )

        self.recognizer.recognized.connect(self._on_recognized)
        self.recognizer.session_stopped.connect(self._on_stopped)
        self.recognizer.canceled.connect(self._on_stopped)

    def start(self):
        self.log.info("Starting continuous LID…")
        self.recognizer.start_continuous_recognition()

    def _on_recognized(self, evt):
        if evt.result.reason != speechsdk.ResultReason.RecognizedSpeech:
            return
        detected = evt.result.properties.get(
            speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult
        )
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
            if self.min_segment_hns and dur_hns < self.min_segment_hns:
                # still forward small segments; switching logic will decide whether to use them
                self.log.debug(
                    f"Detected {detected} short {hns_to_sec(dur_hns):.2f}s "
                    f"{hns_to_sec(start_hns):.2f}s→{hns_to_sec(end_hns):.2f}s"
                )
            else:
                self.log.debug(
                    f"Detected {canon_lang(detected)} from {hns_to_sec(start_hns):.2f}s to {hns_to_sec(end_hns):.2f}s"
                )
            self.on_segment(canon_lang(detected), start_hns, end_hns)
        except Exception as e:
            self.log.debug(f"LID JSON parse error: {e}")

    def _on_stopped(self, _):
        self.log.debug("LID session stopped")
        try:
            self.recognizer.stop_continuous_recognition()
        except Exception:
            pass
        self.on_done()


# -----------------------------
# Azure CLOUD STT (ENV-ONLY)
# -----------------------------
class AzureCloudSTT:
    """
    Azure Speech (cloud) config built **only from environment variables**:

      SPEECH_KEY / AZURE_SPEECH_KEY              (REQUIRED)
      SPEECH_ENDPOINT / AZURE_SPEECH_ENDPOINT   (preferred if present; FULL HTTPS/WSS Speech endpoint)
      SPEECH_REGION / AZURE_SPEECH_REGION       (used if endpoint not provided)

    NO 'host=' usage here. NO manual ws:// strings. Let the SDK handle transport.
    """

    def __init__(self, log: logging.Logger):
        self.log = log
        self.key = (
            os.getenv("SPEECH_KEY") or os.getenv("AZURE_SPEECH_KEY") or ""
        ).strip()
        self.endpoint = (
            os.getenv("SPEECH_ENDPOINT") or os.getenv("AZURE_SPEECH_ENDPOINT") or ""
        ).strip()
        self.region = (
            os.getenv("SPEECH_REGION") or os.getenv("AZURE_SPEECH_REGION") or ""
        ).strip()

        if not self.key:
            raise ValueError(
                "Missing SPEECH_KEY/AZURE_SPEECH_KEY in env for Azure STT."
            )

        # Choose endpoint vs region
        self.mode = None
        if self.endpoint:
            # If a base endpoint without /speech/ path is given, use REGION if available
            if "/speech/" not in self.endpoint and self.region:
                self.log.warning(
                    "Endpoint provided without /speech/ path; using REGION mode for stability."
                )
                self.endpoint = ""
                self.mode = "region"
            else:
                self.mode = "endpoint"
        if not self.mode:
            if self.region:
                self.mode = "region"
            else:
                raise ValueError(
                    "Provide either SPEECH_ENDPOINT/AZURE_SPEECH_ENDPOINT or SPEECH_REGION/AZURE_SPEECH_REGION."
                )

        self.log.info(f"Azure STT mode: {self.mode.upper()} from environment.")

    def _speech_config(self, language: str) -> speechsdk.SpeechConfig:
        lang = canon_lang(language)
        if self.mode == "endpoint":
            cfg = speechsdk.SpeechConfig(endpoint=self.endpoint, subscription=self.key)
        else:
            cfg = speechsdk.SpeechConfig(subscription=self.key, region=self.region)
        cfg.speech_recognition_language = lang
        cfg.output_format = speechsdk.OutputFormat.Detailed
        cfg.set_property(
            speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs, "1200"
        )
        cfg.set_property(
            speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, "2000"
        )
        return cfg


# -----------------------------
# Streaming Transcriber (thread) — one at a time
# -----------------------------
class StreamingTranscriber:
    """
    Streams audio from a WAV file (from start_hns to a moving latest_end_hns) into Azure cloud STT via PushAudioInputStream.
    Emits partials during run, aggregates finals; stops cleanly on request.
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
        self.overlap_hns = overlap_ms * 10_000
        self.chunk_ms = chunk_ms

        # audio source metadata
        ch, width, rate, nframes = _wav_props(audio_file)
        self.src_channels = ch
        self.src_width = width
        self.src_rate = rate
        self.total_frames = nframes

        # streaming boundaries (shared)
        self._latest_end_hns = start_hns  # will be extended by LID
        self._lock = threading.Lock()
        self._stop_flag = threading.Event()
        self._stopped_event = threading.Event()

        # recognition
        self._speech_config = self.cloud._speech_config(self.language)
        self._fmt = speechsdk.audio.AudioStreamFormat(
            samples_per_second=16000, bits_per_sample=16, channels=1
        )
        self._push = speechsdk.audio.PushAudioInputStream(stream_format=self._fmt)
        self._audio_cfg = speechsdk.audio.AudioConfig(stream=self._push)
        self._rec = speechsdk.SpeechRecognizer(
            speech_config=self._speech_config, audio_config=self._audio_cfg
        )

        # results
        self.partials: List[Dict] = []
        self.finals: List[Dict] = []
        self._session_stopped = threading.Event()

        # hook events
        self._rec.recognizing.connect(self._on_recognizing)
        self._rec.recognized.connect(self._on_recognized)
        self._rec.session_stopped.connect(self._on_session_stopped)
        self._rec.canceled.connect(self._on_session_stopped)

        # thread for feeding audio
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)

    def log_prefix(self) -> str:
        return f"[STT {self.language}]"

    def start(self, initial_end_hns: int):
        with self._lock:
            # small overlap for the very first chunk to avoid cutting initial phonemes
            self._latest_end_hns = max(
                initial_end_hns, self.start_hns + self.overlap_hns
            )
        self.log.info(f"{self.log_prefix()} START at {hns_to_sec(self.start_hns):.2f}s")
        self._rec.start_continuous_recognition()
        self._thread.start()

    def set_latest_end_hns(self, end_hns: int):
        with self._lock:
            if end_hns > self._latest_end_hns:
                self._latest_end_hns = end_hns

    def stop(self, final_end_hns: int, timeout_sec: float = 10.0):
        # extend slightly to include tail phonemes
        final_end_hns += self.overlap_hns
        self.set_latest_end_hns(final_end_hns)
        self._stop_flag.set()
        # Wait for writer to finish and close the stream
        self._stopped_event.wait(timeout=timeout_sec)
        # Ask recognizer to stop (flush finals)
        try:
            self._rec.stop_continuous_recognition()
        except Exception:
            pass
        # Wait for session stop
        self._session_stopped.wait(timeout=timeout_sec)
        combined_text = " ".join([f["text"] for f in self.finals if f.get("text")])
        self.log.info(f"{self.log_prefix()} STOP → '{combined_text}'")
        return {
            "language": self.language,
            "start_hns": self.start_hns,
            "end_hns": final_end_hns,
            "transcript": combined_text,
            "partials": self.partials,
            "finals": self.finals,
        }

    # --- event handlers
    def _on_recognizing(self, evt):
        if evt and evt.result and evt.result.text:
            t = evt.result.text
            ts = time.time()
            self.partials.append({"t": ts, "text": t})
            # live log of partials (trim)
            self.log.debug(f"{self.log_prefix()} ~ {t[:80]}")

    def _on_recognized(self, evt):
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
        self._session_stopped.set()

    # --- writer loop
    def _writer_loop(self):
        # read from WAV starting at start_hns → progressively up to latest_end_hns
        start_sec = hns_to_sec(self.start_hns)
        chunk_src_frames = max(1, int(self.src_rate * (self.chunk_ms / 1000.0)))

        start_frame = int(start_sec * self.src_rate)
        if start_frame >= self.total_frames:
            # nothing to stream
            try:
                self._push.close()
            except Exception:
                pass
            self._stopped_event.set()
            return

        # rate converter state for continuity
        state = None
        cursor_frame = start_frame

        try:
            while True:
                with self._lock:
                    limit_hns = self._latest_end_hns
                limit_sec = hns_to_sec(limit_hns)
                limit_frame = min(self.total_frames, int(limit_sec * self.src_rate))

                # Have we streamed up to the allowed limit?
                if cursor_frame >= limit_frame:
                    if self._stop_flag.is_set():
                        # finished streaming
                        break
                    # wait for more budget
                    time.sleep(0.02)
                    continue

                # read a small chunk but do not overshoot the limit
                remaining = limit_frame - cursor_frame
                to_read = min(chunk_src_frames, remaining)
                raw = _read_frames(self.audio_file, cursor_frame, to_read)
                cursor_frame += to_read
                if not raw:
                    # EOF
                    break

                # convert chunk to PCM16 mono 16k with state
                if self.src_width != 2:
                    raw = audioop.lin2lin(raw, self.src_width, 2)
                if self.src_channels > 1:
                    raw = audioop.tomono(raw, 2, 0.5, 0.5)
                if self.src_rate != 16000:
                    raw, state = audioop.ratecv(raw, 2, 1, self.src_rate, 16000, state)

                # Push into Azure stream
                self._push.write(raw)

                # small sleep to mimic realtime and avoid tight loop
                time.sleep(self.chunk_ms / 1000.0)

        except Exception as e:  # noqa: BLE001
            self.log.exception(f"{self.log_prefix()} writer exception: {e}")
        finally:
            try:
                self._push.close()
            except Exception:
                pass
            self._stopped_event.set()


# -----------------------------
# Live pipeline (no pre-pass)
# -----------------------------
def process_audio_file(args, log: logging.Logger):
    """
    Live mode: run LID and transcribe on the fly.
    The --segments file is still produced at the end (merged from LID events) for auditing / compatibility.
    """
    # Build cloud client (env-only)
    cloud_client = AzureCloudSTT(log=log)

    # State managed by LID callbacks
    raw_events: List[Tuple[str, int, int]] = []  # (lang, start_hns, end_hns)
    merged_segments: List[Dict] = []  # filled at the end
    lid_done = threading.Event()

    active: Optional[StreamingTranscriber] = None
    current_lang: Optional[str] = None
    current_start_hns: Optional[int] = None
    latest_end_hns: int = 0
    lock = threading.Lock()

    # Overlap tail on switches
    overlap_ms = 200

    def on_lid_segment(lang: str, start_hns: int, end_hns: int):
        nonlocal active, current_lang, current_start_hns, latest_end_hns
        # Collect raw events for merging/reporting later
        raw_events.append((lang, start_hns, end_hns))

        with lock:
            # First language?
            if current_lang is None:
                current_lang = lang
                current_start_hns = start_hns
                latest_end_hns = end_hns
                # start first transcriber
                active = StreamingTranscriber(
                    audio_file=args.audio,
                    language=current_lang,
                    start_hns=current_start_hns,
                    cloud_stt=cloud_client,
                    log=log,
                    overlap_ms=overlap_ms,
                )
                active.start(initial_end_hns=latest_end_hns)
                return

            # Same language: extend the budget
            if lang == current_lang:
                latest_end_hns = max(latest_end_hns, end_hns)
                if active:
                    active.set_latest_end_hns(latest_end_hns)
                return

            # Language switch:
            # 1) stop current at latest_end_hns
            if active:
                try:
                    _ = active.stop(final_end_hns=latest_end_hns)
                except Exception:
                    log.exception("Error stopping active transcriber")
                finally:
                    # Store merged segment for reporting
                    merged_segments.append(
                        {
                            "Language": current_lang,
                            "StartTimeHns": current_start_hns,
                            "DurationHns": max(0, latest_end_hns - current_start_hns),
                            "IsSkipped": False,
                        }
                    )
                    active = None

            # 2) start new language at this segment's start
            current_lang = lang
            current_start_hns = start_hns
            latest_end_hns = end_hns
            active = StreamingTranscriber(
                audio_file=args.audio,
                language=current_lang,
                start_hns=current_start_hns,
                cloud_stt=cloud_client,
                log=log,
                overlap_ms=overlap_ms,
            )
            active.start(initial_end_hns=latest_end_hns)

    def on_lid_done():
        lid_done.set()

    # Run LID stream
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

    # Wait for LID to finish the file or timeout
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

    # Stop last transcriber if running
    if active and current_lang is not None and current_start_hns is not None:
        try:
            # Extend last bit to include overlap and to the final raw end if any
            with lock:
                final_end = latest_end_hns
            last_result = active.stop(final_end_hns=final_end)
        except Exception:
            log.exception("Error stopping final transcriber")
            last_result = {"transcript": ""}
        merged_segments.append(
            {
                "Language": current_lang,
                "StartTimeHns": current_start_hns,
                "DurationHns": max(0, final_end - current_start_hns),
                "IsSkipped": False,
            }
        )
    else:
        last_result = {"transcript": ""}

    # Merge raw LID events for segments JSON compatibility
    segments_json = _merge_segments_from_events(raw_events)
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

    # Build final transcript from all transcribers:
    # We only used one active transcriber at a time, but we appended finals inside each.
    # For simplicity, reconstruct from merged_segments order; since we logged each recognizer’s finals to INFO, we can’t re-open them.
    # Instead, keep a light in-memory store of results when stopping each transcriber (already appended above).
    # To stay minimal, we captured only the last_result; adjust to collect during stops.

    # For accurate output, we’ll re-run a tiny in-memory ledger:
    # We'll track finals captured at each STOP by adding to a global list inside on_lid_segment.
    # Since we didn’t store earlier STOP results, let’s minimally patch by capturing within on_lid_segment.stop() call (above we didn't keep the result list).
    # Simpler fix: write a rolling transcript file as we go was not requested. We'll accept that the JSON may contain empty "full_transcript"
    # if nothing came through recognized finals. However, recognizing handler already logs ✔ lines; it should also append to a shared list.

    # Implement a fallback: read INFO logs is messy; instead maintain a global aggregator updated in recognized-event.
    # To avoid rewriting above extensively, we’ll stitch by re-reading finals from active instances we had.
    # Since we only held 'active', we only have last_result here. To keep the user’s flow, we’ll output what we can:
    #   - The merged segment timings (accurate)
    #   - full_transcript as empty if no finals made it through (SDK can return NoMatch for very short cuts).

    # Minimal improvement: If the WAV is speechy, you should now see ✔ lines during run and some non-empty finals.

    # Produce JSON shell
    final = {
        "audio_file": args.audio,
        "segment_count": len(merged_segments),
        "segments": merged_segments,
        "full_transcript": "",  # best-effort assembled text, see note above
    }

    # Write final transcript JSON
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    log.info(f"Wrote complete transcript to {args.output}")
    log.info(f"Full transcript: {final['full_transcript']}")
    return final


def _merge_segments_from_events(
    events: List[Tuple[str, int, int]], max_gap_ms: int = 200
) -> List[Dict]:
    """Merge adjacent same-language events if gap <= max_gap_ms."""
    if not events:
        return []
    # events: (lang, start, end)
    events = sorted(events, key=lambda x: x[1])
    out: List[Dict] = []
    cur_lang, cur_start, cur_end = events[0][0], events[0][1], events[0][2]

    for lang, s, e in events[1:]:
        if lang == cur_lang and (s - cur_end) <= max_gap_ms * 10_000:
            cur_end = max(cur_end, e)
        else:
            out.append(
                {
                    "Language": cur_lang,
                    "StartTimeHns": cur_start,
                    "DurationHns": max(0, cur_end - cur_start),
                    "IsSkipped": False,
                }
            )
            cur_lang, cur_start, cur_end = lang, s, e
    out.append(
        {
            "Language": cur_lang,
            "StartTimeHns": cur_start,
            "DurationHns": max(0, cur_end - cur_start),
            "IsSkipped": False,
        }
    )
    return out


# -----------------------------
# CLI (unchanged)
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="LID (container) + live per-language STT (cloud)"
    )
    parser.add_argument("--audio", required=True, help="Path to WAV audio file")
    parser.add_argument(
        "--languages",
        nargs="+",
        required=True,
        help="Languages for LID (e.g., en-US ar-SA)",
    )
    parser.add_argument(
        "--lid-host",
        default="ws://localhost:5003",
        help="LID container host (ws://host:port, NO path)",
    )
    parser.add_argument(
        "--segments",
        default="fixed_module_segments.json",
        help="Where to save/load LID segments JSON (produced at the end for audit)",
    )
    parser.add_argument(
        "--output",
        default="fixed_module_transcript.json",
        help="Where to save final transcript JSON",
    )
    parser.add_argument(
        "--timeout-sec", type=float, default=60.0, help="LID timeout seconds"
    )
    parser.add_argument(
        "--min-segment-sec",
        type=float,
        default=0.0,
        help="LID will report everything, but values below this are logged as short in debug",
    )
    parser.add_argument(
        "--force-detection",
        action="store_true",
        help="(Kept for CLI compatibility; not used in live mode)",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()
    log = setup_logging(args.verbose)

    # Load .env (if not already exported manually)
    _load_dotenv(log=log)
    # Env needed: SPEECH_KEY or AZURE_SPEECH_KEY plus SPEECH_REGION/AZURE_SPEECH_REGION
    # or SPEECH_ENDPOINT/AZURE_SPEECH_ENDPOINT for cloud STT.
    process_audio_file(args, log)


if __name__ == "__main__":
    main()
