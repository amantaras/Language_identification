"""
fixed_transcribe_with_language.py  — LIVE LID + LIVE CLOUD STT

What it does
------------
• LID: Azure Speech CONTAINER via WebSocket "host only" (NO path)
• STT: Azure Speech CLOUD via SDK using ENV VARS ONLY
    - SPEECH_KEY / AZURE_SPEECH_KEY            (required)
    - SPEECH_REGION / AZURE_SPEECH_REGION      (or)
    - SPEECH_ENDPOINT / AZURE_SPEECH_ENDPOINT  (full HTTPS/WSS endpoint)

Behavior
--------
• Starts STT as soon as the first language is detected.
• Streams audio to that recognizer until LID reports a switch.
• On switch: stops current recognizer, stores its final text, then starts the new one.
• Logs partials (~) and finals (✔) during run.
• Writes merged LID segments to --segments and full transcript + per-segment results to --output.
• (Optional) --flush-after-stop updates --output after every language stop.

CLI (unchanged)
---------------
python fixed_transcribe_with_language.py ^
  --audio "audio\\samples\\Arabic_english_mix_optimized.wav" ^
  --languages en-US ar-SA ^
  --lid-host ws://localhost:5003 ^
  --segments fixed_module_segments.json ^
  --output fixed_module_transcript.json ^
  --verbose --flush-after-stop
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time292929
import threading
import wave
from typing import Dict, List, Optional, Tuple

# NOTE: audioop is deprecated in Python 3.13; OK on Python 3.12.
import audioop

import azure.cognitiveservices.speech as speechsdk


# -----------------------------
# Lightweight .env loader
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
                    continue
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
    if not lang:
        return ""
    parts = lang.replace("_", "-").split("-")
    if len(parts) == 1:
        return parts[0].lower()
    return f"{parts[0].lower()}-{parts[1].upper()}"


def collapse_supported(code: str) -> str:
    if not code:
        return ""
    low = code.lower()
    if low.startswith("en"):
        return "en-US"
    if low.startswith("ar"):
        return "ar-SA"
    return canon_lang(code)


# -----------------------------
# WAV helpers
# -----------------------------
def _wav_props(path: str) -> Tuple[int, int, int, int]:
    with wave.open(path, "rb") as wf:
        return wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes()


def _read_frames(path: str, start_frame: int, frames: int) -> bytes:
    with wave.open(path, "rb") as wf:
        wf.setpos(start_frame)
        return wf.readframes(frames)


# -----------------------------
# LID stream wrapper
# -----------------------------
class LIDStream:
    """
    Emits detected segments via callback(language, start_hns, end_hns) as they arrive.
    Calls on_done() when container ends/stops.
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
# Azure CLOUD STT (ENV ONLY)
# -----------------------------
class AzureCloudSTT:
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
# Streaming Transcriber (single active)
# -----------------------------
class StreamingTranscriber:
    """
    Streams from WAV (start_hns → moving end_hns) into Azure via PushAudioInputStream.
    Emits partials (log) and accumulates finals. Stop returns the segment result.
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

        ch, width, rate, nframes = _wav_props(audio_file)
        self.src_channels = ch
        self.src_width = width
        self.src_rate = rate
        self.total_frames = nframes

        self._latest_end_hns = start_hns
        self._lock = threading.Lock()
        self._stop_flag = threading.Event()
        self._stopped_event = threading.Event()

        cfg = self.cloud.speech_config(self.language)
        self._fmt = speechsdk.audio.AudioStreamFormat(
            samples_per_second=16000, bits_per_sample=16, channels=1
        )
        self._push = speechsdk.audio.PushAudioInputStream(stream_format=self._fmt)
        self._audio_cfg = speechsdk.audio.AudioConfig(stream=self._push)
        self._rec = speechsdk.SpeechRecognizer(
            speech_config=cfg, audio_config=self._audio_cfg
        )

        self.partials: List[Dict] = []
        self.finals: List[Dict] = []
        self._session_stopped = threading.Event()

        self._rec.recognizing.connect(self._on_recognizing)
        self._rec.recognized.connect(self._on_recognized)
        self._rec.session_stopped.connect(self._on_session_stopped)
        self._rec.canceled.connect(self._on_session_stopped)

        self._thread = threading.Thread(target=self._writer_loop, daemon=True)

    def log_prefix(self) -> str:
        return f"[STT {self.language}]"

    def start(self, initial_end_hns: int):
        with self._lock:
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
        final_end_hns += self.overlap_hns
        self.set_latest_end_hns(final_end_hns)
        self._stop_flag.set()
        self._stopped_event.wait(timeout=timeout_sec)
        try:
            self._rec.stop_continuous_recognition()
        except Exception:
            pass
        self._session_stopped.wait(timeout=timeout_sec)
        combined_text = " ".join([f["text"] for f in self.finals if f.get("text")])
        self.log.info(f"{self.log_prefix()} STOP → '{combined_text}'")
        return {
            "language": self.language,
            "start_hns": self.start_hns,
            "end_hns": final_end_hns,
            "transcript": combined_text,
            "finals": self.finals,
        }

    def _on_recognizing(self, evt):
        if evt and evt.result and evt.result.text:
            t = evt.result.text
            ts = time.time()
            self.partials.append({"t": ts, "text": t})
            self.log.debug(f"{self.log_prefix()} ~ {t[:120]}")

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

    def _writer_loop(self):
        start_sec = hns_to_sec(self.start_hns)
        chunk_src_frames = max(1, int(self.src_rate * (self.chunk_ms / 1000.0)))
        start_frame = int(start_sec * self.src_rate)
        if start_frame >= self.total_frames:
            try:
                self._push.close()
            except Exception:
                pass
            self._stopped_event.set()
            return

        state = None
        cursor_frame = start_frame

        try:
            while True:
                with self._lock:
                    limit_hns = self._latest_end_hns
                limit_sec = hns_to_sec(limit_hns)
                limit_frame = min(self.total_frames, int(limit_sec * self.src_rate))

                if cursor_frame >= limit_frame:
                    if self._stop_flag.is_set():
                        break
                    time.sleep(0.02)
                    continue

                remaining = limit_frame - cursor_frame
                to_read = min(chunk_src_frames, remaining)
                raw = _read_frames(self.audio_file, cursor_frame, to_read)
                cursor_frame += to_read
                if not raw:
                    break

                # Convert to PCM16 mono 16k
                if self.src_width != 2:
                    raw = audioop.lin2lin(raw, self.src_width, 2)
                if self.src_channels > 1:
                    raw = audioop.tomono(raw, 2, 0.5, 0.5)
                if self.src_rate != 16000:
                    raw, state = audioop.ratecv(raw, 2, 1, self.src_rate, 16000, state)

                self._push.write(raw)
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
# Process (LIVE mode)
# -----------------------------
def _merge_segments_from_events(
    events: List[Tuple[str, int, int]], max_gap_ms: int = 200
) -> List[Dict]:
    if not events:
        return []
    events = sorted(events, key=lambda x: x[1])  # (lang, start, end)
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


def process_audio_file(args, log: logging.Logger):
    cloud_client = AzureCloudSTT(log=log)

    # live LID state
    raw_events: List[Tuple[str, int, int]] = []
    merged_segments: List[Dict] = []
    lid_done = threading.Event()

    # transcription state
    active: Optional[StreamingTranscriber] = None
    current_lang: Optional[str] = None
    current_start_hns: Optional[int] = None
    latest_end_hns: int = 0
    lock = threading.Lock()

    # accumulate per-language results as we stop each recognizer
    segment_results: List[Dict] = []

    def _write_output_if_needed():
        if not args.flush_after_stop:
            return
        # Build full transcript from what we have so far
        full = " ".join(
            f"[{r['language']}] {r['transcript']}".strip()
            for r in segment_results
            if r.get("transcript")
        )
        payload = {
            "audio_file": args.audio,
            "segment_count": len(merged_segments),
            "segments": merged_segments,
            "recognized_results": segment_results,  # ordered
            "full_transcript": full,
        }
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception as e:
            log.warning(f"Failed to flush output: {e}")

    def on_lid_segment(lang: str, start_hns: int, end_hns: int):
        nonlocal active, current_lang, current_start_hns, latest_end_hns
        raw_events.append((lang, start_hns, end_hns))

        with lock:
            # First language
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

            # Same language → extend budget
            if lang == current_lang:
                latest_end_hns = max(latest_end_hns, end_hns)
                if active:
                    active.set_latest_end_hns(latest_end_hns)
                return

            # Language switch:
            # 1) Stop current transcriber and store result
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
            # store merged block for this language
            merged_segments.append(
                {
                    "Language": current_lang,
                    "StartTimeHns": current_start_hns,
                    "DurationHns": max(0, latest_end_hns - current_start_hns),
                    "IsSkipped": False,
                }
            )
            _write_output_if_needed()

            # 2) Start new language
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

    def on_lid_done():
        lid_done.set()

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

    # Wait for LID to end or timeout
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

    # Stop last recognizer if active
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

    # Merge raw LID events for reference/compat
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

    # Build final transcript string
    full = " ".join(
        f"[{r['language']}] {r['transcript']}".strip()
        for r in segment_results
        if r.get("transcript")
    )

    final_payload = {
        "audio_file": args.audio,
        "segment_count": len(merged_segments),
        "segments": merged_segments,
        "recognized_results": segment_results,  # ordered by time
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


# -----------------------------
# CLI (same flags you already use)
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
        help="Where to save merged LID segments JSON (for audit)",
    )
    parser.add_argument(
        "--output",
        default="fixed_module_transcript.json",
        help="Where to save transcript JSON",
    )
    parser.add_argument(
        "--timeout-sec", type=float, default=60.0, help="LID timeout seconds"
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
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    # live-tuning knobs
    parser.add_argument(
        "--chunk-ms", type=int, default=40, help="Audio chunk size pushed to STT"
    )
    parser.add_argument(
        "--tail-overlap-ms",
        type=int,
        default=200,
        help="Tail overlap added when stopping a language",
    )
    parser.add_argument(
        "--flush-after-stop",
        action="store_true",
        help="Write --output after each language stop (incremental saves).",
    )

    args = parser.parse_args()
    log = setup_logging(args.verbose)
    _load_dotenv(log=log)  # load env vars if .env present

    process_audio_file(args, log)


if __name__ == "__main__":
    main()
