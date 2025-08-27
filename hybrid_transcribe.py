"""
fixed_transcribe_with_language.py

Clean, single-file pipeline:

  • Language Identification (LID): Azure Speech CONTAINER via WebSocket **host only** (NO resource path)
  • Transcription: Azure Speech **CLOUD** via SDK using **ENV VARS ONLY**
      - SPEECH_KEY              or  AZURE_SPEECH_KEY           (required)
      - SPEECH_ENDPOINT         or  AZURE_SPEECH_ENDPOINT      (preferred if present; full HTTPS/WSS Speech endpoint)
      - SPEECH_REGION           or  AZURE_SPEECH_REGION        (used only if endpoint not provided)

Absolutely NO 'host=' or 'ws://...' for cloud STT. Let the SDK open transport.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import wave
from typing import Dict, List, Optional

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
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                # Preserve existing real env vars (don't overwrite if already exported)
                if key in os.environ:
                    continue
                # Strip optional surrounding quotes
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
    return logging.getLogger("lid_cloud_stt")


# -----------------------------
# Helpers
# -----------------------------
HNS_PER_SEC = 10_000_000


def hns_to_sec(hns: int) -> float:
    return hns / HNS_PER_SEC


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
# WAV extraction → PCM16 mono 16k
# -----------------------------
def read_segment_as_pcm16_mono_16k(
    wav_path: str, start_hns: int, duration_hns: int
) -> bytes:
    """Extract a segment and convert to 16 kHz, 16-bit PCM mono."""
    with wave.open(wav_path, "rb") as wf:
        ch = wf.getnchannels()
        width = wf.getsampwidth()
        rate = wf.getframerate()
        total_frames = wf.getnframes()

        start_sec = max(0.0, hns_to_sec(start_hns))
        dur_sec = max(0.0, hns_to_sec(duration_hns))

        start_frame = int(start_sec * rate)
        frames_needed = int(dur_sec * rate)

        if start_frame >= total_frames or frames_needed <= 0:
            return b""

        wf.setpos(start_frame)
        frames_to_read = min(frames_needed, total_frames - start_frame)
        raw = wf.readframes(frames_to_read)

    # → 16-bit
    if width != 2:
        raw = audioop.lin2lin(raw, width, 2)
        width = 2

    # → mono
    if ch > 1:
        raw = audioop.tomono(raw, 2, 0.5, 0.5)
        ch = 1

    # → 16 kHz
    if rate != 16000:
        raw, _ = audioop.ratecv(raw, 2, 1, rate, 16000, None)

    return raw  # PCM16 mono 16k


# -----------------------------
# LID via container (SDK host-only)
# -----------------------------
def detect_language_segments(
    audio_file: str,
    languages: List[str],
    lid_host: str,
    output_file: str,
    timeout_sec: float,
    min_segment_sec: float,
    log: logging.Logger,
) -> Dict:
    """
    Continuous LID using your container (host must be ws:// or wss://, NO path).
    Writes merged segments to output_file and returns the dict.
    """
    if not lid_host.startswith(("ws://", "wss://")):
        lid_host = "ws://" + lid_host

    log.info(f"Connecting to LID container at {lid_host} (host-only).")
    # CONTAINER: correct initializer is host= (NO resource path)
    speech_config = speechsdk.SpeechConfig(host=lid_host)
    speech_config.set_property(
        speechsdk.PropertyId.SpeechServiceConnection_LanguageIdMode, "Continuous"
    )

    lid_langs = [canon_lang(code) for code in languages]
    log.debug(f"LID languages: {', '.join(lid_langs)}")

    audio_config = speechsdk.audio.AudioConfig(filename=audio_file)
    auto_detect = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
        languages=lid_langs
    )
    recognizer = speechsdk.SourceLanguageRecognizer(
        speech_config=speech_config,
        auto_detect_source_language_config=auto_detect,
        audio_config=audio_config,
    )

    segments: List[Dict] = []
    done = False

    def on_recognized(evt):
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
            start_s = hns_to_sec(start_hns)
            end_s = hns_to_sec(start_hns + dur_hns)
            if min_segment_sec and (end_s - start_s) < min_segment_sec:
                log.debug(f"Skipping short LID seg {detected} {end_s - start_s:.2f}s")
                return
            lang_norm = canon_lang(detected)
            log.debug(f"Detected {lang_norm} from {start_s:.2f}s to {end_s:.2f}s")
            segments.append(
                {
                    "Language": lang_norm,
                    "StartTimeHns": start_hns,
                    "DurationHns": dur_hns,
                    "IsSkipped": False,
                }
            )
        except Exception as e:
            log.debug(f"LID JSON parse error: {e}")

    def on_stop(_):
        nonlocal done
        log.debug("LID session stopped")
        done = True

    recognizer.recognized.connect(on_recognized)
    recognizer.session_stopped.connect(on_stop)
    recognizer.canceled.connect(on_stop)

    log.info("Starting continuous LID…")
    recognizer.start_continuous_recognition()

    t0 = time.time()
    while not done:
        if timeout_sec and (time.time() - t0) > timeout_sec:
            log.warning(f"LID timeout after {timeout_sec}s — stopping.")
            recognizer.stop_continuous_recognition()
            break
        time.sleep(0.25)

    merged = _merge_segments(segments)
    out = {"AudioFile": audio_file, "SegmentCount": len(merged), "Segments": merged}
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info(f"Wrote {len(merged)} language segments to {output_file}")
    return out


def _merge_segments(segments: List[Dict], max_gap_ms: int = 200) -> List[Dict]:
    """Merge adjacent same-language segments if gap <= max_gap_ms."""
    if not segments:
        return []
    segs = sorted(segments, key=lambda s: s["StartTimeHns"])
    out: List[Dict] = []
    cur = dict(segs[0])

    def end_hns(seg: Dict) -> int:
        return seg["StartTimeHns"] + seg["DurationHns"]

    for s in segs[1:]:
        if (
            s["Language"] == cur["Language"]
            and (s["StartTimeHns"] - end_hns(cur)) <= max_gap_ms * 10_000
        ):
            cur["DurationHns"] = end_hns(s) - cur["StartTimeHns"]
        else:
            out.append(cur)
            cur = dict(s)
    out.append(cur)
    return out


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
        # If endpoint present but missing required path segment, treat as region fallback or auto-append
        if self.endpoint and "/speech/" not in self.endpoint:
            if self.region:
                self.log.warning(
                    "Endpoint provided without /speech/ path; falling back to REGION mode. (Given: %s)",
                    self.endpoint,
                )
                self.endpoint = ""
            else:
                # Attempt to append default conversation path
                appended = (
                    self.endpoint.rstrip("/")
                    + "/speech/recognition/conversation/cognitiveservices/v1"
                )
                self.log.warning(
                    "Endpoint missing path; auto-appending conversation path → %s",
                    appended,
                )
                self.endpoint = appended

        if self.endpoint:
            self.mode = "endpoint"
            self.log.info("Azure STT mode: ENDPOINT from environment.")
        elif self.region:
            self.mode = "region"
            self.log.info("Azure STT mode: REGION from environment.")
        else:
            raise ValueError(
                "Provide either SPEECH_ENDPOINT/AZURE_SPEECH_ENDPOINT or SPEECH_REGION/AZURE_SPEECH_REGION."
            )

    def _new_speech_config_for_language(self, language: str) -> speechsdk.SpeechConfig:
        lang = canon_lang(language)
        if self.mode == "endpoint":
            # IMPORTANT: pass endpoint verbatim; do NOT convert to ws://
            cfg = speechsdk.SpeechConfig(endpoint=self.endpoint, subscription=self.key)
        else:
            cfg = speechsdk.SpeechConfig(subscription=self.key, region=self.region)
        cfg.speech_recognition_language = lang
        cfg.output_format = speechsdk.OutputFormat.Detailed
        # Gentle timeouts help with short segments
        cfg.set_property(
            speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs, "1200"
        )
        cfg.set_property(
            speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, "2000"
        )
        return cfg

    def transcribe_pcm_segment(self, pcm16_mono_16k: bytes, language: str) -> Dict:
        """Create a recognizer for `language`, push PCM, and recognize_once()."""
        speech_config = self._new_speech_config_for_language(language)

        fmt = speechsdk.audio.AudioStreamFormat(
            samples_per_second=16000, bits_per_sample=16, channels=1
        )
        push = speechsdk.audio.PushAudioInputStream(stream_format=fmt)
        audio_cfg = speechsdk.audio.AudioConfig(stream=push)
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config, audio_config=audio_cfg
        )

        push.write(pcm16_mono_16k)
        push.close()

        res = recognizer.recognize_once()

        if res.reason == speechsdk.ResultReason.RecognizedSpeech:
            return {"status": "success", "text": res.text or ""}
        if res.reason == speechsdk.ResultReason.NoMatch:
            return {"status": "nomatch", "note": "No speech recognized"}
        if res.reason == speechsdk.ResultReason.Canceled:
            try:
                details = speechsdk.CancellationDetails.from_result(res)
                props = {}
                try:
                    # Collect some diagnostic properties
                    for pid in [
                        speechsdk.PropertyId.SpeechServiceConnection_JsonResult,
                        speechsdk.PropertyId.SpeechServiceResponse_JsonResult,
                    ]:
                        val = res.properties.get(pid)
                        if val:
                            props[str(pid)] = val[:500]
                except Exception:  # noqa: BLE001
                    pass
                return {
                    "status": "error",
                    "error": f"{details.reason}: {details.error_details or 'Unknown'}",
                    "cancellation_reason": str(details.reason),
                    "error_details": details.error_details,
                    "raw_properties": props,
                }
            except Exception:
                return {
                    "status": "error",
                    "error": "Canceled (unknown): Unknown details",
                }
        return {"status": "error", "error": f"Unexpected result: {res.reason}"}


def transcribe_segments_with_cloud(
    audio_file: str,
    segments: List[Dict],
    stt: AzureCloudSTT,
    log: logging.Logger,
) -> List[Dict]:
    """Transcribe each LID segment against Azure STT (cloud)."""
    results: List[Dict] = []
    for i, seg in enumerate(segments, start=1):
        if seg.get("IsSkipped"):
            continue

        lang = collapse_supported(seg["Language"])
        start_hns = int(seg["StartTimeHns"])
        dur_hns = int(seg["DurationHns"])
        start_s, end_s = hns_to_sec(start_hns), hns_to_sec(start_hns + dur_hns)
        log.info(
            f"[{i}/{len(segments)}] {lang} {start_s:.2f}s → {end_s:.2f}s (Azure STT)"
        )

        try:
            pcm = read_segment_as_pcm16_mono_16k(audio_file, start_hns, dur_hns)
            if not pcm:
                results.append(
                    {
                        "language": lang,
                        "start_time": start_hns,
                        "duration": dur_hns,
                        "start_sec": start_s,
                        "duration_sec": end_s - start_s,
                        "transcript": "",
                        "note": "Empty segment audio",
                    }
                )
                continue

            r = stt.transcribe_pcm_segment(pcm, lang)
            rec = {
                "language": lang,
                "start_time": start_hns,
                "duration": dur_hns,
                "start_sec": start_s,
                "duration_sec": end_s - start_s,
                "transcript": "",
            }
            if r["status"] == "success":
                rec["transcript"] = r["text"]
            elif r["status"] == "nomatch":
                rec["note"] = r.get("note", "No speech recognized")
            else:
                rec["error"] = r.get("error", "Unknown error")
                log.warning(f"  -> Error: {rec['error']}")
            results.append(rec)
        except Exception as e:
            log.exception("Segment transcription exception")
            results.append(
                {
                    "language": lang,
                    "start_time": start_hns,
                    "duration": dur_hns,
                    "start_sec": start_s,
                    "duration_sec": end_s - start_s,
                    "transcript": "",
                    "error": f"Exception: {e}",
                }
            )
    return results


# -----------------------------
# Pipeline
# -----------------------------
def process_audio_file(args, log: logging.Logger):
    # 1) Detect segments (container LID) or reuse file
    if not os.path.exists(args.segments) or args.force_detection:
        lid_result = detect_language_segments(
            audio_file=args.audio,
            languages=args.languages,
            lid_host=args.lid_host,
            output_file=args.segments,
            timeout_sec=args.timeout_sec,
            min_segment_sec=args.min_segment_sec,
            log=log,
        )
    else:
        with open(args.segments, "r", encoding="utf-8") as f:
            lid_result = json.load(f)
        log.info(
            f"Loaded {len(lid_result.get('Segments', []))} segments from {args.segments}"
        )

    segments = lid_result.get("Segments", [])
    log.info(f"Segments to transcribe after merge: {len(segments)}")

    # 2) Azure STT client (CLOUD) from ENV
    stt = AzureCloudSTT(log=log)

    # 3) Transcribe all segments
    results = transcribe_segments_with_cloud(args.audio, segments, stt, log)

    # 4) Final transcript
    results.sort(key=lambda r: r["start_time"])
    full = " ".join(
        f"[{r['language']}] {r['transcript']}".strip()
        for r in results
        if r.get("transcript")
    )

    final = {
        "audio_file": args.audio,
        "segment_count": len(results),
        "transcribed_segments": results,
        "full_transcript": full,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    log.info(f"Wrote complete transcript to {args.output}")
    log.info(f"Full transcript: {final['full_transcript']}")
    return final


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="LID (container) + Azure STT (cloud via ENV) multilingual transcription"
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
        help="Where to save/load LID segments JSON",
    )
    parser.add_argument(
        "--output",
        default="fixed_module_transcript.json",
        help="Where to save final transcript JSON",
    )
    parser.add_argument(
        "--timeout-sec", type=float, default=45.0, help="LID timeout seconds"
    )
    parser.add_argument(
        "--min-segment-sec",
        type=float,
        default=0.0,
        help="Drop LID segments shorter than this many seconds",
    )
    parser.add_argument(
        "--force-detection",
        action="store_true",
        help="Force LID even if segments JSON exists",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()
    log = setup_logging(args.verbose)

    # Load .env (if not already exported manually)
    _load_dotenv(log=log)
    # Env needed: SPEECH_KEY or AZURE_SPEECH_KEY plus SPEECH_REGION/AZURE_SPEECH_REGION
    # or SPEECH_ENDPOINT/AZURE_SPEECH_ENDPOINT (full endpoint path) for cloud STT.
    process_audio_file(args, log)


if __name__ == "__main__":
    main()
