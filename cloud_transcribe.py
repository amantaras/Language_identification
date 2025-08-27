"""
fixed_transcribe_with_language.py

Multilingual transcription that uses:
  • Azure Speech SDK + your LID CONTAINER for language detection (host-only WebSocket)
  • Azure Speech SERVICE (cloud resource: subscription + region) for actual STT transcription

Why this setup?
- You asked to *stop* using containers for the transcribers and use the normal Azure STT instead.
- We keep your LID flow as-is (container), then route each detected segment to Azure STT by setting the proper locale.

Prereqs:
  pip install azure-cognitiveservices-speech

Auth for Azure STT (cloud):
  - Either set env vars AZURE_SPEECH_KEY and AZURE_SPEECH_REGION
  - Or pass --speech-key and --speech-region on the CLI

Run example:
  python fixed_transcribe_with_language.py ^
      --audio "audio/samples/Arabic_english_mix_optimized.wav" ^
      --languages en-US ar-SA ^
      --lid-host ws://localhost:5003 ^
      --output fixed_module_transcript.json ^
      --segments fixed_module_segments.json ^
      --speech-key %AZURE_SPEECH_KEY% ^
      --speech-region %AZURE_SPEECH_REGION% ^
      --force-detection --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import wave
from typing import Dict, List, Optional, Tuple

# NOTE: audioop is deprecated in Python 3.13, but available in 3.12 (your current logs show 3.12).
# It's fine for light resampling/downmixing here. Swap to ffmpeg/soxr later if you move to 3.13+.
import audioop

import azure.cognitiveservices.speech as speechsdk


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
    return logging.getLogger("lid_stt")


# -----------------------------
# Helpers
# -----------------------------
HNS_PER_SEC = 10_000_000


def hns_to_sec(hns: int) -> float:
    return hns / HNS_PER_SEC


def canon_lang(lang: str) -> str:
    """Normalize BCP-47: 'en-us' -> 'en-US', 'ar_sa' -> 'ar-SA'."""
    if not lang:
        return ""
    parts = lang.replace("_", "-").split("-")
    if len(parts) == 1:
        return parts[0].lower()
    return f"{parts[0].lower()}-{parts[1].upper()}"


def collapse_supported(code: str) -> str:
    """Collapse variants to your supported locales. Extend as needed."""
    if not code:
        return ""
    low = code.lower()
    if low.startswith("en"):
        return "en-US"
    if low.startswith("ar"):
        return "ar-SA"
    return canon_lang(code)


# -----------------------------
# WAV segment extraction → PCM16 mono 16k
# -----------------------------
def read_segment_as_pcm16_mono_16k(
    wav_path: str, start_hns: int, duration_hns: int
) -> bytes:
    """Extract segment and convert to 16 kHz, 16-bit PCM mono."""
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

    # -> 16-bit
    if width != 2:
        raw = audioop.lin2lin(raw, width, 2)
        width = 2

    # -> mono
    if ch > 1:
        raw = audioop.tomono(raw, 2, 0.5, 0.5)
        ch = 1

    # -> 16k
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
    speech_config = speechsdk.SpeechConfig(host=lid_host)
    speech_config.set_property(
        speechsdk.PropertyId.SpeechServiceConnection_LanguageIdMode, "Continuous"
    )

    # canonicalize the languages list for LID
    lid_langs = [canon_lang(l) for l in languages]
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

    # Merge adjacent same-language segments with tiny gaps
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
# Azure STT (cloud resource) per segment
# -----------------------------
class AzureCloudSTT:
    """Simple helper for Azure STT using subscription + region."""

    def __init__(self, key: str, region: str):
        if not key or not region:
            raise ValueError("Azure STT requires both subscription key and region.")
        self.key = key
        self.region = region

    def transcribe_pcm_segment(self, pcm16_mono_16k: bytes, language: str) -> Dict:
        """
        Create a recognizer with the given language and push the PCM.
        Uses recognize_once() for bounded segments.
        """
        speech_config = speechsdk.SpeechConfig(
            subscription=self.key, region=self.region
        )
        # IMPORTANT: set the language for this segment
        speech_config.speech_recognition_language = canon_lang(language)
        # Helpful tuning around short segments
        speech_config.set_property(
            speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs, "1200"
        )
        speech_config.set_property(
            speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, "2000"
        )
        speech_config.output_format = speechsdk.OutputFormat.Detailed

        # Use PushAudioInputStream with explicit format
        fmt = speechsdk.audio.AudioStreamFormat(
            samples_per_second=16000, bits_per_sample=16, channels=1
        )
        push = speechsdk.audio.PushAudioInputStream(stream_format=fmt)
        audio_cfg = speechsdk.audio.AudioConfig(stream=push)
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config, audio_config=audio_cfg
        )

        # Stream then close to signal end
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
                return {
                    "status": "error",
                    "error": f"{details.reason}: {details.error_details or 'Unknown'}",
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
    log.info(f"Segments to transcribe: {len(segments)}")

    # 2) Azure STT client (cloud)
    speech_key = args.speech_key or os.getenv("AZURE_SPEECH_KEY")
    speech_region = args.speech_region or os.getenv("AZURE_SPEECH_REGION")
    if not speech_key or not speech_region:
        log.error(
            "Azure STT credentials missing. Provide --speech-key/--speech-region or set AZURE_SPEECH_KEY/AZURE_SPEECH_REGION."
        )
        sys.exit(2)

    stt = AzureCloudSTT(key=speech_key, region=speech_region)

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
        description="LID (container) + Azure STT (cloud) multilingual transcription"
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
        help="LID container host (ws://host:port)",
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
    parser.add_argument(
        "--speech-key", default=None, help="Azure Speech subscription key (cloud STT)"
    )
    parser.add_argument(
        "--speech-region",
        default=None,
        help="Azure Speech region (cloud STT), e.g., eastus",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()
    log = setup_logging(args.verbose)
    process_audio_file(args, log)


if __name__ == "__main__":
    main()
