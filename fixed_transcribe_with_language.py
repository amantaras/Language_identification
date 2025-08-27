# fixed_transcribe_with_language.py
"""
Language-aware transcription with Azure Speech containers:
- LID via Speech SDK (host-only, no resource path in the host URI)
- Per-language STT via container REST endpoint (HTTP), avoiding SDK mode pitfalls
- Segment merging + small boundary padding for smoothness
"""

import os
import io
import json
import time
import uuid
import wave
import math
import argparse
import logging
import tempfile
import contextlib
from typing import List, Dict, Any, Tuple

import requests
import azure.cognitiveservices.speech as speechsdk


# --------------------------
# Logging
# --------------------------
def setup_logging(verbose: bool) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("fixed_module_test.txt", mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("lid_stt")


# --------------------------
# Utilities
# --------------------------
HNS_PER_SECOND = 10_000_000


def hns_to_sec(hns: int) -> float:
    return hns / HNS_PER_SECOND


def sec_to_hns(sec: float) -> int:
    return int(round(sec * HNS_PER_SECOND))


def safe_scheme_to_http(ws_or_wss: str) -> str:
    # Convert ws:// -> http://  and wss:// -> https:// for REST calls
    if ws_or_wss.startswith("ws://"):
        return "http://" + ws_or_wss[len("ws://") :]
    if ws_or_wss.startswith("wss://"):
        return "https://" + ws_or_wss[len("wss://") :]
    # If user already gave http(s), keep it
    if ws_or_wss.startswith("http://") or ws_or_wss.startswith("https://"):
        return ws_or_wss
    # Default to http if no scheme
    return "http://" + ws_or_wss


def normalize_lang_keys(stt_map_items: List[str]) -> Dict[str, str]:
    """
    Build a case-insensitive mapping with helpful aliases:
    - exact (as given)
    - lower and upper variants
    - top-level ISO-639-1 alias (e.g., 'en') if present
    """
    out = {}
    for item in stt_map_items:
        if "=" not in item:
            continue
        lang, host = item.split("=", 1)
        lang = lang.strip()
        host = host.strip()

        # Primary forms
        out[lang] = host
        out[lang.lower()] = host
        out[lang.upper()] = host

        # A short alias "en", "ar"
        parts = lang.replace("_", "-").split("-")
        if parts:
            out[parts[0].lower()] = host
    return out


def coalesce_segments(
    segments: List[Dict[str, Any]], min_duration_sec: float, pad_ms_each_side: int = 120
) -> List[Dict[str, Any]]:
    """
    Merge consecutive segments with the same Language and add small padding
    to avoid clipping at boundaries.

    segments: list with keys: Language, StartTimeHns, EndTimeHns, DurationHns
    returns: list of merged dicts with StartTimeHns/EndTimeHns adjusted.
    """
    if not segments:
        return []

    # Sort by start time to be safe
    segs = sorted(segments, key=lambda s: s["StartTimeHns"])
    merged = []

    current = dict(segs[0])
    for nxt in segs[1:]:
        if (
            nxt["Language"].lower() == current["Language"].lower()
            and nxt["StartTimeHns"] <= current["EndTimeHns"] + 5_000_000
        ):
            # Adjacent (or overlapping) & same language -> extend current
            current["EndTimeHns"] = max(current["EndTimeHns"], nxt["EndTimeHns"])
        else:
            merged.append(current)
            current = dict(nxt)
    merged.append(current)

    # Apply padding
    with_padding = []
    for m in merged:
        start = max(0, m["StartTimeHns"] - pad_ms_each_side * 10_000)
        end = m["EndTimeHns"] + pad_ms_each_side * 10_000
        dur = max(0, end - start)
        if hns_to_sec(dur) >= min_duration_sec:
            with_padding.append(
                {
                    "Language": m["Language"],
                    "StartTimeHns": start,
                    "EndTimeHns": end,
                    "DurationHns": dur,
                }
            )
    return with_padding


def extract_wav_segment(src_wav: str, start_sec: float, duration_sec: float) -> str:
    """
    Slice a WAV file portion to a temp file. We preserve original
    sample rate / channels / width (containers are fine with standard WAV PCM).
    """
    tmp = os.path.join(
        tempfile.gettempdir(),
        f"segment_{uuid.uuid4().hex}_{int(start_sec*1000)}_{int(duration_sec*1000)}.wav",
    )
    with contextlib.ExitStack() as stack:
        src = stack.enter_context(wave.open(src_wav, "rb"))
        n_channels = src.getnchannels()
        sampwidth = src.getsampwidth()
        fr = src.getframerate()

        start_frame = max(0, int(round(start_sec * fr)))
        n_frames = max(0, int(round(duration_sec * fr)))

        # Bound check
        total_frames = src.getnframes()
        if start_frame > total_frames:
            start_frame = total_frames
            n_frames = 0
        if start_frame + n_frames > total_frames:
            n_frames = max(0, total_frames - start_frame)

        src.setpos(start_frame)
        frames = src.readframes(n_frames)

        dst = stack.enter_context(wave.open(tmp, "wb"))
        dst.setnchannels(n_channels)
        dst.setsampwidth(sampwidth)
        dst.setframerate(fr)
        dst.writeframes(frames)

    return tmp


# --------------------------
# LID via Speech SDK (host only, no path)
# --------------------------
def detect_language_segments(
    audio_file: str,
    languages: List[str],
    lid_host: str,
    timeout_sec: float,
    min_seg_sec: float,
    log: logging.Logger,
) -> Dict[str, Any]:
    if not lid_host.startswith(("ws://", "wss://")):
        lid_host = "ws://" + lid_host

    log.info(f"Connecting to LID at {lid_host} (SDK host only; no path)")
    speech_config = speechsdk.SpeechConfig(host=lid_host)

    # Continuous LID
    speech_config.set_property(
        speechsdk.PropertyId.SpeechServiceConnection_LanguageIdMode, "Continuous"
    )

    audio_config = speechsdk.audio.AudioConfig(filename=audio_file)
    auto_detect = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
        languages=languages
    )
    recognizer = speechsdk.SourceLanguageRecognizer(
        speech_config=speech_config,
        auto_detect_source_language_config=auto_detect,
        audio_config=audio_config,
    )

    segments: List[Dict[str, Any]] = []
    done = False

    def on_recognized(evt):
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            detected = evt.result.properties.get(
                speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult
            )
            raw = evt.result.properties.get(
                speechsdk.PropertyId.SpeechServiceResponse_JsonResult
            )
            if detected and raw:
                detail = json.loads(raw)
                start = int(detail.get("Offset", 0))
                dur = int(detail.get("Duration", 0))
                if dur > 0:
                    start_s = hns_to_sec(start)
                    end_s = hns_to_sec(start + dur)
                    log.debug(
                        f"Detected {detected} from {start_s:.2f}s to {end_s:.2f}s"
                    )
                    if hns_to_sec(dur) >= min_seg_sec:
                        segments.append(
                            {
                                "Language": detected,
                                "StartTimeHns": start,
                                "EndTimeHns": start + dur,
                                "DurationHns": dur,
                            }
                        )

    def stop_cb(_):
        nonlocal done
        log.debug("LID session stopped")
        done = True

    recognizer.recognized.connect(on_recognized)
    recognizer.canceled.connect(stop_cb)
    recognizer.session_stopped.connect(stop_cb)

    log.info("Starting continuous LID...")
    recognizer.start_continuous_recognition()

    start_clock = time.time()
    while not done:
        if timeout_sec and (time.time() - start_clock) > timeout_sec:
            log.warning("LID timeout reached; stopping")
            recognizer.stop_continuous_recognition()
            break
        time.sleep(0.2)

    return {
        "AudioFile": audio_file,
        "Segments": segments,
        "SegmentCount": len(segments),
    }


# --------------------------
# STT via Speech SDK (matching LID approach)
# --------------------------
def sdk_transcribe_wav(
    ws_host: str,
    language: str,
    wav_path: str,
    timeout_sec: float,
    log: logging.Logger,
) -> Tuple[str, Dict[str, Any]]:
    """
    Transcribe WAV file using Speech SDK with the same host-only approach as LID.
    Returns (display_text, raw_json)
    """
    if not ws_host.startswith(("ws://", "wss://")):
        ws_host = "ws://" + ws_host

    log.info(f"SDK transcribing {wav_path} with {language} on {ws_host}")

    try:
        # Create speech config with host only, no path (same as LID)
        speech_config = speechsdk.SpeechConfig(host=ws_host)
        speech_config.speech_recognition_language = language

        # Create audio config
        audio_config = speechsdk.audio.AudioConfig(filename=wav_path)

        # Create recognizer
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config, audio_config=audio_config
        )

        # Use synchronous recognition instead of async with events
        log.info("Starting recognition (synchronous)...")
        result = recognizer.recognize_once()

        # Process result
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            log.info(f"RECOGNIZED: {result.text}")
            result_text = result.text

            # Try to get the JSON result
            json_result = result.properties.get(
                speechsdk.PropertyId.SpeechServiceResponse_JsonResult
            )
            if json_result:
                try:
                    result_json = json.loads(json_result)
                except Exception:
                    log.warning("Failed to parse JSON result")
                    result_json = {"raw_text": result.text}
            else:
                result_json = {"raw_text": result.text}

        elif result.reason == speechsdk.ResultReason.NoMatch:
            log.warning(f"NOMATCH: {result.no_match_details}")
            return "", {"error": "no_match", "details": str(result.no_match_details)}

        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation = result.cancellation_details
            log.error(f"CANCELED: {cancellation.reason}")
            if cancellation.reason == speechsdk.CancellationReason.Error:
                log.error(f"Error details: {cancellation.error_details}")
            return "", {
                "error": "canceled",
                "details": (
                    cancellation.error_details
                    if cancellation.reason == speechsdk.CancellationReason.Error
                    else str(cancellation.reason)
                ),
            }

        return result_text, result_json
    except Exception as e:
        log.error(f"Exception during SDK transcription: {str(e)}")
        return "", {"error": "exception", "details": str(e)}


# --------------------------
# STT via REST (container) - DEPRECATED (keeping for reference)
# --------------------------
def rest_transcribe_wav(
    http_base: str,
    language: str,
    wav_path: str,
    timeout_sec: float,
    log: logging.Logger,
) -> Tuple[str, Dict[str, Any]]:
    """
    POST wav bytes to the container REST endpoint.
    Returns (display_text, raw_json)

    Note: This method is deprecated in favor of sdk_transcribe_wav
    """
    log.warning("REST transcription is deprecated; using SDK method instead")
    return sdk_transcribe_wav(
        ws_host=http_base,
        language=language,
        wav_path=wav_path,
        timeout_sec=timeout_sec,
        log=log,
    )


# --------------------------
# Main processing
# --------------------------
def process_audio_file(args, log: logging.Logger):
    # 1) LID (or reuse)
    if not os.path.exists(args.segments) or args.force_detection:
        lid = detect_language_segments(
            audio_file=args.audio,
            languages=args.languages,
            lid_host=args.lid_host,
            timeout_sec=args.timeout_sec,
            min_seg_sec=args.min_segment_sec,
            log=log,
        )
        with open(args.segments, "w", encoding="utf-8") as f:
            json.dump(lid, f, indent=2)
        log.info(f"Wrote {lid['SegmentCount']} language segments to {args.segments}")
    else:
        with open(args.segments, "r", encoding="utf-8") as f:
            lid = json.load(f)
        log.info(f"Loaded {len(lid.get('Segments', []))} segments from {args.segments}")

    # 2) Build mapping for STT containers (aliases & case-insensitive)
    stt_map = normalize_lang_keys(args.stt_map)
    log.info(f"STT host mapping (aliases): {stt_map}")

    segs = lid.get("Segments", [])
    if not segs:
        log.warning("No segments detected; nothing to transcribe.")
        return

    # 3) Coalesce to reduce thrash
    merged = coalesce_segments(
        segs, min_duration_sec=args.min_segment_sec, pad_ms_each_side=args.pad_ms
    )
    log.info(f"Segments to transcribe after merge: {len(merged)}")

    results = []

    # 4) Transcribe each merged region via REST
    for i, m in enumerate(merged):
        lang = m["Language"]
        # Resolve best host for this language
        host = (
            stt_map.get(lang)
            or stt_map.get(lang.lower())
            or stt_map.get(lang.upper())
            or stt_map.get(lang.split("-")[0].lower())
        )
        if not host:
            log.warning(
                f"No STT host configured for language {lang}; skipping segment."
            )
            continue

        start_s = hns_to_sec(m["StartTimeHns"])
        dur_s = hns_to_sec(m["DurationHns"])
        end_s = start_s + dur_s
        log.info(
            f"[{i+1}/{len(merged)}] {lang} {start_s:.2f}s → {end_s:.2f}s on {host}"
        )

        tmp_wav = extract_wav_segment(args.audio, start_s, dur_s)
        try:
            # Use SDK transcribe method with the same host-only approach as LID
            text, raw = sdk_transcribe_wav(
                ws_host=host,
                language=lang,
                wav_path=tmp_wav,
                timeout_sec=args.stt_timeout_sec,
                log=log,
            )
        finally:
            try:
                os.remove(tmp_wav)
            except Exception:
                pass

        results.append(
            {
                "language": lang,
                "start_time_hns": m["StartTimeHns"],
                "duration_hns": m["DurationHns"],
                "start_sec": start_s,
                "duration_sec": dur_s,
                "transcript": text,
                "raw": raw,
            }
        )

    # 5) Assemble final transcript
    results.sort(key=lambda r: r["start_time_hns"])
    full = " ".join(
        [
            f"[{r['language']}] {r['transcript']}".strip()
            for r in results
            if r.get("transcript")
        ]
    )

    final = {
        "audio_file": args.audio,
        "segment_count": len(results),
        "segments": results,
        "full_transcript": full,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)

    log.info(f"Wrote complete transcript to {args.output}")
    log.info(f"Full transcript:\n{full}")


# --------------------------
# CLI
# --------------------------
def main():
    p = argparse.ArgumentParser(
        description="LID + per-language STT (REST) for Azure Speech containers"
    )
    p.add_argument("--audio", required=True, help="Path to WAV file")
    p.add_argument(
        "--languages",
        nargs="+",
        required=True,
        help="Locales for LID, e.g. en-US ar-SA",
    )
    p.add_argument(
        "--lid-host",
        default="ws://localhost:5003",
        help="LID container host (ws://host:port)",
    )
    p.add_argument(
        "--stt-map",
        nargs="+",
        required=True,
        help="Mappings like en-US=ws://localhost:5004 ar-SA=ws://localhost:5005",
    )
    p.add_argument(
        "--segments",
        default="fixed_module_segments.json",
        help="Where to save/load LID segments",
    )
    p.add_argument(
        "--output",
        default="fixed_module_transcript.json",
        help="Where to save the final transcript",
    )
    p.add_argument(
        "--timeout-sec", type=float, default=45.0, help="Max duration for LID pass"
    )
    p.add_argument(
        "--stt-timeout-sec", type=float, default=30.0, help="Per-segment REST timeout"
    )
    p.add_argument(
        "--min-segment-sec",
        type=float,
        default=0.0,
        help="Skip detected LID segments shorter than this duration",
    )
    p.add_argument(
        "--pad-ms",
        type=int,
        default=120,
        help="Padding added to both sides of merged segments (ms)",
    )
    p.add_argument(
        "--force-detection",
        action="store_true",
        help="Force running LID even if segments file exists",
    )
    p.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = p.parse_args()

    log = setup_logging(args.verbose)
    log.info("=== Language-aware transcription (LID via SDK, STT via REST) ===")
    process_audio_file(args, log)


if __name__ == "__main__":
    main()
