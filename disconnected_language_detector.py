"""
fixed_transcribe_with_language.py

Streaming-ish pipeline:
  • LID: Azure Speech CONTAINER over WebSocket (host only, NO resource path).
  • Transcription: Azure Speech CLOUD (SDK) via ENV VARS ONLY.
  • True async handoff: when LID detects a language switch, immediately spawn a
    subprocess to transcribe the closed segment while LID continues.

ENV for cloud STT:
  SPEECH_KEY / AZURE_SPEECH_KEY              (REQUIRED)
  SPEECH_REGION / AZURE_SPEECH_REGION        (PREFERRED)
  SPEECH_ENDPOINT / AZURE_SPEECH_ENDPOINT    (optional; if both region+endpoint present, region is used)

No ws:// for cloud, no host= for cloud — SDK does transport.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import logging
import os
import sys
import time
import wave
from typing import Dict, List, Optional, Tuple

# audioop is deprecated in 3.13; fine in 3.12 (your env).
import audioop

import azure.cognitiveservices.speech as speechsdk


# ---------- tiny dotenv ----------
def _load_dotenv(path: str = ".env", log: Optional[logging.Logger] = None) -> None:
    if not os.path.isfile(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                if k in os.environ:
                    continue
                os.environ[k] = v.strip().strip('"').strip("'")
        if log:
            log.debug("Loaded variables from .env")
    except Exception as e:
        if log:
            log.debug(f".env load skipped: {e}")


# ---------- logging ----------
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
    return logging.getLogger("lid_cloud_async")


# ---------- helpers ----------
HNS_PER_SEC = 10_000_000
MAX_SHOT_SEC = 8.0
TRAIL_SILENCE_SEC = 1.8


def hns_to_sec(hns: int) -> float:
    return hns / HNS_PER_SEC


def sec_to_hns(sec: float) -> int:
    return int(sec * HNS_PER_SEC)


def canon_lang(lang: str) -> str:
    if not lang:
        return ""
    p = lang.replace("_", "-").split("-")
    if len(p) == 1:
        return p[0].lower()
    return f"{p[0].lower()}-{p[1].upper()}"


def collapse_supported(code: str) -> str:
    if not code:
        return ""
    low = code.lower()
    if low.startswith("en"):
        return "en-US"
    if low.startswith("ar"):
        return "ar-SA"
    return canon_lang(code)


# ---------- PCM extraction ----------
def read_segment_as_pcm16_mono_16k(
    wav_path: str, start_hns: int, duration_hns: int
) -> bytes:
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

    if width != 2:
        raw = audioop.lin2lin(raw, width, 2)
        width = 2
    if ch > 1:
        raw = audioop.tomono(raw, 2, 0.5, 0.5)
        ch = 1
    if rate != 16000:
        raw, _ = audioop.ratecv(raw, 2, 1, rate, 16000, None)
    return raw


# ---------- Cloud STT (ENV ONLY) ----------
class AzureCloudSTT:
    def __init__(self, log: Optional[logging.Logger] = None):
        self.log = log
        self.key = (os.getenv("SPEECH_KEY") or os.getenv("AZURE_SPEECH_KEY") or "").strip()
        self.endpoint = (os.getenv("SPEECH_ENDPOINT") or os.getenv("AZURE_SPEECH_ENDPOINT") or "").strip()
        self.region = (os.getenv("SPEECH_REGION") or os.getenv("AZURE_SPEECH_REGION") or "").strip()

        if not self.key:
            raise ValueError("Missing SPEECH_KEY/AZURE_SPEECH_KEY in environment.")

        # Prefer region over endpoint if both set (more robust)
        if self.endpoint and self.region:
            if self.log:
                self.log.warning("Both endpoint and region present; preferring REGION.")
            self.endpoint = ""

        # If endpoint present but missing the /speech/ path, append conversation path
        if self.endpoint and "/speech/" not in self.endpoint:
            self.endpoint = self.endpoint.rstrip("/") + "/speech/recognition/conversation/cognitiveservices/v1"
            if self.log:
                self.log.warning("Endpoint missing path; appended conversation path.")

        self.mode = "endpoint" if self.endpoint else "region"
        if self.log:
            self.log.info(f"Azure STT mode: {self.mode.upper()} from environment.")

        self._cfg_cache: Dict[str, speechsdk.SpeechConfig] = {}

    def _speech_config_for_language(self, language: str) -> speechsdk.SpeechConfig:
        lang = collapse_supported(language)
        if lang in self._cfg_cache:
            return self._cfg_cache[lang]

        if self.mode == "endpoint":
            cfg = speechsdk.SpeechConfig(endpoint=self.endpoint, subscription=self.key)
        else:
            cfg = speechsdk.SpeechConfig(subscription=self.key, region=self.region)
        cfg.speech_recognition_language = lang
        cfg.output_format = speechsdk.OutputFormat.Detailed
        cfg.set_property(speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs, "1800")
        cfg.set_property(speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, "3000")
        self._cfg_cache[lang] = cfg
        return cfg

    def transcribe_shot(self, pcm16_mono_16k: bytes, language: str) -> Dict:
        cfg = self._speech_config_for_language(language)
        fmt = speechsdk.audio.AudioStreamFormat(samples_per_second=16000, bits_per_sample=16, channels=1)
        push = speechsdk.audio.PushAudioInputStream(stream_format=fmt)
        audio_cfg = speechsdk.audio.AudioConfig(stream=push)
        rec = speechsdk.SpeechRecognizer(speech_config=cfg, audio_config=audio_cfg)

        # Append trailing zeros to force final result
        pcm = pcm16_mono_16k + (b"\x00\x00" * int(16000 * TRAIL_SILENCE_SEC))
        push.write(pcm)
        push.close()

        res = rec.recognize_once()
        if res.reason == speechsdk.ResultReason.RecognizedSpeech:
            return {"status": "success", "text": res.text or ""}
        if res.reason == speechsdk.ResultReason.NoMatch:
            return {"status": "nomatch", "note": "No speech recognized"}
        if res.reason == speechsdk.ResultReason.Canceled:
            try:
                det = speechsdk.CancellationDetails.from_result(res)
                return {"status": "error", "error": f"{det.reason}: {det.error_details or 'Unknown'}"}
            except Exception:
                return {"status": "error", "error": "Canceled: Unknown"}
        return {"status": "error", "error": f"Unexpected: {res.reason}"}


def _split_into_shots(start_hns: int, dur_hns: int) -> List[Tuple[int, int]]:
    shots: List[Tuple[int, int]] = []
    total = dur_hns
    max_hns = sec_to_hns(MAX_SHOT_SEC)
    cur = start_hns
    while total > 0:
        take = min(total, max_hns)
        shots.append((cur, take))
        cur += take
        total -= take
    return shots


# ---------- SUBPROCESS WORKER ----------
def _worker_transcribe_segment(
    audio_file: str, language: str, start_hns: int, duration_hns: int
) -> Dict:
    """
    Runs in a separate process. Builds AzureCloudSTT from env (inherited),
    splits the segment into ≤8s shots, transcribes, and returns a result dict.
    """
    try:
        stt = AzureCloudSTT(log=None)  # no cross-process logger
    except Exception as e:
        return {
            "language": collapse_supported(language),
            "start_time": start_hns,
            "duration": duration_hns,
            "start_sec": hns_to_sec(start_hns),
            "duration_sec": hns_to_sec(duration_hns),
            "transcript": "",
            "error": f"Cloud STT init failed: {e}",
            "mode": "cloud",
        }

    lang = collapse_supported(language)
    seg_texts: List[str] = []
    for s_start, s_dur in _split_into_shots(start_hns, duration_hns):
        pcm = read_segment_as_pcm16_mono_16k(audio_file, s_start, s_dur)
        if not pcm:
            continue
        r = stt.transcribe_shot(pcm, lang)
        if r["status"] == "success" and r.get("text"):
            seg_texts.append(r["text"])
        # ignore errors here; they’ll just lose this shot’s text

    return {
        "language": lang,
        "start_time": start_hns,
        "duration": duration_hns,
        "start_sec": hns_to_sec(start_hns),
        "duration_sec": hns_to_sec(duration_hns),
        "transcript": " ".join(seg_texts).strip(),
        "mode": "cloud",
    }


# ---------- LIVE LID → submit jobs immediately ----------
def stream_lid_and_submit(
    audio_file: str,
    languages: List[str],
    lid_host: str,
    timeout_sec: float,
    min_segment_sec: float,
    log: logging.Logger,
    pool: cf.ProcessPoolExecutor,
) -> Tuple[List[cf.Future], List[Dict]]:
    """
    Runs LID continuously and submits transcription jobs immediately when a language
    switch occurs. Returns (futures, submitted_segment_meta_list).
    """
    if not lid_host.startswith(("ws://", "wss://")):
        lid_host = "ws://" + lid_host

    speech_config = speechsdk.SpeechConfig(host=lid_host)
    speech_config.set_property(
        speechsdk.PropertyId.SpeechServiceConnection_LanguageIdMode, "Continuous"
    )

    lid_langs = [canon_lang(code) for code in languages]
    audio_config = speechsdk.audio.AudioConfig(filename=audio_file)
    auto_detect = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
        languages=lid_langs
    )
    rec = speechsdk.SourceLanguageRecognizer(
        speech_config=speech_config,
        auto_detect_source_language_config=auto_detect,
        audio_config=audio_config,
    )

    jobs: List[cf.Future] = []
    metas: List[Dict] = []

    current_lang: Optional[str] = None
    seg_start_hns: Optional[int] = None
    last_seen_end_hns: int = 0
    done = False

    def submit_segment(lang: str, start_hns: int, end_hns: int):
        dur_hns = max(0, end_hns - start_hns)
        dur_sec = hns_to_sec(dur_hns)
        if min_segment_sec and dur_sec < min_segment_sec:
            log.debug(f"[LID] Skip short seg {lang} {dur_sec:.2f}s")
            return
        fut = pool.submit(_worker_transcribe_segment, audio_file, lang, start_hns, dur_hns)
        jobs.append(fut)
        metas.append(
            {
                "Language": lang,
                "StartTimeHns": start_hns,
                "DurationHns": dur_hns,
            }
        )
        log.info(
            f"[DISPATCH] {lang} {hns_to_sec(start_hns):.2f}s → {hns_to_sec(end_hns):.2f}s (cloud)"
        )

    def on_recognized(evt):
        nonlocal current_lang, seg_start_hns, last_seen_end_hns
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
            end_hns = start_hns + dur_hns
            last_seen_end_hns = max(last_seen_end_hns, end_hns)

            det = collapse_supported(detected)
            if current_lang is None:
                # first detection → start first segment
                current_lang = det
                seg_start_hns = start_hns
                log.debug(f"[LID] Start {det} @ {hns_to_sec(start_hns):.2f}s")
                return

            if det != current_lang:
                # language switch → close previous seg at this start
                if seg_start_hns is not None:
                    submit_segment(current_lang, seg_start_hns, start_hns)
                # new current
                current_lang = det
                seg_start_hns = start_hns
                log.debug(f"[LID] Switch → {det} @ {hns_to_sec(start_hns):.2f}s")
        except Exception as e:
            log.debug(f"[LID] JSON parse error: {e}")

    def on_stop(_):
        nonlocal done
        done = True

    rec.recognized.connect(on_recognized)
    rec.session_stopped.connect(on_stop)
    rec.canceled.connect(on_stop)

    log.info("Starting continuous LID (live submit)…")
    rec.start_continuous_recognition()

    t0 = time.time()
    while not done:
        if timeout_sec and (time.time() - t0) > timeout_sec:
            log.warning(f"LID timeout after {timeout_sec}s — stopping.")
            rec.stop_continuous_recognition()
            break
        time.sleep(0.25)

    # Close the trailing open segment, if any
    if current_lang and seg_start_hns is not None and last_seen_end_hns > seg_start_hns:
        submit_segment(current_lang, seg_start_hns, last_seen_end_hns)

    return jobs, metas


# ---------- Pipeline ----------
def process_audio_file(args, log: logging.Logger):
    # Load env (.env optional)
    _load_dotenv(log=log)

    # Process pool for async transcription workers
    # Use max_workers=number of CPU cores or a small fixed number (2–4) to avoid throttling
    max_workers = max(2, os.cpu_count() or 2)
    log.info(f"Starting transcription pool with {max_workers} workers.")
    futures: List[cf.Future] = []
    submitted_meta: List[Dict] = []

    with cf.ProcessPoolExecutor(max_workers=max_workers) as pool:
        # Live LID and immediate job dispatch
        futures, submitted_meta = stream_lid_and_submit(
            audio_file=args.audio,
            languages=args.languages,
            lid_host=args.lid_host,
            timeout_sec=args.timeout_sec,
            min_segment_sec=args.min_segment_sec,
            log=log,
            pool=pool,
        )

        # Optional: while LID is running we could poll finished jobs.
        # Simpler: once LID ended and final job submitted, collect all results here.

        log.info(f"Waiting for {len(futures)} transcription jobs…")
        results: List[Dict] = []
        for fut in cf.as_completed(futures):
            try:
                res = fut.result()
                results.append(res)
            except Exception as e:
                log.error(f"Transcription worker failed: {e}")

    # Sort and stitch
    results.sort(key=lambda r: r.get("start_time", 0))
    full = " ".join(
        f"[{r['language']}] {r.get('transcript','')}".strip()
        for r in results
        if r.get("transcript")
    )

    final = {
        "audio_file": args.audio,
        "segment_count": len(results),
        "transcribed_segments": results,
        "full_transcript": full,
        "submitted_segments": submitted_meta,  # for debugging/traceability
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    log.info(f"Wrote complete transcript to {args.output}")
    log.info(f"Full transcript: {final['full_transcript']}")
    return final


# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(
        description="LIVE LID (container) → async CLOUD STT per language segment (subprocess)."
    )
    parser.add_argument("--audio", required=True, help="Path to WAV audio file")
    parser.add_argument(
        "--languages", nargs="+", required=True, help="Languages for LID (e.g., en-US ar-SA)"
    )
    parser.add_argument(
        "--lid-host",
        default="ws://localhost:5003",
        help="LID container host (ws://host:port, NO path)",
    )
    parser.add_argument(
        "--output", default="fixed_module_transcript.json", help="Where to save final transcript JSON"
    )
    parser.add_argument("--timeout-sec", type=float, default=60.0, help="LID timeout seconds")
    parser.add_argument(
        "--min-segment-sec",
        type=float,
        default=0.0,
        help="Drop LID segments shorter than this many seconds",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()
    log = setup_logging(args.verbose)

    process_audio_file(args, log)


if __name__ == "__main__":
    main()
