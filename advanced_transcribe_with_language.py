"""
Advanced multilingual transcription against Azure Speech containers.

- Verifies containers via /ready and /status.
- Uses Speech SDK host auth (ws://...) per official docs.
- Streams 16 kHz, 16-bit PCM mono via PushAudioInputStream (no temp files).
- Processes segments per-language sequentially to keep containers happy.
- Logs full cancellation diagnostics.

Requires:
  pip install azure-cognitiveservices-speech requests

Environment (optional):
  SPEECH_LID_HOST           (default: ws://localhost:5003)  # Only used by your detect_languages()
  SPEECH_EN_US_HOST         (default: ws://localhost:5004)
  SPEECH_AR_SA_HOST         (default: ws://localhost:5005)
  SPEECH_CONTAINER_HTTP_MAP (optional) JSON map of http probes if your WS ports differ, e.g.:
      {"ws://localhost:5004": "http://localhost:5004", "ws://localhost:5005": "http://localhost:5005"}
"""

import os
import json
import wave
import time
import uuid
import logging
import audioop
import requests
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Optional

import azure.cognitiveservices.speech as speechsdk
from app.language_detect_improved import detect_languages

# ---------- logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            "advanced_transcription_test.txt", mode="w", encoding="utf-8"
        ),
    ],
)
log = logging.getLogger("adv-stt")


# ---------- container health ----------
def http_base_for_ws(ws_url: str) -> str:
    """Translate ws://host:port to http://host:port for readiness/status probes."""
    try:
        mapping_json = os.getenv("SPEECH_CONTAINER_HTTP_MAP")
        if mapping_json:
            m = json.loads(mapping_json)
            if ws_url in m:
                return m[ws_url]
    except Exception:
        pass
    if ws_url.startswith("ws://"):
        return "http://" + ws_url[len("ws://") :]
    if ws_url.startswith("wss://"):
        return "https://" + ws_url[len("wss://") :]
    return ws_url


def probe_container(ws_host: str, timeout=5.0) -> Tuple[bool, str]:
    """Check /ready and /status for a Speech container."""
    base = http_base_for_ws(ws_host).rstrip("/")
    try:
        r1 = requests.get(f"{base}/ready", timeout=timeout)
        r2 = requests.get(f"{base}/status", timeout=timeout)
        ok = (r1.status_code == 200) and (r2.status_code == 200)
        msg = f"/ready={r1.status_code} /status={r2.status_code}"
        return ok, msg
    except Exception as e:
        return False, f"probe error: {e}"


# ---------- audio helpers ----------
def read_segment_as_pcm16_mono_16k(
    wav_path: str, start_sec: float, duration_sec: float
) -> bytes:
    """
    Extract a segment from a WAV file and convert to 16 kHz, 16-bit PCM mono.
    Returns raw PCM bytes ready to push into PushAudioInputStream.
    """
    with wave.open(wav_path, "rb") as wf:
        src_ch = wf.getnchannels()
        src_sampwidth = wf.getsampwidth()
        src_rate = wf.getframerate()

        start_frame = max(0, int(start_sec * src_rate))
        frames_to_read = int(duration_sec * src_rate)
        wf.setpos(min(start_frame, wf.getnframes()))
        raw = wf.readframes(frames_to_read)

    # 1) to mono
    if src_ch > 1:
        # average channels
        raw = audioop.tomono(raw, src_sampwidth, 0.5, 0.5)

    # 2) to 16-bit
    if src_sampwidth != 2:
        raw = audioop.lin2lin(raw, src_sampwidth, 2)

    # 3) resample to 16k
    if src_rate != 16000:
        raw, _state = audioop.ratecv(raw, 2, 1, src_rate, 16000, None)

    return raw


# ---------- recognizer wrapper ----------
class STTContainer:
    def __init__(self, host_ws: str, locale: str):
        """
        host_ws: e.g. ws://localhost:5004
        locale:  e.g. en-US, ar-SA
        """
        self.host = host_ws.rstrip("/")  # SDK is fine with no trailing slash
        self.locale = locale
        self.speech_config = speechsdk.SpeechConfig(host=self.host)
        # IMPORTANT: for containers you generally should NOT set subscription/region.
        self.speech_config.speech_recognition_language = locale

        # nice-to-have: detailed output
        self.speech_config.output_format = speechsdk.OutputFormat.Detailed

        # event plumbing happens per recognizer

    def _new_push_stream_and_config(
        self,
    ) -> Tuple[speechsdk.audio.PushAudioInputStream, speechsdk.audio.AudioConfig]:
        fmt = speechsdk.audio.AudioStreamFormat(
            samples_per_second=16000,
            bits_per_sample=16,
            channels=1,
        )
        push_stream = speechsdk.audio.PushAudioInputStream(stream_format=fmt)
        audio_config = speechsdk.audio.AudioConfig(stream=push_stream)
        return push_stream, audio_config

    def transcribe_pcm16_mono_16k(
        self, pcm: bytes, log_prefix: str
    ) -> Tuple[str, Optional[str], Optional[str]]:
        """
        Streams raw PCM into the container and returns (text, cancel_reason, cancel_details).
        """
        push_stream, audio_cfg = self._new_push_stream_and_config()
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=self.speech_config, audio_config=audio_cfg
        )

        # attach event handlers to glean detail if it cancels
        cancel_reason_holder = {"reason": None, "details": None, "error_code": None}

        def _canceled(evt: speechsdk.SpeechRecognitionCanceledEventArgs):
            cancel_reason_holder["reason"] = str(evt.reason)
            cancel_reason_holder["details"] = str(evt.error_details)
            cancel_reason_holder["error_code"] = str(evt.error_code)

        recognizer.canceled.connect(_canceled)

        # Helpful to pre-open the websocket to force fast failures
        try:
            conn = speechsdk.Connection.from_recognizer(recognizer)
            conn.open(True)
        except Exception as e:
            log.error(f"{log_prefix} connection open failed: {e}")

        # stream audio
        push_stream.write(pcm)
        push_stream.close()

        # single-utterance recognition for a short segment
        result = recognizer.recognize_once()

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return result.text, None, None

        if result.reason == speechsdk.ResultReason.NoMatch:
            return "", "NoMatch", "No speech recognized"

        if result.reason == speechsdk.ResultReason.Canceled:
            details = speechsdk.CancellationDetails(result)
            # Prefer handler’s values if present
            cr = cancel_reason_holder["reason"] or str(details.reason)
            ed = cancel_reason_holder["details"] or str(details.error_details)
            ec = cancel_reason_holder["error_code"]
            msg = f"{cr}: {ed}" + (f" (code={ec})" if ec else "")
            return "", "Canceled", msg

        # unexpected
        return "", "Unknown", f"Unhandled result.reason={result.reason}"


# ---------- pipeline ----------
def advanced_transcribe_with_language():
    log.info("=== TESTING ADVANCED MULTILINGUAL TRANSCRIPTION ===")

    # --- config ---
    lid_host = os.getenv(
        "SPEECH_LID_HOST", "ws://localhost:5003"
    )  # your LID container (used by detect_languages)
    audio_file = "audio/samples/Arabic_english_mix_optimized.wav"
    locales = ["en-US", "ar-SA"]
    segments_out = "advanced_segments.json"
    transcript_out = "advanced_transcript.json"

    stt_host_map: Dict[str, str] = {
        "en-US": os.getenv("SPEECH_EN_US_HOST", "ws://localhost:5004"),
        "ar-SA": os.getenv("SPEECH_AR_SA_HOST", "ws://localhost:5005"),
    }

    log.info(f"Using language detection host: {lid_host}")
    log.info(f"Languages: {locales}")
    log.info(f"Audio file: {os.path.abspath(audio_file)}")
    log.info(f"Output segments: {os.path.abspath(segments_out)}")
    log.info(f"Output transcript: {os.path.abspath(transcript_out)}")
    log.info(f"STT host mapping: {stt_host_map}")

    # --- Step 0: sanity-check STT containers are alive and licensed ---
    for lang, ws in stt_host_map.items():
        ok, msg = probe_container(ws)
        if not ok:
            log.error(f"[{lang}] container {ws} not ready: {msg}")
        else:
            log.info(f"[{lang}] container {ws} OK: {msg}")

    # --- Step 1: LID segments ---
    log.info("\nStep 1: Detecting language segments...")
    lid_result = detect_languages(
        audio_file=audio_file,
        lid_host=lid_host,
        languages=locales,
        out_segments=segments_out,
        timeout_sec=60,
        logger=log,
        _force_continuous=True,  # you were already doing continuous; keep it
    )

    # --- Step 2: load segments ---
    log.info("\nStep 2: Loading language segments...")
    with open(segments_out, "r", encoding="utf-8") as f:
        seg_data = json.load(f)
    segments = seg_data.get("segments", [])
    log.info(f"Found {len(segments)} language segments")

    # group by language to **serialize per-container**
    by_lang: Dict[str, List[dict]] = {}
    for s in segments:
        # your LID returns language keys in various casings; normalize to match map
        lang = s.get("language")
        # normalize canonical casing if needed (en-us -> en-US)
        if lang and lang.lower() == "en-us":
            lang = "en-US"
        if lang and lang.lower() == "ar-sa":
            lang = "ar-SA"
        s["language"] = lang
        by_lang.setdefault(lang, []).append(s)

    # --- Step 3: Init STT clients ---
    log.info("\nStep 3: Initializing STT container clients...")
    clients: Dict[str, STTContainer] = {}
    for lang, host in stt_host_map.items():
        try:
            clients[lang] = STTContainer(host_ws=host, locale=lang)
            log.info(f"Initialized client for {lang} at {host}")
        except Exception as e:
            log.error(f"Failed to init client for {lang} at {host}: {e}")

    # --- Step 4: Transcribe (serialize per language, small parallelism across languages) ---
    log.info("\nStep 4: Transcribing segments per language...")

    # function to process one language’s queue sequentially
    def process_language(lang: str) -> List[dict]:
        out: List[dict] = []
        client = clients.get(lang)
        if not client:
            for seg in by_lang.get(lang, []):
                out.append(
                    {
                        "language": lang,
                        "start_time": seg["start_hns"],
                        "duration": seg["end_hns"] - seg["start_hns"],
                        "start_sec": seg["start_hns"] / 10_000_000.0,
                        "duration_sec": (seg["end_hns"] - seg["start_hns"])
                        / 10_000_000.0,
                        "transcript": "",
                        "error": f"No client for language {lang}",
                    }
                )
            return out

        for seg in by_lang.get(lang, []):
            start_hns = seg["start_hns"]
            end_hns = seg["end_hns"]
            start_sec = start_hns / 10_000_000.0
            duration_sec = (end_hns - start_hns) / 10_000_000.0
            pretty = f"[{lang}] {start_sec:.3f}s–{start_sec+duration_sec:.3f}s"
            log.info(f"Transcribing {pretty}")

            try:
                pcm = read_segment_as_pcm16_mono_16k(
                    audio_file, start_sec, duration_sec
                )
                text, cancel_reason, cancel_details = client.transcribe_pcm16_mono_16k(
                    pcm, log_prefix=f"{pretty}"
                )
                rec = {
                    "language": lang,
                    "start_time": start_hns,
                    "duration": end_hns - start_hns,
                    "start_sec": start_sec,
                    "duration_sec": duration_sec,
                    "transcript": text or "",
                }
                if cancel_reason:
                    rec["error"] = f"{cancel_reason}: {cancel_details}"
                    log.error(f"{pretty} {rec['error']}")
                elif not text:
                    rec["note"] = "No speech recognized"
                    log.warning(f"{pretty} no speech recognized")
                else:
                    log.info(f"{pretty} -> {text}")
                out.append(rec)
            except Exception as e:
                log.exception(f"{pretty} streaming error: {e}")
                out.append(
                    {
                        "language": lang,
                        "start_time": start_hns,
                        "duration": end_hns - start_hns,
                        "start_sec": start_sec,
                        "duration_sec": duration_sec,
                        "transcript": "",
                        "error": f"Exception: {e}",
                    }
                )
        return out

    results: List[dict] = []
    # slight parallelism: run English queue and Arabic queue in parallel, but each queue is serial
    with ThreadPoolExecutor(max_workers=min(2, len(by_lang))) as ex:
        futs = []
        for lang in by_lang.keys():
            futs.append(ex.submit(process_language, lang))
        for f in futs:
            results.extend(f.result())

    # --- Step 5: Final glue ---
    results_sorted = sorted(results, key=lambda r: r["start_time"])
    full_text = " ".join(
        [
            f"[{r['language']}] {r['transcript']}".strip()
            for r in results_sorted
            if r.get("transcript")
        ]
    )

    final = {
        "audio_file": audio_file,
        "segment_count": len(results_sorted),
        "transcribed_segments": results_sorted,
        "full_transcript": full_text,
    }

    with open(transcript_out, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    log.info("\nStep 5: Done")
    log.info(f"Wrote complete transcript to {transcript_out}")
    log.info(f"Full transcript: {full_text}")


if __name__ == "__main__":
    advanced_transcribe_with_language()
