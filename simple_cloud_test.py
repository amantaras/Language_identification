"""Minimal Azure Speech cloud STT sanity check.

Usage (PowerShell):
  $env:SPEECH_KEY = '...'
  $env:SPEECH_REGION = 'eastus2'  # or set SPEECH_ENDPOINT instead
  python simple_cloud_test.py --audio audio\samples\Arabic_english_mix_optimized.wav --language en-US --verbose

This bypasses LID/segmentation. It just pushes entire file as a single recognize_once
using region or endpoint from environment. Prints detailed cancellation diagnostics.
"""

from __future__ import annotations
import argparse
import os
import wave
import audioop
import azure.cognitiveservices.speech as speechsdk


def load_wav_as_pcm16_mono_16k(path: str) -> bytes:
    with wave.open(path, "rb") as wf:
        ch = wf.getnchannels()
        width = wf.getsampwidth()
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    if width != 2:
        frames = audioop.lin2lin(frames, width, 2)
        width = 2
    if ch > 1:
        frames = audioop.tomono(frames, 2, 0.5, 0.5)
    if rate != 16000:
        frames, _ = audioop.ratecv(frames, 2, 1, rate, 16000, None)
    return frames


def build_config(lang: str):
    key = (os.getenv("SPEECH_KEY") or os.getenv("AZURE_SPEECH_KEY") or "").strip()
    endpoint = (
        os.getenv("SPEECH_ENDPOINT") or os.getenv("AZURE_SPEECH_ENDPOINT") or ""
    ).strip()
    region = (
        os.getenv("SPEECH_REGION") or os.getenv("AZURE_SPEECH_REGION") or ""
    ).strip()
    if not key:
        raise SystemExit("Missing SPEECH_KEY/AZURE_SPEECH_KEY in environment")
    if endpoint:
        if "/speech/" not in endpoint:
            endpoint = (
                endpoint.rstrip("/")
                + "/speech/recognition/conversation/cognitiveservices/v1"
            )
        cfg = speechsdk.SpeechConfig(endpoint=endpoint, subscription=key)
    elif region:
        cfg = speechsdk.SpeechConfig(subscription=key, region=region)
    else:
        raise SystemExit("Need SPEECH_ENDPOINT or SPEECH_REGION environment variable")
    cfg.speech_recognition_language = lang
    cfg.output_format = speechsdk.OutputFormat.Detailed
    return cfg


def run(audio: str, language: str, verbose: bool):
    pcm = load_wav_as_pcm16_mono_16k(audio)
    cfg = build_config(language)
    fmt = speechsdk.audio.AudioStreamFormat(
        samples_per_second=16000, bits_per_sample=16, channels=1
    )
    push = speechsdk.audio.PushAudioInputStream(stream_format=fmt)
    audio_cfg = speechsdk.audio.AudioConfig(stream=push)
    rec = speechsdk.SpeechRecognizer(speech_config=cfg, audio_config=audio_cfg)
    push.write(pcm)
    push.close()
    res = rec.recognize_once()
    print("Reason:", res.reason)
    if res.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Text:", res.text)
    elif res.reason == speechsdk.ResultReason.NoMatch:
        print("NoMatch:", res.no_match_details)
    elif res.reason == speechsdk.ResultReason.Canceled:
        details = speechsdk.CancellationDetails.from_result(res)
        print("Canceled reason:", details.reason)
        print("Error details:", details.error_details)
        # Dump a few properties for debugging
        for pid in [
            speechsdk.PropertyId.SpeechServiceConnection_JsonResult,
            speechsdk.PropertyId.SpeechServiceResponse_JsonResult,
        ]:
            try:
                val = res.properties.get(pid)
                if val:
                    print(str(pid), ":", val[:500])
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--language", default="en-US")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    run(args.audio, args.language, args.verbose)


if __name__ == "__main__":
    main()
