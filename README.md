# Live Language Switch Aware Transcription (Container LID + Cloud STT)

## Quick Run Example (PowerShell)

The fastest way to try the live pipeline once your LID container is running and your Speech cloud key/region are in the environment (or `.env`) is:

```pwsh
python disconnected_language_detector.py `
  --audio "audio\samples\Arabic_english_mix_optimized.wav" `
  --languages en-US ar-SA `
  --lid-host ws://localhost:5003 `
  --segments fixed_module_segments.json `
  --output fixed_module_transcript.json `
  --force-detection `
  --verbose
```

### Parameter Breakdown

| Flag | Required | Example | Purpose |
|------|----------|---------|---------|
| `--audio` | Yes | `audio\samples\Arabic_english_mix_optimized.wav` | Path to a WAV (16 kHz not required; script resamples). Full or relative path accepted. |
| `--languages` | Yes (>=1) | `en-US ar-SA` | Candidate languages for LID. Order does not matter. Provide valid BCP‑47 codes. Detection will only choose among these. |
| `--lid-host` | No (default `ws://localhost:5003`) | `ws://localhost:5003` | WebSocket host:port for the Language Detection container. Supply host:port; if you omit `ws://` it is added. Must NOT include `/speech/` path. |
| `--segments` | No | `fixed_module_segments.json` | Where merged LID segments (audit reference) are written at the end. You can choose any filename. |
| `--output` | No | `fixed_module_transcript.json` | Final transcript JSON (recognized_results + full_transcript). Overwritten each run (and also after each language block if `--flush-after-stop` is used). |
| `--force-detection` | Optional (legacy) | (present) | Kept for backward compatibility with older scripts; live mode always performs LID so this flag currently has no effect. Safe to include or omit. |
| `--verbose` | Optional | (present) | Enables debug logging: partial hypotheses, raw detection spans, internal timing. Omit for quieter logs. |

Additional useful live‑mode flags (not shown above):

| Flag | Default | Why adjust? |
|------|---------|-------------|
| `--timeout-sec` | 60 | Maximum wall‑clock time to wait for LID completion (stop earlier if audio shorter). Set 0 to disable. |
| `--chunk-ms` | 40 | Audio push size to cloud STT. Smaller = lower latency, more calls. Larger = fewer calls, potentially higher latency. Typical 20–60. |
| `--tail-overlap-ms` | 200 | Extra audio pushed after a detected language end before stopping recognizer to avoid cutting trailing phonemes. Increase (e.g. 300–400) if you see clipped last words. |
| `--flush-after-stop` | (off) | Write `--output` incrementally after each language block; useful for very long files / monitoring progress. |

Environment expectations (choose one mode):
* Region mode: `SPEECH_KEY` + `SPEECH_REGION` (or `AZURE_SPEECH_KEY` / `AZURE_SPEECH_REGION`).
* Endpoint mode: `SPEECH_KEY` + full `SPEECH_ENDPOINT` containing `/speech/` path (auto‑appended if missing).

If these are in `.env`, the script auto‑loads them unless already present in the real environment.

---

Modern pipeline for mixed‑language audio where we want:

* Real‑time (single pass) continuous Language Identification (LID) from a local Azure Speech container.
* Immediate handoff to Azure Speech **cloud** STT (SDK) for the currently active language – switching recognizers on the fly when LID changes.
* Fine‑grained logging of partial (~) and final (✔) hypotheses while running.
* Automatic segment merging + optional incremental flushing of results to disk.

Implemented in `disconnected_language_detector.py` (name will likely be shortened later but kept for clarity in commits).

> Previous README described a two‑pass “detect then slice then per‑language container STT” flow. The current default is a **live streaming** single pass: container only for LID; cloud for recognition.

## High‑Level Architecture

1. **LID Container (WebSocket)** – You run only the Language Detection container locally (`--lid-host`, default `ws://localhost:5003`). We connect using `speechsdk.SpeechConfig(host=...)` (HOST ONLY, no `/speech/` path).
2. **LID Event Stream** – Each recognized event includes an offset + duration + detected language. We accumulate language spans.
3. **Live Cloud STT Stream** – On first language detection we start a cloud SpeechRecognizer (PushAudioInputStream). Raw audio is fed chunk by chunk. When LID signals a language switch we gracefully stop the current recognizer, store its transcript, then spin up a new recognizer for the new language and continue feeding audio from that moment onward (with a tail overlap to avoid boundary loss).
4. **Merging & Output** – Finalized per‑language blocks become “segments”. A consolidated transcript plus raw segment metadata is persisted to JSON (`--output`). LID merged segments (audit) are also written to `--segments`.

## Key Features

* Continuous multi‑language LID via container.
* Dynamic runtime STT switching (no second pass slicing required).
* Overlap tail padding (`--tail-overlap-ms`) to reduce word truncation at switches.
* Adjustable push chunk size (`--chunk-ms`).
* Optional incremental flushing after each language stop (`--flush-after-stop`).
* `.env` autoload (values are only applied if not already present as real env vars).
* Robust cancellation/diagnostic logging on errors.

## Environment Variables (Cloud STT)

One of the following sets must be present for cloud STT:

| Variable | Purpose |
|----------|---------|
| `SPEECH_KEY` or `AZURE_SPEECH_KEY` | Speech resource key (required) |
| (`SPEECH_REGION` or `AZURE_SPEECH_REGION`) | Region (use if endpoint not supplied) |
| OR `SPEECH_ENDPOINT` / `AZURE_SPEECH_ENDPOINT` | Full HTTPS/WSS endpoint (if used, must contain `/speech/` path; we auto‑append if missing) |

The script loads a local `.env` automatically at startup (lines with `KEY=VALUE`, ignoring comments) but does not overwrite already‑exported env vars.

Minimal `.env` example:
```
AZURE_SPEECH_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
AZURE_SPEECH_REGION=eastus2
LID_PORT=5003
```

If you prefer endpoint mode (sometimes needed for network routing):
```
AZURE_SPEECH_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
AZURE_SPEECH_ENDPOINT=https://eastus2.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1
```

## Installation

```pwsh
python -m venv .venv
./.venv/Scripts/Activate.ps1
pip install --upgrade pip
pip install -e .
```

> `azure-cognitiveservices-speech` is specified in `requirements.txt`.

## Run the LID Container

You only need the LID container for this live mode. (STT containers are not required unless you use hybrid scripts.)

PowerShell helper (if you kept the original scripts) – adjust image tag if needed:
```pwsh
./scripts/run-containers.ps1 -EnvFile ./.env -OnlyLid
```
If `-OnlyLid` isn’t implemented, just start the language detection container manually mapping `LID_PORT`.

Stop when done:
```pwsh
./scripts/stop-containers.ps1
```

## Usage (Live LID + Cloud STT)

```pwsh
python disconnected_language_detector.py `
  --audio "audio\samples\Arabic_english_mix_optimized.wav" `
  --languages en-US ar-SA `
  --lid-host ws://localhost:5003 `
  --segments live_segments.json `
  --output live_transcript.json `
  --timeout-sec 60 `
  --tail-overlap-ms 200 `
  --chunk-ms 40 `
  --flush-after-stop `
  --verbose
```

### Important Flags
| Flag | Meaning |
|------|---------|
| `--lid-host` | WebSocket host:port of LID container (NO path). If you enter `localhost:5003` we add `ws://`. |
| `--languages` | Candidate language codes for detection (BCP‑47). |
| `--timeout-sec` | Hard stop for LID phase (wall clock). 0 disables timeout. |
| `--min-segment-sec` | Only used for logging/flagging short spans; live mode still streams immediately. |
| `--chunk-ms` | Milliseconds per push chunk to cloud STT. Lower = more responsive; higher = fewer calls. |
| `--tail-overlap-ms` | Extra audio included after a detected end before stopping recognizer (reduces word clipping). |
| `--flush-after-stop` | Write `--output` after each language block completes (incremental). |
| `--verbose` | Debug logging (partials, detection events, internal timings). |

## Output Files

* `--segments` (e.g. `live_segments.json`): merged LID segments (audit reference). Structure:
```json
{
  "AudioFile": "audio\\samples\\Arabic_english_mix_optimized.wav",
  "SegmentCount": 5,
  "Segments": [
    {"Language": "en-US", "StartTimeHns": 0, "DurationHns": 55000000, "IsSkipped": false},
    {"Language": "ar-SA", "StartTimeHns": 55000000, "DurationHns": 30000000, "IsSkipped": false}
  ]
}
```

* `--output` (e.g. `live_transcript.json`):
```json
{
  "audio_file": "...",
  "segment_count": 2,
  "segments": [
    {"Language": "en-US", "StartTimeHns": 0, "DurationHns": 55000000, "IsSkipped": false},
    {"Language": "ar-SA", "StartTimeHns": 55000000, "DurationHns": 30000000, "IsSkipped": false}
  ],
  "recognized_results": [
    {"language": "en-US", "start_hns": 0, "end_hns": 55000000, "transcript": "Hello ..."},
    {"language": "ar-SA", "start_hns": 55000000, "end_hns": 85000000, "transcript": "مرحبا ..."}
  ],
  "full_transcript": "[en-US] Hello ... [ar-SA] مرحبا ..."
}
```

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Every segment shows `Canceled` | Bad key / region mismatch / network block | Verify key & region; attempt region mode instead of endpoint; test with `simple_cloud_test.py`. |
| No detection events | Wrong `--lid-host` or container not running | Confirm container logs; ensure no path appended; correct port. |
| Truncated words at switch | Overlap too small | Increase `--tail-overlap-ms` (e.g. 400). |
| Latency too high | Chunk too large | Reduce `--chunk-ms` (e.g. 20–30). |

## Related / Legacy Scripts

* `hybrid_transcribe.py` – offline segmented pass (and hybrid cloud/container support) if you need deterministic slice-based transcription.
* `fixed_transcribe_with_language.py` – earlier two‑phase version kept for comparison.
* `simple_cloud_test.py` – minimal single recognition sanity check (full file, one language) to isolate cloud issues.

## Roadmap / Ideas
* Optional fallback to container STT for air‑gapped scenarios.
* Confidence gating / smoothing for rapid code switches.
* Real‑time websocket output (JSON events) for UI integration.
* Speaker diarization integration layer.

## License
MIT (adjust as needed).

