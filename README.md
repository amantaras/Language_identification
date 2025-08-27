# Language Switch Aware Transcription

This project provides a reference implementation for processing an audio file / stream, detecting language changes (code-switching), and routing audio segments to different Azure Speech-to-Text container endpoints for transcription.

## Key Features
- Continuous multi-language identification using Azure Speech Language Detection container.
- Segment timeline construction (start/end offsets in HNS and seconds).
- Per-language buffering of audio samples for precise segment extraction.
- Dynamic transcription routing: each supported language maps to its own Azure Speech STT container endpoint (host + key).
- Unified output combining all hypotheses with timestamps and language tags.

## Quick Start

### 1. Clone & Install
```pwsh
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -e .
```

### 2. Prepare `.env`
Copy `.env.example` to `.env` and fill in:
```
SPEECH_BILLING_ENDPOINT=https://<your-speech-resource>.cognitiveservices.azure.com/
SPEECH_API_KEY=<your-key>
# (Optionally adjust ports/locales)
```

### 3. Run Containers (PowerShell script)
```pwsh
./scripts/run-containers.ps1 -EnvFile ./.env
# (Add -Interactive to stream logs, -ForceRecreate to remove existing)
```
This starts:
- Language Identification on `LID_PORT` (default 5003)
- English STT on `EN_PORT` (default 5004)
- Arabic STT on `AR_PORT` (default 5005)

Stop containers:
```pwsh
./scripts/stop-containers.ps1
```

### 4. Run Detection + Transcription
```pwsh
lang-switch-stt full --audio enar.wav `
  --languages en-US --languages ar-SA `
  --lid-host ws://localhost:5003 `
  --map en-US=ws://localhost:5004 --map ar-SA=ws://localhost:5005 `
  --key $env:SPEECH_API_KEY --billing $env:SPEECH_BILLING_ENDPOINT `
  --min-segment-sec 0.4 `
  --timeout-sec 120 `
  --verbose `
  --out transcript.json
```

## Architecture Overview
1. **Language Detection Pass**: Uses container endpoint to receive recognized events with offsets + detected language.
2. **Segmentation Engine**: Builds segments when language changes.
3. **Audio Slicing**: Loads original audio, slices by offsets.
4. **Per-Language Transcription**: Each segment routed to its locale-specific STT container.
5. **Merge**: Output JSON with ordered segments.

## Output Format
`transcript.json`:
```json
{
  "audio": "enar.wav",
  "segments": [
    {"id": 0, "language": "en-US", "start_sec": 0.00, "end_sec": 3.21, "text": "Hello ..."},
    {"id": 1, "language": "ar-SA", "start_sec": 3.21, "end_sec": 6.88, "text": "مرحبا ..."}
  ]
}
```

## Scripts
- `scripts/run-containers.ps1`: Launch all required containers using `.env` values.
- `scripts/stop-containers.ps1`: Stop/remove the containers.

## Advanced Options
CLI flags added for robustness:
- `--timeout-sec`: Abort detection if it exceeds this wall-clock time.
- `--min-segment-sec`: Drop very short language blips below this duration.
- `--verbose`: Enable debug logging (prints segment start/end and language switches).

## Next Enhancements
- Confidence / threshold based switching.
- Real-time streaming (single pass) pipeline.
- Parallel transcription.
- Silence-aware segmentation.

## License
MIT (adjust as needed)
