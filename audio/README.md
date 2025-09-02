# Audio Files Directory

This directory contains audio files for testing the language identification system.

## Structure

- `samples/` - Small test audio files for development and testing
- `input/` - Audio files to be processed
- `output/` - Generated results (segments.json, transcriptions, etc.)

## Supported Formats

The tool supports various audio formats:
- WAV (recommended for best quality)
- MP3
- M4A
- FLAC

## Usage Examples

```bash
# Process an audio file
lang-switch-stt audio/input/meeting.wav en,ar ws://localhost:5003 audio/output/segments.json

# With additional options
lang-switch-stt audio/samples/test.wav en,ar ws://localhost:5003 audio/output/test-segments.json --timeout-sec 30 --min-segment-sec 2.0 --verbose
```

## Test Files

Place your test audio files in the appropriate subdirectory:
- Short samples (< 1 minute): `samples/`
- Longer files for processing: `input/`
