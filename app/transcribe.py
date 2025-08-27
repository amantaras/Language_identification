import json
import tempfile
import os
from typing import Dict, List
import azure.cognitiveservices.speech as speechsdk
import soundfile as sf
from .segmentation import HNS_PER_SECOND

class TranscriptionSegment:
    def __init__(self, seg_id: int, language: str, start_sec: float, end_sec: float, text: str):
        self.id = seg_id
        self.language = language
        self.start_sec = start_sec
        self.end_sec = end_sec
        self.text = text

    def to_dict(self):
        return {
            "id": self.id,
            "language": self.language,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "text": self.text
        }


def _transcribe_chunk(samples, sample_rate: int, host: str, language: str, key: str, billing: str) -> str:
    """Transcribe a short PCM segment.

    Uses a temporary WAV file + file-based AudioConfig for simplicity & correctness.
    (Previous in-memory PushAudioInputStream approach incorrectly fed a WAV container
    header over a stream that expects raw PCM frames.)
    """
    if samples.size == 0:
        return ""

    # Write to a temp wav file (SDK handles reading & format negotiation)
    with tempfile.TemporaryDirectory() as td:
        tmp_path = os.path.join(td, "chunk.wav")
        sf.write(tmp_path, samples, sample_rate, format="WAV")

        speech_config = speechsdk.SpeechConfig(host=host)
        if language:
            speech_config.speech_recognition_language = language
        # Central billing / key (optional for containers; keep if provided)
        if billing:
            speech_config.set_property(speechsdk.PropertyId.SpeechServiceConnection_Endpoint, billing)
        if key:
            speech_config.set_property(speechsdk.PropertyId.SpeechServiceConnection_Key, key)

        audio_config = speechsdk.audio.AudioConfig(filename=tmp_path)
        recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
        result = recognizer.recognize_once()
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return result.text
        return ""


def transcribe_segments(audio_file: str, segments_json: str, language_host_map: Dict[str, str], key: str, billing: str, out_path: str):
    with open(segments_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    segments = data.get('segments', [])
    # Load full audio
    audio, sr = sf.read(audio_file)
    out_segments: List[TranscriptionSegment] = []
    for seg in segments:
        lang = seg['language']
        host = language_host_map.get(lang)
        if not host:
            continue
        start_sample = int((seg['start_hns'] / HNS_PER_SECOND) * sr)
        end_sample = int((seg['end_hns'] / HNS_PER_SECOND) * sr)
        # Guard indexes
        if end_sample <= start_sample:
            continue
        chunk = audio[start_sample:end_sample]
        text = _transcribe_chunk(chunk, sample_rate=sr, host=host, language=lang, key=key, billing=billing)
        out_segments.append(TranscriptionSegment(seg['id'], lang, seg['start_sec'], seg['end_sec'], text))

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            "audio": audio_file,
            "segments": [s.to_dict() for s in out_segments]
        }, f, ensure_ascii=False, indent=2)
