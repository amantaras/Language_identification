import json
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


def _transcribe_chunk(wav_bytes: bytes, host: str, language: str, key: str, billing: str) -> str:
    speech_config = speechsdk.SpeechConfig(host=host)
    # Provide central billing + key for accounting
    if billing:
        speech_config.set_property(speechsdk.PropertyId.SpeechServiceConnection_Endpoint, billing)
    if key:
        speech_config.set_property(speechsdk.PropertyId.SpeechServiceConnection_Key, key)
    if language:
        speech_config.speech_recognition_language = language
    # Use PullAudioInputStream from bytes
    stream = speechsdk.audio.PushAudioInputStream()
    audio_config = speechsdk.audio.AudioConfig(stream=stream)
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    # Push audio then close
    stream.write(wav_bytes)
    stream.close()
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
        chunk = audio[start_sample:end_sample]
        # Write temp wav in memory
        import io, soundfile as sf2
        buf = io.BytesIO()
        sf2.write(buf, chunk, sr, format='WAV')
        wav_bytes = buf.getvalue()
        text = _transcribe_chunk(wav_bytes, host=host, language=lang, key=key, billing=billing)
        out_segments.append(TranscriptionSegment(seg['id'], lang, seg['start_sec'], seg['end_sec'], text))

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            "audio": audio_file,
            "segments": [s.to_dict() for s in out_segments]
        }, f, ensure_ascii=False, indent=2)
