import time
import json
import logging
from typing import List, Optional
import azure.cognitiveservices.speech as speechsdk
from .segmentation import SegmentBuilder, HNS_PER_SECOND

# Note: This module performs a pass over the audio file to detect language switches.


class LanguageDetectionResult:
    def __init__(self, segments_json_path: str):
        self.segments_json_path = segments_json_path


def detect_languages(
    audio_file: str,
    lid_host: str,
    languages: List[str],
    out_segments: str,
    timeout_sec: Optional[float] = None,
    min_segment_sec: float = 0.0,
    logger: Optional[logging.Logger] = None,
) -> LanguageDetectionResult:
    """Perform continuous language identification and emit segments JSON.

    timeout_sec: Optional overall timeout to abort recognition loop.
    min_segment_sec: Drop segments shorter than this duration.
    """
    log = logger or logging.getLogger(__name__)
    speech_config = speechsdk.SpeechConfig(host=lid_host)
    audio_config = speechsdk.audio.AudioConfig(filename=audio_file)

    auto_detect = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
        languages=languages
    )
    speech_config.set_property(
        property_id=speechsdk.PropertyId.SpeechServiceConnection_LanguageIdMode,
        value="Continuous",
    )

    recognizer = speechsdk.SourceLanguageRecognizer(
        speech_config=speech_config,
        auto_detect_source_language_config=auto_detect,
        audio_config=audio_config,
    )

    min_hns = int(min_segment_sec * HNS_PER_SECOND)
    builder = SegmentBuilder(min_duration_hns=min_hns, logger=log)
    done = False
    last_end = 0

    def recognized(evt: speechsdk.SpeechRecognitionEventArgs):
        nonlocal last_end
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            detected = evt.result.properties.get(
                speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult
            )
            if detected:
                json_result = evt.result.properties.get(
                    speechsdk.PropertyId.SpeechServiceResponse_JsonResult
                )
                if json_result:
                    detail = json.loads(json_result)
                    start = detail.get("Offset", 0)
                    duration = detail.get("Duration", 0)
                    end_offset = start + duration if duration >= 0 else start
                    log.debug(f"LID event lang={detected} start={start} dur={duration}")
                    builder.on_detection(detected, start, end_offset)
                    last_end = max(last_end, end_offset)

    def stop_cb(evt):
        nonlocal done
        log.debug("Recognition stopped/canceled event received")
        done = True

    recognizer.recognized.connect(recognized)
    recognizer.session_stopped.connect(stop_cb)
    recognizer.canceled.connect(stop_cb)

    log.info("Starting continuous language identification")
    recognizer.start_continuous_recognition()
    start_time = time.time()
    while not done:
        if timeout_sec is not None and (time.time() - start_time) > timeout_sec:
            log.warning("Timeout reached; stopping recognition")
            break
        time.sleep(0.5)
    recognizer.stop_continuous_recognition()

    builder.finalize(final_end_hns=last_end)
    builder.to_json(out_segments, audio_file)
    log.info(f"Wrote segments to {out_segments}")
    return LanguageDetectionResult(out_segments)
