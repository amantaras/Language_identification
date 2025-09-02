"""
Modified language detection implementation for Azure Speech containers
- Uses synchronous recognition instead of continuous for more reliable detection
- Handles cancellation details properly in SDK v1.45.0
- Implements proper error reporting for container debugging
"""

import time
import json
import logging
import os
from typing import List, Optional
import azure.cognitiveservices.speech as speechsdk
from .segmentation import SegmentBuilder, HNS_PER_SECOND


class LanguageDetectionResult:
    def __init__(self, segments_json_path: str):
        self.segments_json_path = segments_json_path


def detect_language_sync(
    audio_file: str,
    lid_host: str,
    languages: List[str],
    logger: Optional[logging.Logger] = None,
    connection_timeout: float = 5.0,  # Add timeout parameter
) -> Optional[str]:
    """
    Perform one-shot synchronous language detection (simpler than continuous).
    Returns the detected language or None if detection failed.
    """
    log = logger or logging.getLogger(__name__)

    # Ensure the host has the ws:// prefix required for container communication
    if not lid_host.startswith(("ws://", "wss://")):
        lid_host = f"ws://{lid_host}"
        log.debug(f"Added ws:// prefix. Using host URL: {lid_host}")

    # Check if container is reachable
    host_without_prefix = lid_host.replace("ws://", "").replace("wss://", "")
    host_parts = host_without_prefix.split(":")
    host_ip = host_parts[0]
    host_port = int(host_parts[1]) if len(host_parts) > 1 else 5000

    log.info(f"Checking container connectivity at {host_ip}:{host_port}...")

    import socket

    try:
        # Simple socket test to check if port is open
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(connection_timeout)
        result = sock.connect_ex((host_ip, host_port))
        sock.close()

        if result != 0:
            log.error(
                f"Container at {host_ip}:{host_port} is not reachable. Error code: {result}"
            )
            log.error("Please ensure the language detection container is running.")
            return None
        else:
            log.debug(f"Container at {host_ip}:{host_port} is reachable.")
    except Exception as e:
        log.error(f"Error checking container connectivity: {e}")
        # Continue anyway - the SDK will also check connectivity

    # Create the speech config with the direct host method
    try:
        speech_config = speechsdk.SpeechConfig(host=lid_host)
        log.debug(f"Successfully created SpeechConfig with host: {lid_host}")
    except Exception as e:
        log.error(f"Failed to create SpeechConfig: {e}")
        return None

    # Set Continuous mode for detecting language changes throughout the audio
    try:
        # Try using the property from the newer SDK versions
        if hasattr(speechsdk.PropertyId, "SpeechServiceConnection_LanguageIdMode"):
            speech_config.set_property(
                property_id=speechsdk.PropertyId.SpeechServiceConnection_LanguageIdMode,
                value="Continuous",
            )
            log.debug("Set language detection mode to Continuous")
        else:
            # For older SDK versions, we can try using the string directly
            speech_config.set_property(
                property_id="SpeechServiceConnection_LanguageIdMode",
                value="Continuous",
            )
            log.debug(
                "Set language detection mode to Continuous using string property ID"
            )
    except Exception as e:
        log.error(f"Failed to set language detection mode: {e}")
        # Continue anyway - this might work without setting the mode explicitly

    # Create audio config
    try:
        audio_config = speechsdk.audio.AudioConfig(filename=audio_file)
        log.debug(f"Created AudioConfig with file: {audio_file}")
    except Exception as e:
        log.error(f"Failed to create AudioConfig: {e}")
        return None

    # Create language config
    try:
        auto_detect = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
            languages=languages
        )
        log.debug(f"Created AutoDetectSourceLanguageConfig with languages: {languages}")
    except Exception as e:
        log.error(f"Failed to create AutoDetectSourceLanguageConfig: {e}")
        return None

    # Create recognizer and run recognition
    try:
        recognizer = speechsdk.SourceLanguageRecognizer(
            speech_config=speech_config,
            auto_detect_source_language_config=auto_detect,
            audio_config=audio_config,
        )
        log.debug("Created SourceLanguageRecognizer successfully")

        # Use synchronous recognition
        log.info("Starting synchronous language recognition...")
        result = recognizer.recognize_once()

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            detected_lang = result.properties.get(
                speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult
            )
            log.info(f"DETECTED LANGUAGE: {detected_lang}")
            log.info(f"Recognized text: {result.text}")
            return detected_lang
        else:
            log.warning(f"Recognition failed with reason: {result.reason}")

            # Handle cancellation in a way that works with different SDK versions
            if result.reason == speechsdk.ResultReason.Canceled:
                try:
                    if hasattr(result, "cancellation_details"):
                        cancellation = result.cancellation_details
                        log.error(f"CANCELED: Reason={cancellation.reason}")

                        # Check what attributes are available in this SDK version
                        if hasattr(cancellation, "error_code"):
                            log.error(f"CANCELED: ErrorCode={cancellation.error_code}")
                        if hasattr(cancellation, "error_details"):
                            log.error(
                                f"CANCELED: ErrorDetails={cancellation.error_details}"
                            )

                            # Check for common container WebSocket issues
                            error_details = (
                                cancellation.error_details
                                if hasattr(cancellation, "error_details")
                                else ""
                            )
                            if (
                                "WS_OPEN_ERROR" in error_details
                                or "Connection failed" in error_details
                            ):
                                log.error(
                                    "CONTAINER CONNECTION ERROR: The WebSocket connection to the container failed."
                                )
                                log.error("Possible causes:")
                                log.error(
                                    "1. The container might be running but not fully initialized"
                                )
                                log.error("2. The container might be misconfigured")
                                log.error(
                                    "3. The container API might not match the SDK version"
                                )
                                log.error(
                                    "Try restarting the container with: ./scripts/run-containers-simple.ps1"
                                )

                        # Sometimes the error details are in a different attribute
                        if hasattr(cancellation, "reason_details"):
                            log.error(
                                f"CANCELED: ReasonDetails={cancellation.reason_details}"
                            )
                    else:
                        # For older SDK versions
                        log.error("CANCELED: No detailed information available")
                except Exception as ex:
                    log.error(f"Error accessing cancellation details: {ex}")

            return None
    except Exception as e:
        log.error(f"Recognition failed with error: {e}")
        return None


def detect_languages(
    audio_file: str,
    lid_host: str,
    languages: List[str],
    out_segments: str,
    timeout_sec: Optional[float] = None,
    min_segment_sec: float = 0.0,
    logger: Optional[logging.Logger] = None,
    connection_timeout: float = 5.0,  # Add timeout parameter
    _force_continuous: bool = False,  # New parameter to force continuous mode
) -> LanguageDetectionResult:
    """Perform language identification and emit segments JSON.

    This function will attempt both synchronous and continuous methods to maximize chances of success.
    If container connectivity issues are detected, it will provide diagnostic information and
    fall back to using the first language in the list as a placeholder.

    timeout_sec: Optional overall timeout to abort recognition loop.
    min_segment_sec: Drop segments shorter than this duration.
    connection_timeout: Timeout for container connectivity check in seconds.
    """
    log = logger or logging.getLogger(__name__)

    log.info("Starting language identification...")

    # First try the simpler synchronous method, unless forced to use continuous mode
    detected_language = None
    if not _force_continuous:
        detected_language = detect_language_sync(
            audio_file,
            lid_host,
            languages,
            logger=log,
            connection_timeout=connection_timeout,
        )

    if detected_language and not _force_continuous:
        log.info(
            f"Successfully detected language with sync method: {detected_language}"
        )

        # Create a simple segment with the detected language for the entire file
        builder = SegmentBuilder(
            min_duration_hns=int(min_segment_sec * HNS_PER_SECOND), logger=log
        )

        # Add a single segment with the detected language
        # Start at 0, end will be determined when reading the file
        end_time_hns = (
            10000000  # Default 1 second duration if we can't determine real length
        )
        builder.on_detection(detected_language, 0, end_time_hns)
        builder.finalize(final_end_hns=end_time_hns)

        # Write the segments to the output file
        builder.to_json(out_segments, audio_file)
        log.info(f"Wrote simple segment to {out_segments}")

        return LanguageDetectionResult(out_segments)

    # If sync method failed, try the original continuous method (from original implementation)
    log.warning(
        "Synchronous language detection failed, falling back to continuous method..."
    )

    # Create speech config
    if not lid_host.startswith(("ws://", "wss://")):
        lid_host = f"ws://{lid_host}"

    speech_config = speechsdk.SpeechConfig(host=lid_host)

    # Set the required continuous language detection mode
    try:
        # Try using the property from the newer SDK versions
        if hasattr(speechsdk.PropertyId, "SpeechServiceConnection_LanguageIdMode"):
            speech_config.set_property(
                property_id=speechsdk.PropertyId.SpeechServiceConnection_LanguageIdMode,
                value="Continuous",
            )
            log.debug("Set language detection mode to Continuous")
        else:
            # For older SDK versions, we can try using the string directly
            speech_config.set_property(
                property_id="SpeechServiceConnection_LanguageIdMode",
                value="Continuous",
            )
            log.debug(
                "Set language detection mode to Continuous using string property ID"
            )
    except Exception as e:
        log.error(f"Failed to set language detection mode: {e}")
        # Continue anyway - this might work without setting the mode explicitly

    audio_config = speechsdk.audio.AudioConfig(filename=audio_file)

    # Microsoft docs confirm: Use SourceLanguageRecognizer for container language ID
    auto_detect = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
        languages=languages
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
    success = False

    def recognized(evt: speechsdk.SpeechRecognitionEventArgs):
        nonlocal last_end, success
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            detected = evt.result.properties.get(
                speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult
            )
            if detected:
                success = True
                json_result = evt.result.properties.get(
                    speechsdk.PropertyId.SpeechServiceResponse_JsonResult
                )
                if json_result:
                    detail = json.loads(json_result)
                    start = detail.get("Offset", 0)
                    duration = detail.get("Duration", 0)
                    end_offset = start + duration if duration >= 0 else start
                    log.info(
                        f"LID event detected: {detected} start={start/HNS_PER_SECOND:.2f}s dur={(duration/HNS_PER_SECOND) if duration >= 0 else 'unknown'}s"
                    )
                    log.debug(
                        f"LID event detailed: lang={detected} start={start} dur={duration}"
                    )
                    builder.on_detection(detected, start, end_offset)
                    last_end = max(last_end, end_offset)

    def canceled(evt: speechsdk.SpeechRecognitionCanceledEventArgs):
        nonlocal done
        try:
            error_info = []
            # Check attributes in a version-agnostic way
            if hasattr(evt, "reason"):
                error_info.append(f"reason={evt.reason}")
            if hasattr(evt, "error_code"):
                error_info.append(f"error_code={evt.error_code}")
            if hasattr(evt, "error_details"):
                error_info.append(f"error_details={evt.error_details}")
            if hasattr(evt, "reason_details"):
                error_info.append(f"reason_details={evt.reason_details}")

            log.error(f"Canceled: {' '.join(error_info)}")
        except Exception as e:
            log.error(f"Error accessing cancellation details: {e}")
        done = True

    def stop_cb(evt):
        nonlocal done
        log.debug("Recognition stopped/canceled event received")
        done = True

    recognizer.recognized.connect(recognized)
    recognizer.session_stopped.connect(stop_cb)
    recognizer.canceled.connect(canceled)

    log.info("Starting continuous language identification")
    recognizer.start_continuous_recognition()
    start_time = time.time()
    while not done:
        if timeout_sec is not None and (time.time() - start_time) > timeout_sec:
            log.warning("Timeout reached; stopping recognition")
            break
        if success:
            # If we've already got a successful detection, don't wait too long
            if (time.time() - start_time) > 5:
                log.info("Successfully detected language; stopping early")
                break
        time.sleep(0.5)

    recognizer.stop_continuous_recognition()

    if not success:
        # If no segments were detected, create a fallback segment with the first language
        log.warning(
            f"No language segments detected, using fallback language: {languages[0]}"
        )
        builder.on_detection(languages[0], 0, 10000000)  # Default 1 second duration

    builder.finalize(final_end_hns=last_end)

    # Always use the original audio file path for the JSON output
    # This ensures that downstream processes refer to the original file
    builder.to_json(out_segments, audio_file)
    log.info(f"Wrote segments to {out_segments}")

    return LanguageDetectionResult(out_segments)
