#!/usr/bin/env python
"""
Enhanced audio conversion utility with precise parameters for Speech SDK
"""
import os
import sys
import logging
from pydub import AudioSegment

# Set up logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def get_audio_info(file_path):
    """Get detailed information about an audio file"""
    if not os.path.exists(file_path):
        log.error(f"File not found: {file_path}")
        return None

    try:
        audio = AudioSegment.from_file(file_path)
        info = {
            "channels": audio.channels,
            "sample_width": audio.sample_width,
            "frame_rate": audio.frame_rate,
            "frame_width": audio.frame_width,
            "duration_seconds": len(audio) / 1000.0,
            "file_size_bytes": os.path.getsize(file_path),
        }
        return info
    except Exception as e:
        log.error(f"Error getting audio info: {e}")
        return None


def convert_audio_for_speech_sdk(input_path, output_path=None, sample_rate=16000):
    """
    Convert audio to the optimal format for Speech SDK:
    - WAV format
    - PCM encoding
    - 16-bit sample width
    - Mono channel
    - 16kHz sample rate
    """
    if not os.path.exists(input_path):
        log.error(f"Input file not found: {input_path}")
        return None

    # If no output path specified, create one with a suffix
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_optimized.wav"

    log.info(f"Converting {input_path} to {output_path}")

    try:
        # Load audio file
        audio = AudioSegment.from_file(input_path)

        # Get original properties
        log.info(
            f"Original audio: {len(audio)/1000:.2f}s, {audio.channels} channels, {audio.frame_rate}Hz, {audio.sample_width*8} bits"
        )

        # Convert to mono if stereo
        if audio.channels > 1:
            log.info("Converting stereo to mono")
            audio = audio.set_channels(1)

        # Set sample rate to specified value (default 16kHz)
        if audio.frame_rate != sample_rate:
            log.info(
                f"Converting sample rate from {audio.frame_rate}Hz to {sample_rate}Hz"
            )
            audio = audio.set_frame_rate(sample_rate)

        # Ensure 16-bit PCM
        if audio.sample_width != 2:  # 2 bytes = 16 bits
            log.info(f"Converting bit depth from {audio.sample_width*8} to 16 bits")
            audio = audio.set_sample_width(2)

        # Normalize audio to -3dB
        log.info("Normalizing audio level")
        audio = audio.normalize(headroom=-3.0)

        # Export as WAV with PCM format
        log.info("Exporting as WAV with PCM format")
        audio.export(output_path, format="wav", parameters=["-acodec", "pcm_s16le"])

        # Check the output file
        output_info = get_audio_info(output_path)
        if output_info:
            log.info(
                f"Converted audio: {output_info['duration_seconds']:.2f}s, "
                + f"{output_info['channels']} channels, {output_info['frame_rate']}Hz, "
                + f"{output_info['sample_width']*8} bits"
            )

        return output_path

    except Exception as e:
        log.error(f"Error converting audio: {e}")
        return None


if __name__ == "__main__":
    # Process command line arguments or use default
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = "audio/samples/Arabic_english_mix.wav"

    # Get info about the file
    log.info(f"Checking audio file: {input_file}")
    info = get_audio_info(input_file)
    if info:
        log.info(
            f"Audio info: {info['duration_seconds']:.2f}s, "
            + f"{info['channels']} channels, {info['frame_rate']}Hz, "
            + f"{info['sample_width']*8} bits, {info['file_size_bytes']/1024:.2f} KB"
        )

    # Convert the file with optimal parameters
    output_file = convert_audio_for_speech_sdk(input_file)

    if output_file:
        log.info(f"Successfully converted to: {output_file}")
        log.info("Now try this optimized file with the language detection code")
    else:
        log.error("Conversion failed")
