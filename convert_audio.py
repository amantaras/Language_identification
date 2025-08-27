import os
from pydub import AudioSegment
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def convert_mp3_to_wav(mp3_path, wav_path=None):
    """Convert MP3 file to WAV format suitable for Speech SDK"""
    if not os.path.exists(mp3_path):
        log.error(f"Source file not found: {mp3_path}")
        return None

    # If no wav_path specified, create one based on mp3_path
    if wav_path is None:
        wav_path = os.path.splitext(mp3_path)[0] + ".wav"

    log.info(f"Converting {mp3_path} to {wav_path}")

    try:
        # Load MP3 file
        audio = AudioSegment.from_mp3(mp3_path)

        # Convert to mono if stereo
        if audio.channels > 1:
            log.info("Converting stereo to mono")
            audio = audio.set_channels(1)

        # Set sample rate to 16kHz if needed
        if audio.frame_rate != 16000:
            log.info(f"Converting sample rate from {audio.frame_rate}Hz to 16000Hz")
            audio = audio.set_frame_rate(16000)

        # Export as WAV with PCM format
        audio.export(wav_path, format="wav")
        log.info(f"Successfully converted to {wav_path}")

        return wav_path

    except Exception as e:
        log.error(f"Error converting audio: {e}")
        return None


if __name__ == "__main__":
    # Convert the sample file
    mp3_file = "audio/samples/Scenario1_20250323_S1056_New Postpaid Activation1.mp3"
    wav_file = convert_mp3_to_wav(mp3_file)

    if wav_file:
        log.info(f"File converted: {wav_file}")
        log.info("Now try using this WAV file with the language detection code")
    else:
        log.error("Conversion failed")
