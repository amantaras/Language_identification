# Example usage for the transcribe_with_language.py script

# Make sure your containers are running:
# - Language ID container on port 5003
# - English STT container on port 5000
# - Arabic STT container on port 5002

# Run this script with Python:

python transcribe_with_language.py \
  --audio "audio/samples/Scenario1_20250323_S1056_New Postpaid Activation1.mp3" \
  --languages en-US ar-SA \
  --lid-host "ws://localhost:5003" \
  --stt-map "en-US=ws://localhost:5000" "ar-SA=ws://localhost:5002" \
  --output "audio/output/transcript.json" \
  --verbose

# PowerShell version:

python transcribe_with_language.py `
  --audio "audio/samples/Scenario1_20250323_S1056_New Postpaid Activation1.mp3" `
  --languages en-US ar-SA `
  --lid-host "ws://localhost:5003" `
  --stt-map "en-US=ws://localhost:5000" "ar-SA=ws://localhost:5002" `
  --output "audio/output/transcript.json" `
  --verbose

# This will:
# 1. Detect language segments in the audio file
# 2. Save the segments to segments.json
# 3. Transcribe each segment using the appropriate STT container
# 4. Save the final transcript to audio/output/transcript.json

# The output will include:
# - Individual segment transcriptions with timestamps
# - A combined transcript with language markers: [en-US] English text [ar-SA] Arabic text
