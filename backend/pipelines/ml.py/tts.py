import sys
from TTS.api import TTS


text = sys.argv[1]
speaker_wav = sys.argv[2]
language = sys.argv[3]
output_path = sys.argv[4]

tts = TTS(
    model_name="tts_models/multilingual/multi-dataset/xtts_v2"
)


tts.tts_to_file(
    text=text,
    speaker_wav=speaker_wav,
    language=language,
    file_path=output_path
)

print(f"Generated file: {output_path}")