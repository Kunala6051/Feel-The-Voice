import whisper

def load_whisper_model(model_size="base"):
    return whisper.load_model(model_size)

def transcribe_audio(model, audio_path, language=None):

    result = model.transcribe(
        audio_path,
        task="transcribe",
        language=language,
        fp16=False
    )

    return {
        "detected_language": result["language"],
        "transcribed_text": result["text"]
    }

def translate_audio_to_english(model, audio_path):

    result = model.transcribe(
        audio_path,
        task="translate",
        fp16=False
    )

    return {
        "translated_text": result["text"]
    }