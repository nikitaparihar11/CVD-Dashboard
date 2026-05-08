# ===============================
# VOICE ENGINE (LIGHT VERSION)
# ===============================
import speech_recognition as sr
import tempfile
import os

def transcribe_audio(audio_bytes):
    try:
        recognizer = sr.Recognizer()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_bytes)
            path = f.name

        with sr.AudioFile(path) as source:
            audio = recognizer.record(source)

        text = recognizer.recognize_google(audio)

        os.remove(path)

        return text

    except Exception as e:
        print("Speech error:", e)
        return "" 