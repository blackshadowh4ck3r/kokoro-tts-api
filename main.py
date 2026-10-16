from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from kokoro_onnx import Kokoro
import soundfile as sf
import io
import os
import urllib.request

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Automatic Model & Voices Downloader
MODEL_FILE = "kokoro-v1.0.onnx"
VOICES_FILE = "voices.json"

MODEL_URL = "https://github.com/thebloke/kokoro-onnx-models/releases/download/v1.0/kokoro-v1.0.onnx"
VOICES_URL = "https://github.com/thebloke/kokoro-onnx-models/releases/download/v1.0/voices.json"

def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
        print(f"{filename} downloaded successfully!")

# Auto download required files on server start
download_file(MODEL_URL, MODEL_FILE)
download_file(VOICES_URL, VOICES_FILE)

# Initialize Kokoro Engine
kokoro = Kokoro(MODEL_FILE, VOICES_FILE)

@app.get("/")
def home():
    return {"status": "Kokoro ONNX TTS Engine is Ready & Live!"}

@app.get("/synthesize")
def synthesize(text: str, voice: str = "af_bella"):
    try:
        samples, sample_rate = kokoro.create(text, voice=voice, speed=1.0, lang="en-us")
        
        audio_io = io.BytesIO()
        sf.write(audio_io, samples, sample_rate, format='WAV')
        audio_io.seek(0)
        
        return Response(content=audio_io.read(), media_type="audio/wav")
    except Exception as e:
        return {"error": str(e)}
