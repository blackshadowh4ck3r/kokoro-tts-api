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

# Aapke GitHub Release ke direct links
MODEL_URL = "https://github.com/blackshadowh4ck3r/kokoro-tts-api/releases/download/V1/model_quantized.onnx"
VOICES_URL = "https://github.com/blackshadowh4ck3r/kokoro-tts-api/releases/download/V1/af_heart.bin"

MODEL_FILE = "model_quantized.onnx"
VOICES_FILE = "voices.json"  # kokoro-onnx ke liye voices file ka format json hota hai

# Agar files server par nahi hain, toh direct link se download kar lo
if not os.path.exists(MODEL_FILE):
    print("Downloading model_quantized.onnx from GitHub release...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
    print("Model downloaded successfully!")

if not os.path.exists(VOICES_FILE):
    print("Downloading voices file from GitHub release...")
    urllib.request.urlretrieve(VOICES_URL, VOICES_FILE)
    print("Voices file downloaded successfully!")

# Initialize Kokoro Engine
print("Initializing Kokoro Engine...")
kokoro = Kokoro(MODEL_FILE, VOICES_FILE)
print("Kokoro Engine Ready!")

@app.get("/")
def home():
    return {"status": "Kokoro TTS API is Live using Direct GitHub Links!"}

@app.get("/synthesize")
def synthesize(text: str, voice: str = "af_heart"):
    try:
        samples, sample_rate = kokoro.create(text, voice=voice, speed=1.0, lang="en-us")
        
        audio_io = io.BytesIO()
        sf.write(audio_io, samples, sample_rate, format='WAV')
        audio_io.seek(0)
        
        return Response(content=audio_io.read(), media_type="audio/wav")
    except Exception as e:
        return {"error": str(e)}
