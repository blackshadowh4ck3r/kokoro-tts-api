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

# Sahi URLs: Model apka GitHub release wala hai, aur voices official JSON file hai
MODEL_URL = "https://github.com/blackshadowh4ck3r/kokoro-tts-api/releases/download/V1/model_quantized.onnx"
VOICES_URL = "https://huggingface.co/onnx-community/Kokoro-82M-ONNX/raw/main/voices.json"

MODEL_FILE = "model_quantized.onnx"
VOICES_FILE = "voices.json"

# Files download karna agar server par nahi hain
if not os.path.exists(MODEL_FILE):
    print("Downloading model_quantized.onnx...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
    print("Model downloaded!")

if not os.path.exists(VOICES_FILE):
    print("Downloading voices.json...")
    urllib.request.urlretrieve(VOICES_URL, VOICES_FILE)
    print("Voices file downloaded!")

# Initialize Kokoro Engine safely
try:
    print("Initializing Kokoro Engine...")
    kokoro = Kokoro(MODEL_FILE, VOICES_FILE)
    print("Kokoro Engine Initialized Successfully!")
except Exception as e:
    print(f"Error initializing Kokoro: {e}")

@app.get("/")
def home():
    return {"status": "Kokoro TTS API is Live and Working!"}

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
