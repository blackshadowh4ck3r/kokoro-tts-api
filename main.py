from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from kokoro_onnx import Kokoro
from huggingface_hub import hf_hub_download
import soundfile as sf
import io
import os
import urllib.request
import shutil

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_URL = "https://github.com/blackshadowh4ck3r/kokoro-tts-api/releases/download/V1/model_quantized.onnx"
MODEL_FILE = "model_quantized.onnx"
VOICES_FILE = "voices.json"

# 1. Model ko aapke GitHub Release se download karein
if not os.path.exists(MODEL_FILE):
    print("Downloading model_quantized.onnx from GitHub release...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
    print("Model downloaded successfully!")

# 2. Voices file ko Hugging Face Hub se safely download karein (404 nahi aayega)
if not os.path.exists(VOICES_FILE):
    print("Downloading voices.json from Hugging Face Hub...")
    downloaded_voices_path = hf_hub_download(repo_id="onnx-community/Kokoro-82M-ONNX", filename="voices.json")
    shutil.copy(downloaded_voices_path, VOICES_FILE)
    print("Voices file downloaded successfully!")

# 3. Kokoro Engine Initialize karein
print("Initializing Kokoro Engine...")
kokoro = Kokoro(MODEL_FILE, VOICES_FILE)
print("Kokoro Engine Ready!")

@app.get("/")
def home():
    return {"status": "Kokoro TTS API is Live and Running!"}

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
