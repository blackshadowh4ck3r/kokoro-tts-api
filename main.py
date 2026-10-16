from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from kokoro_onnx import Kokoro
from huggingface_hub import hf_hub_download
import soundfile as sf
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Downloading Kokoro ONNX model and voices from Hugging Face...")

# Sahi ONNX repository aur filenames
model_path = hf_hub_download(repo_id="onnx-community/Kokoro-82M-ONNX", filename="kokoro-v1.0.onnx")
voices_path = hf_hub_download(repo_id="onnx-community/Kokoro-82M-ONNX", filename="voices.json")

# Initialize Kokoro Engine
kokoro = Kokoro(model_path, voices_path)
print("Kokoro Engine initialized successfully!")

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
