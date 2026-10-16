from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
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

MODEL_URL = "https://github.com/blackshadowh4ck3r/kokoro-tts-api/releases/download/V1/model.onnx"
VOICES_URL = "https://github.com/blackshadowh4ck3r/kokoro-tts-api/releases/download/V1/voices.json"

MODEL_FILE = "model.onnx"
VOICES_FILE = "voices.json"

kokoro = None

@app.on_event("startup")
def load_model():
    global kokoro
    try:
        if not os.path.exists(MODEL_FILE):
            print("Downloading model.onnx from GitHub release...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
            print("Model downloaded successfully!")
        
        if not os.path.exists(VOICES_FILE):
            print("Downloading voices.json from GitHub release...")
            urllib.request.urlretrieve(VOICES_URL, VOICES_FILE)
            print("Voices file downloaded successfully!")

        from kokoro_onnx import Kokoro
        print("Initializing Kokoro Engine...")
        kokoro = Kokoro(MODEL_FILE, VOICES_FILE)
        print("Kokoro Engine Ready!")
    except Exception as e:
        print(f"CRITICAL ERROR during startup: {str(e)}")

@app.get("/")
def home():
    if kokoro is None:
        return {"status": "Server is running, but Kokoro engine failed to load. Please check Render logs."}
    return {"status": "Kokoro TTS API is Live and Working!"}

@app.get("/synthesize")
def synthesize(text: str, voice: str = "af_heart"):
    if kokoro is None:
        return {"error": "Kokoro engine is not initialized. Check server startup logs."}
    try:
        samples, sample_rate = kokoro.create(text, voice=voice, speed=1.0, lang="en-us")
        
        audio_io = io.BytesIO()
        sf.write(audio_io, samples, sample_rate, format='WAV')
        audio_io.seek(0)
        
        return Response(content=audio_io.read(), media_type="audio/wav")
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
