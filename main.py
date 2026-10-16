from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from kokoro import KPipeline
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

pipeline = KPipeline(lang_code='a') 

@app.get("/")
def home():
    return {"status": "Kokoro TTS API is running free on Render!"}

@app.get("/synthesize")
def synthesize(text: str, voice: str = "af_bella"):
    try:
        generator = pipeline(text, voice=voice, speed=1.0, split_pattern=r'\n+')
        for i, (gs, ps, audio) in enumerate(generator):
            audio_io = io.BytesIO()
            sf.write(audio_io, audio, 24000, format='WAV')
            audio_io.seek(0)
            return Response(content=audio_io.read(), media_type="audio/wav")
        return {"error": "Failed to generate audio"}
    except Exception as e:
        return {"error": str(e)}
      
