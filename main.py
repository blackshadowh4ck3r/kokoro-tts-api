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

print(10 * "-" + " DOWNLOADING KOKORO MODEL " + 10 * "-")
# Hugging Face se model automatically download hoga (render par bina kisi limit ke)
model_path = hf_hub_download(repo_id="onnx-community/Kokoro-82M-ONNX", filename="model.onnx")
voices_path = hf_hub_download(repo_id="onnx-community/Kokoro-82M-ONNX", filename="voices.json")

kokoro = Kokoro(model_path, voices_path)
print("Kokoro Engine Ready!")

@app.get("/")
def home():
    return {"status": "Server is live and ready!"}

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
