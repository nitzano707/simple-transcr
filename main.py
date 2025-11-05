import os
import requests
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Optional
import whisper
from pyannote.audio import Pipeline
import tempfile

# טוען מודל תמלול
whisper_model = whisper.load_model(os.getenv("WHISPER_MODEL", "large"))

# טוען מודל דיאריזציה מהוגינג פייס
HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization@2023.07",
    use_auth_token=HUGGINGFACE_TOKEN
)

# FastAPI
app = FastAPI()

class InputData(BaseModel):
    file_url: str
    language: Optional[str] = "he"
    diarize: Optional[bool] = False
    vad: Optional[bool] = False

class InferenceRequest(BaseModel):
    input: InputData

@app.post("/")
async def transcribe(request: InferenceRequest):
    url = request.input.file_url
    language = request.input.language
    do_diarize = request.input.diarize

    # הורדת קובץ
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
        response = requests.get(url)
        temp_file.write(response.content)
        audio_path = temp_file.name

    # תמלול
    result = whisper_model.transcribe(audio_path, language=language)
    text = result["text"]

    # זיהוי דוברים (אם נבחר)
    speakers = []
    if do_diarize:
        diarization = diarization_pipeline(audio_path)
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speakers.append({
                "speaker": speaker,
                "start": turn.start,
                "end": turn.end
            })

    return {
        "transcription": text,
        "speakers": speakers
    }
