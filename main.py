import os
import requests
import tempfile
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import whisper
from pyannote.audio import Pipeline

# טען מודל Whisper
whisper_model = whisper.load_model(os.getenv("WHISPER_MODEL", "large"))

# טען מודל דיאריזציה של pyannote עם טוקן מהסביבה
HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization@2023.07",
    use_auth_token=HUGGINGFACE_TOKEN
)

# FastAPI setup
app = FastAPI()

class InputData(BaseModel):
    file_url: str
    language: Optional[str] = "he"
    diarize: Optional[bool] = False
    vad: Optional[bool] = False  # עדיין לא בשימוש

class InferenceRequest(BaseModel):
    input: InputData

@app.post("/")
async def transcribe(request: InferenceRequest):
    url = request.input.file_url
    language = request.input.language
    do_diarize = request.input.diarize

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        response = requests.get(url)
        tmp.write(response.content)
        tmp.flush()
        audio_path = tmp.name

    # תמלול
    result = whisper_model.transcribe(audio_path, language=language)
    transcription = result["text"]

    # זיהוי דוברים
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
        "transcription": transcription,
        "speakers": speakers
    }
