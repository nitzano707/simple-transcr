import os
import runpod
import requests
import tempfile
import whisper
from pyannote.audio import Pipeline

# טעינת מודלים בזמן אתחול (cold start)
print("Loading Whisper model...")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
whisper_model = whisper.load_model(WHISPER_MODEL)

print("Loading Diarization pipeline...")
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_TOKEN
    )
else:
    print("Warning: HF_TOKEN not set - diarization will fail")
    diarization_pipeline = None

def handler(event):
    """
    Handler לפונקציה ב-Runpod Serverless
    
    Input format:
    {
        "input": {
            "file_url": "https://...",
            "language": "he",  # אופציונלי
            "diarize": false   # אופציונלי
        }
    }
    """
    try:
        # קבלת פרמטרים
        input_data = event.get("input", {})
        file_url = input_data.get("file_url")
        language = input_data.get("language", "he")
        do_diarize = input_data.get("diarize", False)
        
        if not file_url:
            return {"error": "file_url is required"}
        
        # הורדת הקובץ
        print(f"Downloading audio from: {file_url}")
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            response = requests.get(file_url, timeout=300)
            response.raise_for_status()
            tmp.write(response.content)
            tmp.flush()
            audio_path = tmp.name
        
        # תמלול
        print("Transcribing audio...")
        result = whisper_model.transcribe(
            audio_path, 
            language=language,
            fp16=False  # CPU mode
        )
        transcription = result["text"]
        
        # זיהוי דוברים
        speakers = []
        if do_diarize:
            if not diarization_pipeline:
                return {
                    "error": "Diarization requested but HF_TOKEN not configured"
                }
            
            print("Running speaker diarization...")
            diarization = diarization_pipeline(audio_path)
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speakers.append({
                    "speaker": speaker,
                    "start": round(turn.start, 2),
                    "end": round(turn.end, 2)
                })
        
        # מחיקת קובץ זמני
        os.unlink(audio_path)
        
        # החזרת תוצאות
        return {
            "transcription": transcription,
            "speakers": speakers,
            "language": language
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return {"error": str(e)}

# הרצת Runpod Serverless
if __name__ == "__main__":
    print("Starting Runpod Serverless worker...")
    runpod.serverless.start({"handler": handler})
