import os
import runpod
import requests
import tempfile
import whisper
from pyannote.audio import Pipeline

# בדיקת גרסת NumPy
import numpy as np
print(f"NumPy version: {np.__version__}")
if not np.__version__.startswith('1.'):
    raise RuntimeError(f"Wrong NumPy version! Got {np.__version__}, need 1.x")

# טעינת מודלים
print("Loading Whisper model...")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
whisper_model = whisper.load_model(WHISPER_MODEL)
print(f"✓ Whisper model '{WHISPER_MODEL}' loaded successfully")

print("Loading Diarization pipeline...")
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    try:
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HF_TOKEN
        )
        print("✓ Diarization pipeline loaded successfully")
    except Exception as e:
        print(f"⚠ Warning: Could not load diarization pipeline: {e}")
        diarization_pipeline = None
else:
    print("⚠ Warning: HF_TOKEN not set - diarization disabled")
    diarization_pipeline = None

def handler(event):
    """
    Handler function for Runpod Serverless
    
    Expected input:
    {
        "input": {
            "file_url": "https://example.com/audio.mp3",
            "language": "he",  # optional, default: "he"
            "diarize": false   # optional, default: false
        }
    }
    """
    try:
        # Parse input
        input_data = event.get("input", {})
        file_url = input_data.get("file_url")
        language = input_data.get("language", "he")
        do_diarize = input_data.get("diarize", False)
        
        if not file_url:
            return {"error": "Missing required parameter: file_url"}
        
        print(f"Processing: {file_url}")
        print(f"Language: {language}, Diarize: {do_diarize}")
        
        # Download audio file
        print("Downloading audio...")
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            response = requests.get(file_url, timeout=300)
            response.raise_for_status()
            tmp.write(response.content)
            tmp.flush()
            audio_path = tmp.name
        
        print(f"Audio downloaded: {os.path.getsize(audio_path)} bytes")
        
        # Transcribe
        print("Transcribing...")
        result = whisper_model.transcribe(
            audio_path, 
            language=language,
            fp16=False
        )
        transcription = result["text"]
        print(f"✓ Transcription complete: {len(transcription)} characters")
        
        # Speaker diarization
        speakers = []
        if do_diarize:
            if not diarization_pipeline:
                return {
                    "error": "Diarization requested but not available (HF_TOKEN not configured or loading failed)"
                }
            
            print("Running diarization...")
            diarization = diarization_pipeline(audio_path)
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speakers.append({
                    "speaker": speaker,
                    "start": round(turn.start, 2),
                    "end": round(turn.end, 2)
                })
            print(f"✓ Found {len(speakers)} speaker segments")
        
        # Cleanup
        try:
            os.unlink(audio_path)
        except:
            pass
        
        # Return results
        return {
            "transcription": transcription,
            "speakers": speakers,
            "language": language,
            "duration": result.get("duration", 0)
        }
        
    except requests.RequestException as e:
        return {"error": f"Failed to download audio: {str(e)}"}
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error occurred:\n{error_details}")
        return {"error": str(e), "details": error_details}

if __name__ == "__main__":
    print("=" * 60)
    print("Starting Runpod Serverless Worker")
    print("=" * 60)
    runpod.serverless.start({"handler": handler})
