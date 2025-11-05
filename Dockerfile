FROM python:3.10-slim

WORKDIR /app

# התקנת תלות מערכת
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# העתקת קבצים
COPY requirements.txt .
COPY main.py .

# התקנה
RUN pip install --no-cache-dir --upgrade pip

# התקן numpy 2
RUN pip install --no-cache-dir "numpy>=2.0,<3.0"

# התקן torch
RUN pip install --no-cache-dir \
    torch==2.4.0 \
    torchaudio==2.4.0

# התקן pyannote-audio 3.x (עם מקף!)
RUN pip install --no-cache-dir pyannote-audio==3.1.1

# שאר החבילות
RUN pip install --no-cache-dir \
    openai-whisper \
    lightning==2.3.0 \
    runpod==1.6.2 \
    requests \
    soundfile

# וידוא גרסאות
RUN python -c "import numpy; print('NumPy:', numpy.__version__)"
RUN python -c "import pyannote.audio; print('pyannote.audio:', pyannote.audio.__version__)"

# הורדת מודל Whisper
RUN python -c "import whisper; whisper.load_model('base')"

EXPOSE 8000

CMD ["python", "-u", "main.py"]
