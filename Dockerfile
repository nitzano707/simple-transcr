FROM python:3.10-slim

WORKDIR /app

# התקנת תלות מערכת
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# העתקת קבצים
COPY main.py .

# שדרוג pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# שלב 1: NumPy 2.x
RUN pip install --no-cache-dir "numpy>=2.0,<3.0"

# שלב 2: PyTorch
RUN pip install --no-cache-dir torch==2.4.0 torchaudio==2.4.0

# שלב 3: Lightning (לפני pyannote!)
RUN pip install --no-cache-dir lightning==2.3.0

# שלב 4: pyannote-audio 3.x (עם מקף!)
RUN pip install --no-cache-dir pyannote-audio==3.1.1

# שלב 5: Whisper
RUN pip install --no-cache-dir openai-whisper

# שלב 6: Runpod + שאר החבילות
RUN pip install --no-cache-dir runpod==1.6.2 requests soundfile

# וידוא גרסאות - אם זה נכשל הבנייה תעצור!
RUN python -c "import numpy; print('NumPy:', numpy.__version__); assert numpy.__version__.startswith('2'), 'Wrong NumPy!'"
RUN python -c "import pyannote.audio; print('pyannote.audio:', pyannote.audio.__version__); assert pyannote.audio.__version__.startswith('3'), 'Wrong pyannote.audio!'"

# בדיקה שהכל עובד
RUN python -c "from pyannote.audio import Pipeline; print('✅ Pipeline OK')"

# הורדת מודל Whisper
RUN python -c "import whisper; whisper.load_model('base')"

EXPOSE 8000

CMD ["python", "-u", "main.py"]
