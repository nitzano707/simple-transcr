FROM python:3.10-slim

WORKDIR /app

# התקנת תלות מערכת
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# העתקת קבצים
COPY requirements.txt .
COPY main.py .

# שלב 1: שדרוג pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# שלב 2: התקנת NumPy 1.x בלבד (נעילה מוחלטת!)
RUN pip install --no-cache-dir "numpy==1.24.3" --force-reinstall

# שלב 3: התקנת PyTorch ללא תלות אוטומטיות
RUN pip install --no-cache-dir --no-deps \
    torch==2.0.1 \
    torchaudio==2.0.2

# שלב 4: התקנת שאר הספריות (עם נעילת numpy)
RUN pip install --no-cache-dir \
    openai-whisper==20230314 \
    pyannote.audio==2.1.1 \
    pytorch-lightning==2.0.9 \
    runpod==1.6.2 \
    requests==2.31.0 \
    soundfile==0.12.1 \
    typing-extensions \
    filelock

# שלב 5: וידוא שגרסת NumPy נכונה
RUN python -c "import numpy; print(f'NumPy version: {numpy.__version__}'); assert numpy.__version__.startswith('1.24'), 'Wrong NumPy version!'"

# שלב 6: הורדת מודל Whisper מראש (אופציונלי)
RUN python -c "import whisper; whisper.load_model('base')"

EXPOSE 8000

CMD ["python", "-u", "main.py"]
