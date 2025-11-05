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

# התקנת חבילות Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir numpy==1.24.3 && \
    pip install --no-cache-dir torch==2.0.1 torchaudio==2.0.2 && \
    pip install --no-cache-dir -r requirements.txt

# הורדת מודל Whisper מראש (אופציונלי - מאיץ את ההתחלה)
RUN python -c "import whisper; whisper.load_model('base')"

# Runpod Serverless צריך port 8000
EXPOSE 8000

# הרצת ה-handler
CMD ["python", "-u", "main.py"]
