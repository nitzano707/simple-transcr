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

# התקנה בסדר נכון
RUN pip install --no-cache-dir --upgrade pip

# התקן numpy 2 ראשון
RUN pip install --no-cache-dir "numpy>=2.0,<3.0"

# התקן torch + torchaudio
RUN pip install --no-cache-dir \
    torch==2.4.0 \
    torchaudio==2.4.0

# התקן שאר החבילות
RUN pip install --no-cache-dir -r requirements.txt

# וידוא שהכל עובד
RUN python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# הורדת מודל Whisper
RUN python -c "import whisper; whisper.load_model('base')"

EXPOSE 8000

CMD ["python", "-u", "main.py"]
