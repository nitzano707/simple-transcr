FROM python:3.10-slim

WORKDIR /app

# FFmpeg ו-Lib dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg git libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# העתקת הקוד
COPY . .

# התקנת תלויות
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# פתיחת פורט 8000
EXPOSE 8000

# הרצת השרת
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
