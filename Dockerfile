FROM python:3.10-slim

WORKDIR /app

# תלות חשובה: ffmpeg + libsndfile + git
RUN apt-get update && apt-get install -y \
    ffmpeg libsndfile1 git && \
    rm -rf /var/lib/apt/lists/*

COPY . .

# התקנת תלות עם גרסאות תואמות
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
