# ─── Stage 1: build React frontend ────────────────────────────────
FROM node:20-slim AS frontend
WORKDIR /fe
COPY frontend/package*.json ./
RUN npm install --silent
COPY frontend/ ./
RUN npm run build

# ─── Stage 2: Python runtime ──────────────────────────────────────
FROM python:3.11-slim
WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Python deps. torch CPU wheels are smaller than default.
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download RoBERTa weights into the image so first request isn't slow
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
    m='cardiffnlp/twitter-roberta-base-sentiment-latest'; \
    AutoTokenizer.from_pretrained(m); \
    AutoModelForSequenceClassification.from_pretrained(m)"

# NLTK data
RUN python -c "import nltk; [nltk.download(p, quiet=True) for p in ('punkt','punkt_tab','stopwords')]"

# Copy backend source
COPY backend/ ./

# Copy built frontend from stage 1
COPY --from=frontend /fe/build ./build

ENV PORT=5000 \
    DB_PATH=/tmp/sentiment.db \
    PYTHONUNBUFFERED=1 \
    TRANSFORMERS_OFFLINE=0 \
    HF_HUB_DISABLE_TELEMETRY=1

EXPOSE 5000

# uvicorn workers=1 because the model is loaded per-process and Railway's free tier has limited RAM
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-5000} --workers 1 --timeout-keep-alive 120"]
