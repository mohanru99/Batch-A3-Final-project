# ─── Single-stage Dockerfile (resilient) ────────────────────────────
# Works whether frontend/ exists or not. If it's missing, ships API +
# placeholder page so you can confirm the API is up while debugging
# the repo layout.

FROM python:3.11-slim

WORKDIR /app

# System deps + Node 20 (for React build)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ curl ca-certificates && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

# Copy whole context, then inspect
COPY . /app/

# Print what Railway actually mounted (visible in build logs)
RUN echo "─── /app ───" && ls -la /app/ && \
    echo "─── /app/backend ───" && (ls -la /app/backend/ 2>/dev/null || echo "MISSING") && \
    echo "─── /app/frontend ───" && (ls -la /app/frontend/ 2>/dev/null || echo "MISSING")

# Locate requirements.txt (works whether you push backend/ or contents only)
RUN if   [ -f /app/backend/requirements.txt ]; then REQ=/app/backend/requirements.txt; \
    elif [ -f /app/requirements.txt ]; then REQ=/app/requirements.txt; \
    else echo "✗ requirements.txt not found anywhere"; exit 1; fi && \
    echo "Using $REQ" && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r $REQ

# Pre-download RoBERTa weights so first request isn't slow
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
    m='cardiffnlp/twitter-roberta-base-sentiment-latest'; \
    AutoTokenizer.from_pretrained(m); \
    AutoModelForSequenceClassification.from_pretrained(m)"

# NLTK data
RUN python -c "import nltk; [nltk.download(p, quiet=True) for p in ('punkt','punkt_tab','stopwords')]"

# Build React if frontend/ is present; else write a usable placeholder
RUN if [ -d "/app/frontend" ] && [ -f "/app/frontend/package.json" ]; then \
        echo "▶ Building React frontend..."; \
        cd /app/frontend && npm install --silent && npm run build && \
        cp -r /app/frontend/build /app/build && \
        echo "✓ React build complete"; \
    else \
        echo "▶ frontend/ missing — installing placeholder"; \
        mkdir -p /app/build && \
        printf '%s\n' \
          '<!DOCTYPE html><html><head><meta charset="utf-8">' \
          '<title>Sentiment API</title>' \
          '<style>body{font-family:sans-serif;background:#0b1020;color:#e6e9ef;padding:40px;line-height:1.6}' \
          'a{color:#06b6d4}code{background:#1a224a;padding:2px 6px;border-radius:4px}</style>' \
          '</head><body><h1>Sentiment Analyzer API</h1>' \
          '<p>Backend is running. Frontend not built (push the <code>frontend/</code> folder to enable the UI).</p>' \
          '<ul>' \
          '<li><a href="/api/health">GET /api/health</a></li>' \
          '<li><code>GET /api/scrape?query=tesla&amp;sources=reddit,hackernews</code></li>' \
          '<li><code>GET /api/scrape/stream?query=tesla</code> (SSE)</li>' \
          '<li><code>POST /api/predict</code> {"text":"..."}</li>' \
          '<li><a href="/api/stats">GET /api/stats</a></li>' \
          '</ul></body></html>' > /app/build/index.html; \
    fi

# Detect backend path so CMD works either way
RUN if   [ -f /app/backend/app.py ]; then echo "BACKEND=/app/backend" > /app/.runenv; \
    elif [ -f /app/app.py ]; then echo "BACKEND=/app" > /app/.runenv; \
    else echo "✗ app.py not found"; exit 1; fi && \
    cat /app/.runenv

# If backend is in /app/backend, the build/ folder needs to be findable
# by the FastAPI static mount which expects /app/build — already satisfied above.

ENV PORT=5000 \
    DB_PATH=/tmp/sentiment.db \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1

EXPOSE 5000

CMD ["sh", "-c", ". /app/.runenv && cd $BACKEND && uvicorn app:app --host 0.0.0.0 --port ${PORT:-5000} --workers 1 --timeout-keep-alive 120"]
