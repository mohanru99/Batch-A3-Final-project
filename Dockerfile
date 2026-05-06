# ─── Single-stage Dockerfile ─────────────────────────────────────────
# Resilient: prints the build context, auto-detects backend/frontend
# locations, and pretrains all sklearn models at build time so the
# image ships ready to serve real predictions.

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

RUN echo "─── /app ───" && ls -la /app/ && \
    echo "─── /app/backend ───" && (ls -la /app/backend/ 2>/dev/null || echo "MISSING") && \
    echo "─── /app/frontend ───" && (ls -la /app/frontend/ 2>/dev/null || echo "MISSING")

# Locate requirements.txt
RUN if   [ -f /app/backend/requirements.txt ]; then REQ=/app/backend/requirements.txt; \
    elif [ -f /app/requirements.txt ]; then REQ=/app/requirements.txt; \
    else echo "✗ requirements.txt not found"; exit 1; fi && \
    echo "Using $REQ" && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r $REQ

# NLTK data — including the corpora needed for pretraining
RUN python -c "import nltk; \
    [nltk.download(p, quiet=True) for p in \
     ('punkt','punkt_tab','stopwords','movie_reviews','twitter_samples')]"

# Pre-download RoBERTa weights
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
    m='cardiffnlp/twitter-roberta-base-sentiment-latest'; \
    AutoTokenizer.from_pretrained(m); \
    AutoModelForSequenceClassification.from_pretrained(m)"

# ▼▼▼ PRETRAIN ALL 8 SKLEARN MODELS AT BUILD TIME ▼▼▼
# This bakes real, trained models into the image so first request is instant.
RUN if   [ -f /app/backend/pretrain.py ]; then cd /app/backend && python pretrain.py; \
    elif [ -f /app/pretrain.py ]; then cd /app && python pretrain.py; \
    else echo "✗ pretrain.py not found"; exit 1; fi && \
    echo "─── /app/models ───" && ls -la /app/models/

# Build React frontend if present, else placeholder
RUN if [ -d "/app/frontend" ] && [ -f "/app/frontend/package.json" ]; then \
        echo "▶ Building React..."; \
        cd /app/frontend && npm install --silent && npm run build && \
        cp -r /app/frontend/build /app/build && \
        echo "✓ React build complete"; \
    else \
        echo "▶ Frontend missing — placeholder"; \
        mkdir -p /app/build && \
        printf '%s\n' \
          '<!DOCTYPE html><html><head><meta charset="utf-8">' \
          '<title>Sentiment API</title>' \
          '<style>body{font-family:sans-serif;background:#0b1020;color:#e6e9ef;padding:40px;line-height:1.6}' \
          'a{color:#06b6d4}code{background:#1a224a;padding:2px 6px;border-radius:4px}</style>' \
          '</head><body><h1>Sentiment Analyzer API</h1>' \
          '<p>Backend running. Push the frontend folder to enable the UI.</p>' \
          '<ul>' \
          '<li><a href="/api/health">/api/health</a></li>' \
          '<li><code>GET /api/scrape?query=tesla</code></li>' \
          '<li><code>GET /api/scrape/stream?query=tesla</code></li>' \
          '<li><a href="/api/stats">/api/stats</a></li>' \
          '</ul></body></html>' > /app/build/index.html; \
    fi

# Detect backend path
RUN if   [ -f /app/backend/app.py ]; then echo "BACKEND=/app/backend" > /app/.runenv; \
    elif [ -f /app/app.py ]; then echo "BACKEND=/app" > /app/.runenv; \
    else echo "✗ app.py not found"; exit 1; fi && \
    cat /app/.runenv

ENV PORT=5000 \
    DB_PATH=/tmp/sentiment.db \
    MODEL_DIR=/app/models \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1

EXPOSE 5000

CMD ["sh", "-c", ". /app/.runenv && cd $BACKEND && uvicorn app:app --host 0.0.0.0 --port ${PORT:-5000} --workers 1 --timeout-keep-alive 120"]
