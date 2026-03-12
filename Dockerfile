FROM python:3.11-slim

WORKDIR /app

# System deps + Node.js
RUN apt-get update && \
    apt-get install -y gcc g++ curl && \
    rm -rf /var/lib/apt/lists/* && \
    curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs

# Copy everything
COPY . /app/

# Debug: show what was copied (check Railway logs)
RUN echo "=== FILES IN /app ===" && ls -la /app/ && echo "=== END ==="

# Python deps
RUN pip install --no-cache-dir -r /app/requirements.txt && \
    python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"

# Build React frontend (create inline if frontend/ folder is missing)
RUN if [ -d "/app/frontend" ] && [ -f "/app/frontend/package.json" ]; then \
      echo "Frontend folder found, building React..."; \
      cd /app/frontend && npm install && npm run build && \
      cp -r /app/frontend/build /app/build; \
    else \
      echo "No frontend folder — creating minimal index.html..."; \
      mkdir -p /app/build && \
      echo '<!DOCTYPE html><html><head><meta charset="utf-8"><title>AI Sentiment Analyzer</title></head><body><div id="root">Frontend not built. Push frontend/ folder to Git.</div></body></html>' > /app/build/index.html; \
    fi

EXPOSE 5000

CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:${PORT:-5000} --workers 2 --timeout 120"]
