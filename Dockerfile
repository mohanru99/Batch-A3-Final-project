FROM python:3.11-slim

WORKDIR /app

# System deps + Node.js
RUN apt-get update && \
    apt-get install -y gcc g++ curl && \
    rm -rf /var/lib/apt/lists/* && \
    curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs

# Copy EVERYTHING at once (avoids path issues)
COPY . /app/

# Python deps
RUN pip install --no-cache-dir -r /app/requirements.txt && \
    python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"

# Build React frontend
RUN cd /app/frontend && npm install && npm run build && \
    cp -r /app/frontend/build /app/build

EXPOSE 5000

CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:${PORT:-5000} --workers 2 --timeout 120"]
