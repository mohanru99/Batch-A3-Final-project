# AI Sentiment Analyzer — v2 (Real-time edition)

End-to-end rebuild of the v1 project. **No more 300 hardcoded seed reviews. No more regex pretending to be a transformer.**

## What changed vs v1

| Concern | v1 (old) | v2 (this) |
|---|---|---|
| **Transformer** | Regex pattern-match called "RoBERTa simulation" | **Real `cardiffnlp/twitter-roberta-base-sentiment-latest`** loaded on startup |
| **Training data** | 300 hand-written seed reviews | **Live scraped reviews persisted in SQLite, model retrains automatically** after each scrape |
| **Scraping** | Trustpilot only (often blocked from Railway IPs); fell back to grabbing any `<p>` tag | **Multi-source: Reddit JSON API (never blocks), Hacker News Algolia API, Trustpilot HTML** — fan-out async with `asyncio.gather` |
| **Response delivery** | Blocking — UI spins for 30s | **Server-Sent Events** — reviews stream into the UI one-by-one as they're scraped + classified |
| **Caching** | None | **30-min query+sources cache in SQLite** — instant repeat queries |
| **Persistence** | Lost on every restart | **SQLite at `/tmp/sentiment.db`** — full history, dedup by content hash |
| **Framework** | Flask sync | **FastAPI async** — handles 10s of concurrent scrapes |
| **Batch inference** | One-by-one prediction loop | **Vectorized RoBERTa batch path** for upload + non-streaming scrape |
| **Ensemble vote** | Plurality of 8 sklearn models | **Weighted vote**: RoBERTa weight=2, each sklearn weight=1 |

## Architecture

```
                 ┌─────────────────────────────────────────────┐
                 │  React frontend (Recharts, EventSource)     │
                 │  · Live tab streams reviews via SSE         │
                 │  · Predict tab calls /api/predict           │
                 │  · Upload tab posts CSV/XLSX                │
                 │  · History tab reads /api/stats + history   │
                 └────────────┬────────────────────────────────┘
                              │
                              ▼
        ┌────────────────── FastAPI ────────────────────────┐
        │  /api/scrape/stream  ── SSE: meta → review → done │
        │  /api/scrape         ── blocking JSON variant     │
        │  /api/predict        ── single-text inference     │
        │  /api/upload         ── CSV/XLSX → batch + retrain│
        │  /api/history        ── persisted reviews         │
        │  /api/stats          ── aggregates                │
        └────────────────┬────────────────────┬─────────────┘
                         │                    │
                         ▼                    ▼
   ┌─────────────────────────────┐   ┌──────────────────────────┐
   │   Async scrapers (httpx)    │   │   Inference layer        │
   │   · Reddit (JSON, public)   │   │   · RoBERTa (real, GPU/  │
   │   · Hacker News (Algolia)   │   │     CPU autodetect)      │
   │   · Trustpilot (HTML+JSON-  │   │   · 8 sklearn models     │
   │     LD, best-effort)        │   │     (LR/NB/RF/MLP × 2vec)│
   │   · gather() in parallel    │   │   · Weighted ensemble    │
   └─────────────────────────────┘   └────────────┬─────────────┘
                                                  │
                                                  ▼
                                    ┌─────────────────────────┐
                                    │  SQLite (/tmp/sent.db)  │
                                    │  · reviews (dedup hash) │
                                    │  · cache (TTL=30min)    │
                                    │  · auto-retrain trigger │
                                    └─────────────────────────┘
```

## Run locally

```bash
# Backend
cd backend
pip install -r requirements.txt
python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
  m='cardiffnlp/twitter-roberta-base-sentiment-latest'; \
  AutoTokenizer.from_pretrained(m); AutoModelForSequenceClassification.from_pretrained(m)"
uvicorn app:app --reload --port 5000

# Frontend (new terminal)
cd frontend
npm install
npm start                  # http://localhost:3000
```

## Deploy on Railway

1. Push this folder to GitHub
2. New project → Deploy from repo
3. Railway auto-detects `Dockerfile` and `railway.json`
4. (Optional) set `OUTSCRAPER_API_KEY` env var — only needed if you want Google Maps reviews later
5. First build is ~3–4 min (downloads RoBERTa weights into the image)

## API examples

```bash
# Health
curl https://your-app.up.railway.app/api/health

# Live stream (Server-Sent Events)
curl -N "https://your-app.up.railway.app/api/scrape/stream?query=tesla&sources=reddit,hackernews&limit=20"

# Blocking scrape
curl "https://your-app.up.railway.app/api/scrape?query=iphone+15&sources=reddit&limit=15"

# Predict single text
curl -X POST https://your-app.up.railway.app/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"absolutely loved the camera quality"}'

# Upload + retrain
curl -X POST https://your-app.up.railway.app/api/upload -F "file=@reviews.csv"
```

## Notes on the scraping strategy

- **Reddit** is the workhorse. The public `/search.json` endpoint returns up to 100 results per call, never rate-limits hobby usage, and contains long-form opinion text (perfect for sentiment).
- **Hacker News** via Algolia is free, fast, JSON. Especially good for tech topics.
- **Trustpilot** is included for completeness but is best-effort — it works from local dev and many cloud IPs but Railway's IP range is sometimes blocked. If it returns 0, the other sources still fill in.
- For **Amazon** specifically, direct scraping is blocked at the network level by Amazon's anti-bot. The right answer there is the **Amazon Product Advertising API** (requires affiliate account) — drop that in as another async source if needed.

## Tunables

In `backend/app.py`:

- `CACHE_TTL_MIN = 30` — repeat-query cache duration
- `SCRAPE_TIMEOUT = 12` — per-source HTTP timeout
- `ROBERTA_MODEL` — swap to any HF sentiment model
- `min_rows=60` in `retrain_from_db` — threshold to retrain ensemble
