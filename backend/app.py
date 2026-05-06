"""
AI-Based Intelligent Customer Feedback Analyzer — backend.

Architecture:
  • Sklearn ensemble: 8 models pretrained at Docker build time (pretrain.py),
    pickled to /app/models/, loaded on startup. No cold-start needed.
  • RoBERTa transformer: cardiffnlp/twitter-roberta-base-sentiment-latest,
    weights baked into image.
  • Scrapers: news + HN + reddit + ddg, all working from cloud IPs.
  • Storage: SQLite for history, in-mem cache for recent queries.
"""
import asyncio
import hashlib
import json
import logging
import os
import pickle
import re
import sqlite3
from collections import Counter
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import scrapers

# ─────────────────────────────────────────────────────────────────────
# Config & globals
# ─────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

DB_PATH = os.environ.get("DB_PATH", "/tmp/sentiment.db")
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/app/models"))
ROBERTA_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
LABELS = ["negative", "neutral", "positive"]
CACHE_TTL_MIN = 30

for pkg in ("punkt", "punkt_tab", "stopwords"):
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

stemmer = PorterStemmer()
try:
    STOP = set(stopwords.words("english"))
except Exception:
    STOP = set()


def preprocess(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [stemmer.stem(w) for w in text.split() if w not in STOP and len(w) > 2]
    return " ".join(tokens)


# ─────────────────────────────────────────────────────────────────────
# SQLite for history + cache
# ─────────────────────────────────────────────────────────────────────
def db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hash TEXT UNIQUE,
            query TEXT, source TEXT, text TEXT, author TEXT, url TEXT,
            rating INTEGER, sentiment TEXT, confidence REAL,
            roberta_sentiment TEXT, roberta_conf REAL,
            scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_query   ON reviews(query);
        CREATE INDEX IF NOT EXISTS idx_scraped ON reviews(scraped_at);

        CREATE TABLE IF NOT EXISTS cache (
            key TEXT PRIMARY KEY,
            payload TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        conn.commit()


def cache_get(key: str) -> Optional[list]:
    with db() as conn:
        row = conn.execute(
            "SELECT payload, created_at FROM cache WHERE key=?", (key,)
        ).fetchone()
        if not row:
            return None
        ts = datetime.fromisoformat(row["created_at"])
        if datetime.now() - ts > timedelta(minutes=CACHE_TTL_MIN):
            return None
        return json.loads(row["payload"])


def cache_set(key: str, payload: list):
    with db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO cache(key,payload,created_at) VALUES (?,?,?)",
            (key, json.dumps(payload, default=str), datetime.now().isoformat()),
        )
        conn.commit()


def save_review(query: str, rev: dict):
    h = hashlib.md5(((rev.get("text") or "")[:300]).encode()).hexdigest()
    with db() as conn:
        try:
            conn.execute("""
                INSERT OR IGNORE INTO reviews
                (hash, query, source, text, author, url, rating,
                 sentiment, confidence, roberta_sentiment, roberta_conf)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (h, query, rev.get("source", ""), rev.get("text", "")[:2000],
                 rev.get("author", ""), rev.get("url", ""), rev.get("rating"),
                 rev.get("sentiment"), rev.get("confidence"),
                 rev.get("roberta", {}).get("sentiment"),
                 rev.get("roberta", {}).get("confidence")))
            conn.commit()
        except Exception as e:
            log.warning("save_review failed: %s", e)


def cache_key(query: str, sources: list[str]) -> str:
    return hashlib.md5(f"{query}|{','.join(sorted(sources))}".encode()).hexdigest()


# ─────────────────────────────────────────────────────────────────────
# Pretrained sklearn ensemble (loaded from /app/models)
# ─────────────────────────────────────────────────────────────────────
class Ensemble:
    def __init__(self):
        self.models: dict = {}
        self.vectorizers: dict = {}
        self.metrics: dict = {}
        self.trained_on = 0
        self.loaded = False

    def load(self):
        if not MODEL_DIR.exists():
            log.warning("MODEL_DIR %s missing — pretrain.py was not run", MODEL_DIR)
            return
        try:
            with open(MODEL_DIR / "tfidf.pkl", "rb") as f:
                self.vectorizers["tfidf"] = pickle.load(f)
            with open(MODEL_DIR / "bow.pkl", "rb") as f:
                self.vectorizers["bow"] = pickle.load(f)
        except Exception as e:
            log.error("Vectorizer load failed: %s", e)
            return

        for vn in ("tfidf", "bow"):
            for mn in ("logistic_regression", "naive_bayes", "random_forest", "feedforward_nn"):
                key = f"{mn}_{vn}"
                p = MODEL_DIR / f"{key}.pkl"
                if not p.exists():
                    log.warning("missing model: %s", key)
                    continue
                try:
                    with open(p, "rb") as f:
                        self.models[key] = pickle.load(f)
                except Exception as e:
                    log.error("Failed to load %s: %s", key, e)

        try:
            with open(MODEL_DIR / "metrics.json") as f:
                meta = json.load(f)
            self.metrics = {k: v.get("accuracy", 0) for k, v in meta.get("metrics", {}).items()}
            self.trained_on = meta.get("trained_on", 0)
        except Exception as e:
            log.warning("metrics.json load: %s", e)

        self.loaded = bool(self.models and self.vectorizers)
        log.info("Ensemble loaded: %d models, trained_on=%d", len(self.models), self.trained_on)

    def predict_one(self, text: str, key: str) -> Optional[dict]:
        clean = preprocess(text)
        if not clean:
            return None
        vt = "tfidf" if "tfidf" in key else "bow"
        vec = self.vectorizers.get(vt)
        mdl = self.models.get(key)
        if not vec or not mdl:
            return None
        X = vec.transform([clean])
        pred = mdl.predict(X)[0]
        if hasattr(mdl, "predict_proba"):
            proba = mdl.predict_proba(X)[0]
            classes = list(mdl.classes_)
            scores = {c: round(float(proba[i]), 4) for i, c in enumerate(classes)}
            conf = float(np.max(proba))
        else:
            scores = {pred: 0.8}
            conf = 0.8
        return {"sentiment": pred, "confidence": round(conf, 4), "all_scores": scores, "model": key}

    def predict_all(self, text: str) -> dict:
        out = {}
        for key in self.models:
            r = self.predict_one(text, key)
            if r:
                out[key] = r
        return out


ensemble = Ensemble()


# ─────────────────────────────────────────────────────────────────────
# RoBERTa
# ─────────────────────────────────────────────────────────────────────
class Roberta:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.id2label = {0: "negative", 1: "neutral", 2: "positive"}
        self.ready = False

    def load(self):
        if self.ready:
            return
        log.info("Loading RoBERTa on %s ...", self.device)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(ROBERTA_NAME)
            self.model = AutoModelForSequenceClassification.from_pretrained(ROBERTA_NAME)
            self.model.to(self.device).eval()
            self.ready = True
            log.info("RoBERTa ready")
        except Exception as e:
            log.error("RoBERTa load failed: %s", e)

    @torch.no_grad()
    def predict(self, text: str) -> dict:
        if not self.ready:
            return {"sentiment": "neutral", "confidence": 0.0, "all_scores": {}, "model": "roberta_unloaded"}
        text = (text or "")[:512]
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        out = self.model(**enc)
        probs = torch.softmax(out.logits, dim=-1)[0].cpu().numpy()
        idx = int(np.argmax(probs))
        return {
            "sentiment": self.id2label[idx],
            "confidence": round(float(probs[idx]), 4),
            "all_scores": {self.id2label[i]: round(float(probs[i]), 4) for i in range(3)},
            "model": "roberta_transformer",
        }

    @torch.no_grad()
    def predict_batch(self, texts: list[str]) -> list[dict]:
        if not self.ready or not texts:
            return [self.predict(t) for t in texts]
        clean = [(t or "")[:512] for t in texts]
        enc = self.tokenizer(clean, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)
        out = self.model(**enc)
        probs = torch.softmax(out.logits, dim=-1).cpu().numpy()
        results = []
        for p in probs:
            idx = int(np.argmax(p))
            results.append({
                "sentiment": self.id2label[idx],
                "confidence": round(float(p[idx]), 4),
                "all_scores": {self.id2label[i]: round(float(p[i]), 4) for i in range(3)},
                "model": "roberta_transformer",
            })
        return results


roberta = Roberta()


# ─────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────
def fuse(rob_pred: dict, sk_preds: dict) -> dict:
    """Weighted vote: RoBERTa weight=2, each sklearn model weight=1."""
    votes = Counter()
    confs = []
    if rob_pred and rob_pred.get("sentiment") and rob_pred.get("model") != "roberta_unloaded":
        votes[rob_pred["sentiment"]] += 2
        confs.append(rob_pred.get("confidence", 0))
    for v in sk_preds.values():
        votes[v["sentiment"]] += 1
        confs.append(v.get("confidence", 0))
    if not votes:
        return {"sentiment": "neutral", "confidence": 0.5}
    sent = votes.most_common(1)[0][0]
    conf = float(np.mean(confs)) if confs else 0.5
    return {"sentiment": sent, "confidence": round(conf, 4)}


def analyze_one(text: str) -> dict:
    rob = roberta.predict(text)
    sk = ensemble.predict_all(text)
    return {"roberta": rob, "models": sk, "ensemble": fuse(rob, sk)}


def analyze_batch(texts: list[str]) -> list[dict]:
    if not texts:
        return []
    robs = roberta.predict_batch(texts)
    out = []
    for t, rob in zip(texts, robs):
        sk = ensemble.predict_all(t)
        out.append({"roberta": rob, "models": sk, "ensemble": fuse(rob, sk)})
    return out


# ─────────────────────────────────────────────────────────────────────
# FastAPI
# ─────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    log.info("DB ready: %s", DB_PATH)
    ensemble.load()
    try:
        roberta.load()
    except Exception as e:
        log.error("Deferred roberta load: %s", e)
    yield


app = FastAPI(title="Sentiment Analyzer v2", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ─── routes ─────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "roberta_ready": roberta.ready,
        "device": roberta.device,
        "ensemble_loaded": ensemble.loaded,
        "sklearn_models": list(ensemble.models.keys()),
        "sklearn_metrics": ensemble.metrics,
        "trained_on": ensemble.trained_on,
        "available_sources": list(scrapers.SOURCE_FNS.keys()),
        "ts": datetime.now().isoformat(),
    }


class PredictBody(BaseModel):
    text: str


@app.post("/api/predict")
def predict(body: PredictBody):
    if not body.text or not body.text.strip():
        raise HTTPException(400, "text is required")
    return {"text": body.text, **analyze_one(body.text)}


@app.get("/api/scrape")
async def scrape(
    query: str = Query(..., min_length=2),
    sources: str = Query("news,hackernews,reddit"),
    limit: int = Query(30, ge=1, le=100),
    use_cache: bool = Query(True),
):
    """Blocking scrape — returns the whole batch once ready."""
    src_list = [s.strip() for s in sources.split(",") if s.strip()]
    ck = cache_key(query, src_list)

    if use_cache:
        cached = cache_get(ck)
        if cached:
            return {"reviews": cached, "count": len(cached), "cached": True, "query": query}

    raw = await scrapers.scrape_all(query, src_list, limit)
    if not raw:
        return {
            "reviews": [], "count": 0, "cached": False, "query": query,
            "warning": (
                "No content found. Try a different query — broader terms, "
                "or different sources. News + HN tend to give the best coverage."
            ),
        }

    analyses = analyze_batch([r["text"] for r in raw])
    out = []
    for rev, a in zip(raw, analyses):
        rev["roberta"] = a["roberta"]
        rev["models"] = a["models"]
        rev["ensemble"] = a["ensemble"]["sentiment"]
        rev["sentiment"] = a["ensemble"]["sentiment"]
        rev["confidence"] = a["ensemble"]["confidence"]
        save_review(query, rev)
        out.append(rev)

    cache_set(ck, out)
    return {"reviews": out, "count": len(out), "cached": False, "query": query}


@app.get("/api/scrape/stream")
async def scrape_stream(
    query: str = Query(..., min_length=2),
    sources: str = Query("news,hackernews,reddit"),
    limit: int = Query(30, ge=1, le=100),
    use_cache: bool = Query(True),
):
    """SSE stream — UI gets reviews as each source returns."""
    src_list = [s.strip() for s in sources.split(",") if s.strip()]
    ck = cache_key(query, src_list)

    async def gen():
        if use_cache:
            cached = cache_get(ck)
            if cached:
                yield _sse({"type": "meta", "cached": True, "count": len(cached), "query": query, "sources": src_list})
                for rev in cached:
                    yield _sse({"type": "review", "data": rev})
                yield _sse({"type": "done", "count": len(cached), "cached": True})
                return

        yield _sse({"type": "meta", "cached": False, "query": query, "sources": src_list})

        collected = []
        try:
            async for rev in scrapers.scrape_stream(query, src_list, limit):
                try:
                    a = analyze_one(rev["text"])
                except Exception as e:
                    log.warning("analyze failed: %s", e)
                    continue
                rev["roberta"] = a["roberta"]
                rev["models"] = a["models"]
                rev["ensemble"] = a["ensemble"]["sentiment"]
                rev["sentiment"] = a["ensemble"]["sentiment"]
                rev["confidence"] = a["ensemble"]["confidence"]
                save_review(query, rev)
                collected.append(rev)
                yield _sse({"type": "review", "data": rev})
        except Exception as e:
            log.error("stream error: %s", e)
            yield _sse({"type": "error", "message": str(e)})

        if collected:
            cache_set(ck, collected)

        if not collected:
            yield _sse({
                "type": "warning",
                "message": "No content found. Try broader terms or different sources.",
            })
        yield _sse({"type": "done", "count": len(collected), "cached": False})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, default=str)}\n\n"


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    ext = (file.filename or "").lower().split(".")[-1]
    raw = await file.read()
    path = f"/tmp/{file.filename}"
    with open(path, "wb") as f:
        f.write(raw)
    try:
        df = pd.read_csv(path, on_bad_lines="skip") if ext == "csv" else pd.read_excel(path)
    except Exception as e:
        raise HTTPException(400, f"Could not read file: {e}")

    tcol = next((c for c in ["Text", "text", "review", "Review", "review_body", "comment",
                              "review_text", "Summary"] if c in df.columns), None)
    if not tcol:
        for c in df.columns:
            if df[c].dtype == "object" and df[c].astype(str).str.len().mean() > 30:
                tcol = c
                break
    if not tcol:
        raise HTTPException(400, f"No text column found in {list(df.columns)}")

    rcol = next((c for c in ["Score", "score", "rating", "Rating", "stars",
                              "star_rating", "overall"] if c in df.columns), None)

    df = df.dropna(subset=[tcol]).head(300)
    texts = [str(t)[:2000] for t in df[tcol].tolist()]
    analyses = analyze_batch(texts)

    out = []
    for i, (text, a) in enumerate(zip(texts, analyses)):
        if len(text) < 10:
            continue
        rating = None
        if rcol:
            try:
                rating = int(float(df.iloc[i][rcol]))
            except Exception:
                pass
        rev = {
            "text": text, "rating": rating, "author": "upload",
            "source": f"upload:{file.filename}", "url": "",
            "score": rating or 0, "ts": 0,
            "roberta": a["roberta"], "models": a["models"],
            "ensemble": a["ensemble"]["sentiment"],
            "sentiment": a["ensemble"]["sentiment"],
            "confidence": a["ensemble"]["confidence"],
        }
        save_review(f"upload:{file.filename}", rev)
        out.append(rev)

    return {
        "reviews": out, "count": len(out), "total_rows": int(len(df)),
        "text_column": tcol, "rating_column": rcol,
        "metrics": ensemble.metrics, "trained_on": ensemble.trained_on,
    }


@app.get("/api/history")
def history(query: Optional[str] = None, limit: int = 100):
    sql = "SELECT * FROM reviews"
    args: tuple = ()
    if query:
        sql += " WHERE query LIKE ?"
        args = (f"%{query}%",)
    sql += " ORDER BY scraped_at DESC LIMIT ?"
    args = args + (limit,)
    with db() as conn:
        rows = [dict(r) for r in conn.execute(sql, args).fetchall()]
    return {"count": len(rows), "rows": rows}


@app.get("/api/stats")
def stats():
    with db() as conn:
        total = conn.execute("SELECT COUNT(*) c FROM reviews").fetchone()["c"]
        by_sent = {r["sentiment"]: r["c"] for r in conn.execute(
            "SELECT sentiment, COUNT(*) c FROM reviews WHERE sentiment IS NOT NULL GROUP BY sentiment"
        ).fetchall()}
        by_source = {r["source"]: r["c"] for r in conn.execute(
            "SELECT source, COUNT(*) c FROM reviews GROUP BY source ORDER BY c DESC LIMIT 20"
        ).fetchall()}
    return {
        "total_reviews": total,
        "by_sentiment": by_sent,
        "by_source": by_source,
        "ensemble_metrics": ensemble.metrics,
        "trained_on": ensemble.trained_on,
        "roberta_ready": roberta.ready,
    }


# ─── static frontend ────────────────────────────────────────────────
BUILD_DIR = "/app/build"
if os.path.isdir(BUILD_DIR):
    if os.path.isdir(os.path.join(BUILD_DIR, "static")):
        app.mount("/static", StaticFiles(directory=os.path.join(BUILD_DIR, "static")), name="static")

    @app.get("/")
    def root():
        return FileResponse(os.path.join(BUILD_DIR, "index.html"))

    @app.get("/{path:path}")
    def spa(path: str):
        full = os.path.join(BUILD_DIR, path)
        if os.path.isfile(full):
            return FileResponse(full)
        return FileResponse(os.path.join(BUILD_DIR, "index.html"))
else:
    @app.get("/")
    def root():
        return JSONResponse({
            "msg": "Sentiment Analyzer v2 API",
            "endpoints": [
                "/api/health",
                "/api/predict",
                "/api/scrape?query=...&sources=news,hackernews,reddit",
                "/api/scrape/stream?query=...",
                "/api/upload",
                "/api/history",
                "/api/stats",
            ],
        })


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, log_level="info")
