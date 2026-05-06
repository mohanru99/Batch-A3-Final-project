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
        self.metrics: dict = {}        # accuracy per model (flat)
        self.full_metrics: dict = {}   # full metrics.json contents
        self.confusion: dict = {}      # confusion matrices per model
        self.per_class: dict = {}      # per-class precision/recall/f1
        self.classes: list = []
        self.test_set: list = []       # held-out samples for live eval
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
            self.full_metrics = meta
            self.metrics = {k: v.get("accuracy", 0) for k, v in meta.get("metrics", {}).items()}
            self.confusion = meta.get("confusion", {})
            self.per_class = meta.get("per_class", {})
            self.classes = meta.get("classes", ["negative", "neutral", "positive"])
            self.trained_on = meta.get("trained_on", 0)
        except Exception as e:
            log.warning("metrics.json load: %s", e)

        try:
            with open(MODEL_DIR / "test_set.json") as f:
                self.test_set = json.load(f)
            log.info("Loaded %d held-out test samples", len(self.test_set))
        except Exception as e:
            log.warning("test_set.json load: %s", e)

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
    """
    Weighted vote with neutral-band correction.

    Problem we're fixing: news headlines and HN comments are heavily binary in
    polarity (or look that way to RoBERTa). If RoBERTa always says pos/neg with
    moderate confidence, neutral never wins.

    Solution:
      1. Equal weights — RoBERTa weight=1, each sklearn model weight=1.
         (Was 2 vs 1 before, which let RoBERTa dominate.)
      2. Confidence-margin neutral band: if RoBERTa's pos/neg confidence
         is below 0.65 AND its margin over neutral is < 0.20, treat as neutral.
      3. If sklearn ensemble is split (no clear majority), use mean confidence
         to break the tie toward neutral when below 0.55.
    """
    votes = Counter()
    confs = []

    rob_used = rob_pred and rob_pred.get("model") != "roberta_unloaded"
    if rob_used:
        rob_sent = rob_pred["sentiment"]
        rob_conf = rob_pred.get("confidence", 0.0)
        rob_scores = rob_pred.get("all_scores", {})
        # Confidence-margin neutral band
        if rob_sent in ("positive", "negative"):
            neu_score = rob_scores.get("neutral", 0.0)
            margin = rob_conf - neu_score
            if rob_conf < 0.65 and margin < 0.20:
                rob_sent = "neutral"
        votes[rob_sent] += 1
        confs.append(rob_conf)

    # Sklearn votes — also apply a per-model neutral correction when the model is uncertain
    for v in sk_preds.values():
        sent = v["sentiment"]
        scores = v.get("all_scores", {})
        conf = v.get("confidence", 0.0)
        if sent in ("positive", "negative"):
            neu_score = scores.get("neutral", 0.0)
            if conf < 0.55 and (conf - neu_score) < 0.10:
                sent = "neutral"
        votes[sent] += 1
        confs.append(conf)

    if not votes:
        return {"sentiment": "neutral", "confidence": 0.5, "agreement": 0.0}

    top = votes.most_common()
    sent = top[0][0]
    total_votes = sum(votes.values())
    agreement = top[0][1] / total_votes

    # If the vote is split close to 50/50 and confidence is moderate, lean neutral
    mean_conf = float(np.mean(confs)) if confs else 0.5
    if agreement < 0.55 and mean_conf < 0.65 and "neutral" in votes:
        sent = "neutral"

    return {
        "sentiment": sent,
        "confidence": round(mean_conf, 4),
        "agreement": round(float(agreement), 4),
    }


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
        rev["agreement"] = a["ensemble"].get("agreement", 0.0)
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
                rev["agreement"] = a["ensemble"].get("agreement", 0.0)
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
        # Average confidence across all stored reviews
        row = conn.execute(
            "SELECT AVG(confidence) c FROM reviews WHERE confidence IS NOT NULL"
        ).fetchone()
        avg_conf = float(row["c"]) if row and row["c"] is not None else None
    return {
        "total_reviews": total,
        "by_sentiment": by_sent,
        "by_source": by_source,
        "avg_confidence": round(avg_conf, 4) if avg_conf is not None else None,
        "ensemble_metrics": ensemble.metrics,
        "trained_on": ensemble.trained_on,
        "roberta_ready": roberta.ready,
    }


# ─────────────────────────────────────────────────────────────────────
# Confusion matrix + per-class metrics (computed at pretrain time)
# ─────────────────────────────────────────────────────────────────────
@app.get("/api/evaluate")
def evaluate():
    """
    Returns confusion matrices and per-class precision/recall/F1 for all 8
    sklearn models, computed on the held-out test set during pretraining.
    Plus a live evaluation of the ensemble fusion (RoBERTa + sklearn) on
    the same held-out set so you can see how the *combined* system performs.
    """
    if not ensemble.loaded:
        return {"error": "Ensemble not loaded"}

    out = {
        "classes": ensemble.classes,
        "test_size": ensemble.full_metrics.get("test_size", 0),
        "models": {},
    }

    # Pretrained per-model metrics (loaded from metrics.json)
    for key in ensemble.models:
        out["models"][key] = {
            "accuracy":  ensemble.full_metrics.get("metrics", {}).get(key, {}).get("accuracy"),
            "f1_macro":  ensemble.full_metrics.get("metrics", {}).get(key, {}).get("f1_macro"),
            "confusion": ensemble.confusion.get(key, []),
            "per_class": ensemble.per_class.get(key, {}),
        }

    # Live ensemble eval on held-out set (also includes RoBERTa contribution)
    if ensemble.test_set:
        sample = ensemble.test_set[:200]  # cap to keep response fast
        texts = [s["text"] for s in sample]
        truth = [s["label"] for s in sample]
        analyses = analyze_batch(texts)
        preds = [a["ensemble"]["sentiment"] for a in analyses]
        confs = [a["ensemble"]["confidence"] for a in analyses]

        from sklearn.metrics import (
            accuracy_score as _acc, f1_score as _f1,
            precision_recall_fscore_support as _prfs,
            confusion_matrix as _cm,
        )
        labels = ensemble.classes
        acc = float(_acc(truth, preds))
        f1m = float(_f1(truth, preds, average="macro", zero_division=0))
        cm = _cm(truth, preds, labels=labels).tolist()
        p, r, f, s = _prfs(truth, preds, labels=labels, zero_division=0)
        per_class = {
            cls: {
                "precision": round(float(p[i]), 4),
                "recall":    round(float(r[i]), 4),
                "f1":        round(float(f[i]), 4),
                "support":   int(s[i]),
            }
            for i, cls in enumerate(labels)
        }
        out["ensemble_live"] = {
            "accuracy":  round(acc, 4),
            "f1_macro":  round(f1m, 4),
            "avg_confidence": round(float(np.mean(confs)) if confs else 0.0, 4),
            "confusion": cm,
            "per_class": per_class,
            "n_samples": len(sample),
        }
    return out


# ─────────────────────────────────────────────────────────────────────
# Keyword / aspect extraction per sentiment class
# ─────────────────────────────────────────────────────────────────────
@app.get("/api/keywords")
def keywords(query: Optional[str] = None, top_n: int = 15):
    """
    Returns the top N most distinctive words/bigrams per sentiment class
    for either: (a) all stored reviews, or (b) a specific query.
    Uses TF-IDF scoring within each class.
    """
    sql = "SELECT text, sentiment FROM reviews WHERE sentiment IS NOT NULL AND text IS NOT NULL"
    args: tuple = ()
    if query:
        sql += " AND query LIKE ?"
        args = (f"%{query}%",)
    sql += " LIMIT 2000"
    with db() as conn:
        rows = conn.execute(sql, args).fetchall()

    if not rows:
        return {"error": "No reviews found", "query": query}

    # Group by sentiment, then run TF-IDF per class
    by_sent: dict[str, list[str]] = {"positive": [], "neutral": [], "negative": []}
    for r in rows:
        s = r["sentiment"]
        if s in by_sent:
            by_sent[s].append(preprocess(r["text"]))

    out = {}
    for sent, texts in by_sent.items():
        if len(texts) < 3:
            out[sent] = []
            continue
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            v = TfidfVectorizer(
                max_features=200, ngram_range=(1, 2),
                min_df=2, stop_words="english",
            )
            X = v.fit_transform(texts)
            scores = np.asarray(X.mean(axis=0)).ravel()
            terms = v.get_feature_names_out()
            top_idx = np.argsort(scores)[::-1][:top_n]
            out[sent] = [
                {"term": terms[i], "score": round(float(scores[i]), 4)}
                for i in top_idx
            ]
        except Exception as e:
            log.warning("keywords %s failed: %s", sent, e)
            out[sent] = []
    return {"keywords": out, "query": query, "total_reviews": len(rows)}


# ─────────────────────────────────────────────────────────────────────
# Time-series sentiment trend
# ─────────────────────────────────────────────────────────────────────
@app.get("/api/trend")
def trend(query: Optional[str] = None, bucket: str = "day"):
    """
    Returns sentiment counts bucketed over time for a specific query.
    bucket: 'hour' | 'day' (default day).
    Useful for seeing how sentiment shifts as new reviews arrive.
    """
    fmt = "%Y-%m-%d %H:00" if bucket == "hour" else "%Y-%m-%d"
    sql = (
        "SELECT strftime(?, scraped_at) AS t, sentiment, COUNT(*) AS c "
        "FROM reviews WHERE sentiment IS NOT NULL"
    )
    args: tuple = (fmt,)
    if query:
        sql += " AND query LIKE ?"
        args = args + (f"%{query}%",)
    sql += " GROUP BY t, sentiment ORDER BY t ASC"
    with db() as conn:
        rows = [dict(r) for r in conn.execute(sql, args).fetchall()]

    # Pivot into a chart-ready shape: [{t, positive, neutral, negative}]
    by_t: dict = {}
    for r in rows:
        t = r["t"]
        by_t.setdefault(t, {"t": t, "positive": 0, "neutral": 0, "negative": 0})
        if r["sentiment"] in by_t[t]:
            by_t[t][r["sentiment"]] = r["c"]
    series = sorted(by_t.values(), key=lambda x: x["t"])
    return {"series": series, "bucket": bucket, "query": query}


# ─────────────────────────────────────────────────────────────────────
# Compare two queries side-by-side
# ─────────────────────────────────────────────────────────────────────
@app.get("/api/compare")
async def compare(
    a: str = Query(..., min_length=2),
    b: str = Query(..., min_length=2),
    sources: str = Query("news,hackernews,reddit"),
    limit: int = Query(20, ge=5, le=50),
):
    """Scrape + analyze two queries and return aggregated comparison."""
    src_list = [s.strip() for s in sources.split(",") if s.strip()]

    async def run(q):
        raw = await scrapers.scrape_all(q, src_list, limit)
        if not raw:
            return {"query": q, "count": 0, "by_sentiment": {}, "avg_confidence": 0}
        analyses = analyze_batch([r["text"] for r in raw])
        sents = [x["ensemble"]["sentiment"] for x in analyses]
        confs = [x["ensemble"]["confidence"] for x in analyses]
        for rev, an in zip(raw, analyses):
            rev["sentiment"] = an["ensemble"]["sentiment"]
            rev["confidence"] = an["ensemble"]["confidence"]
            save_review(q, rev)
        c = Counter(sents)
        return {
            "query": q,
            "count": len(raw),
            "by_sentiment": dict(c),
            "avg_confidence": round(float(np.mean(confs)) if confs else 0.0, 4),
            "score": round(
                (c.get("positive", 0) - c.get("negative", 0)) / max(len(sents), 1),
                4,
            ),  # net sentiment from -1 (all neg) to +1 (all pos)
        }

    res_a, res_b = await asyncio.gather(run(a), run(b))
    return {"a": res_a, "b": res_b, "sources": src_list}


# ─────────────────────────────────────────────────────────────────────
# Export stored reviews to CSV
# ─────────────────────────────────────────────────────────────────────
@app.get("/api/export")
def export(query: Optional[str] = None, limit: int = 1000):
    """Download reviews as CSV. Filter by query if provided."""
    sql = "SELECT scraped_at, query, source, author, sentiment, confidence, rating, text, url FROM reviews"
    args: tuple = ()
    if query:
        sql += " WHERE query LIKE ?"
        args = (f"%{query}%",)
    sql += " ORDER BY scraped_at DESC LIMIT ?"
    args = args + (limit,)
    with db() as conn:
        df = pd.read_sql_query(sql, conn, params=args)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    fname = f"reviews_{query or 'all'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    from fastapi.responses import Response
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


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
