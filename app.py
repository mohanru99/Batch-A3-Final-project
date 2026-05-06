"""
AI-Based Intelligent Customer Feedback Analyzer — v2
Real-time multi-source scraping + RoBERTa + sklearn ensemble.

Pipeline:
  user query  ─► router picks sources (Reddit, HN, Trustpilot)
              ─► async fetch with timeout + cache
              ─► stream results to frontend via SSE as each review is analyzed
              ─► RoBERTa (real transformer) + sklearn ensemble vote
              ─► persist to SQLite for history + retraining
"""
import os
import re
import json
import time
import sqlite3
import hashlib
import asyncio
import logging
from datetime import datetime, timedelta
from collections import Counter
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

import numpy as np
import pandas as pd
import httpx
from bs4 import BeautifulSoup

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import resample

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ─────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("sentiment")

DB_PATH = os.environ.get("DB_PATH", "/tmp/sentiment.db")
ROBERTA_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
LABELS = ["negative", "neutral", "positive"]
CACHE_TTL_MIN = 30  # cache scrape results for 30 min
SCRAPE_TIMEOUT = 12  # seconds per source

# ─────────────────────────────────────────────────────────────────────
# NLTK bootstrap (quiet)
# ─────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────
# SQLite — persists scraped reviews + history across restarts
# ─────────────────────────────────────────────────────────────────────
def db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS reviews (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            hash        TEXT UNIQUE,
            query       TEXT,
            source      TEXT,
            text        TEXT,
            author      TEXT,
            rating      INTEGER,
            sentiment   TEXT,
            confidence  REAL,
            roberta_sentiment TEXT,
            roberta_conf      REAL,
            ensemble    TEXT,
            scraped_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_reviews_query ON reviews(query);
        CREATE INDEX IF NOT EXISTS idx_reviews_scraped ON reviews(scraped_at);

        CREATE TABLE IF NOT EXISTS cache (
            key         TEXT PRIMARY KEY,
            payload     TEXT,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
            "INSERT OR REPLACE INTO cache(key, payload, created_at) VALUES (?, ?, ?)",
            (key, json.dumps(payload), datetime.now().isoformat()),
        )
        conn.commit()


def save_review(query: str, rev: dict):
    h = hashlib.md5(rev["text"][:300].encode()).hexdigest()
    with db() as conn:
        try:
            conn.execute("""
                INSERT OR IGNORE INTO reviews
                (hash, query, source, text, author, rating, sentiment, confidence,
                 roberta_sentiment, roberta_conf, ensemble)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, (
                h, query, rev.get("source", ""), rev["text"][:2000],
                rev.get("author", ""), rev.get("rating"),
                rev.get("sentiment"), rev.get("confidence"),
                rev.get("roberta", {}).get("sentiment"),
                rev.get("roberta", {}).get("confidence"),
                rev.get("ensemble"),
            ))
            conn.commit()
        except Exception as e:
            log.warning("save_review failed: %s", e)


# ─────────────────────────────────────────────────────────────────────
# Text preprocessing
# ─────────────────────────────────────────────────────────────────────
def preprocess(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [stemmer.stem(w) for w in text.split() if w not in STOP and len(w) > 2]
    return " ".join(tokens)


def rating_to_sentiment(r) -> str:
    try:
        v = int(float(r))
        if v <= 2:
            return "negative"
        if v == 3:
            return "neutral"
        return "positive"
    except (TypeError, ValueError):
        return "neutral"


# ─────────────────────────────────────────────────────────────────────
# RoBERTa — real transformer, loaded once
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
            self.tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL)
            self.model = AutoModelForSequenceClassification.from_pretrained(ROBERTA_MODEL)
            self.model.to(self.device)
            self.model.eval()
            self.ready = True
            log.info("RoBERTa ready")
        except Exception as e:
            log.error("RoBERTa load failed: %s", e)
            self.ready = False

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
# Sklearn ensemble — trained on real scraped data once we have enough
# ─────────────────────────────────────────────────────────────────────
class Ensemble:
    def __init__(self):
        self.models: dict = {}
        self.vectorizers: dict = {}
        self.metrics: dict = {}
        self.trained_on = 0

    def cold_start(self):
        """Bootstrap with a small balanced seed so predictions don't crash before any data arrives."""
        seed = self._tiny_seed()
        self._train(seed["text"], seed["label"], note="cold_start")

    def retrain_from_db(self, min_rows: int = 60):
        """Pull every labeled review from SQLite and retrain. Called after each scrape batch."""
        with db() as conn:
            df = pd.read_sql_query(
                "SELECT text, rating, roberta_sentiment FROM reviews WHERE text IS NOT NULL",
                conn,
            )
        if df.empty:
            return False
        # Use rating when present, otherwise RoBERTa label as the silver label
        df["label"] = df.apply(
            lambda r: rating_to_sentiment(r["rating"]) if pd.notna(r["rating"]) else r["roberta_sentiment"],
            axis=1,
        )
        df = df.dropna(subset=["label"])
        if len(df) < min_rows:
            return False
        self._train(df["text"].tolist(), df["label"].tolist(), note=f"db_n={len(df)}")
        return True

    def _train(self, texts, labels, note=""):
        clean = [preprocess(t) for t in texts]
        # Drop empties
        pairs = [(c, l) for c, l in zip(clean, labels) if c]
        if len(pairs) < 20:
            log.warning("Not enough samples to train (%d)", len(pairs))
            return
        X, y = zip(*pairs)
        df = pd.DataFrame({"x": X, "y": y})
        # Balance classes via upsampling the minority
        mx = df["y"].value_counts().max()
        parts = []
        for cls in df["y"].unique():
            sub = df[df["y"] == cls]
            if len(sub) < mx:
                sub = resample(sub, replace=True, n_samples=mx, random_state=42)
            parts.append(sub)
        bal = pd.concat(parts).sample(frac=1, random_state=42)

        try:
            Xtr, Xte, ytr, yte = train_test_split(
                bal["x"], bal["y"], test_size=0.2, random_state=42, stratify=bal["y"]
            )
        except ValueError:
            Xtr, Xte, ytr, yte = train_test_split(bal["x"], bal["y"], test_size=0.2, random_state=42)

        tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
        bow = CountVectorizer(max_features=8000, ngram_range=(1, 2))
        Xtr_tf, Xte_tf = tfidf.fit_transform(Xtr), tfidf.transform(Xte)
        Xtr_bw, Xte_bw = bow.fit_transform(Xtr), bow.transform(Xte)
        self.vectorizers = {"tfidf": tfidf, "bow": bow}

        cfgs = {
            "logistic_regression": LogisticRegression(max_iter=600, C=1.0),
            "naive_bayes": MultinomialNB(alpha=0.3),
            "random_forest": RandomForestClassifier(n_estimators=120, max_depth=20, random_state=42, n_jobs=-1),
            "feedforward_nn": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400, random_state=42, early_stopping=True),
        }

        new_models, new_metrics = {}, {}
        for vn, (Xt, Xe) in [("tfidf", (Xtr_tf, Xte_tf)), ("bow", (Xtr_bw, Xte_bw))]:
            for mn, base in cfgs.items():
                key = f"{mn}_{vn}"
                m = type(base)(**base.get_params())
                m.fit(Xt, ytr)
                acc = accuracy_score(yte, m.predict(Xe))
                new_models[key] = m
                new_metrics[key] = round(float(acc), 4)
                log.info("  %s: %.3f", key, acc)
        self.models = new_models
        self.metrics = new_metrics
        self.trained_on = len(bal)
        log.info("Ensemble trained on %d samples (%s)", len(bal), note)

    def predict_one(self, text: str, key: str):
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
        return {
            "sentiment": pred,
            "confidence": round(conf, 4),
            "all_scores": scores,
            "model": key,
        }

    def predict_all(self, text: str) -> dict:
        out = {}
        for key in self.models:
            r = self.predict_one(text, key)
            if r:
                out[key] = r
        return out

    def _tiny_seed(self):
        # Just enough samples to keep the API responsive on first request.
        # Real training happens after the first scrape.
        pos = [
            "absolutely love this product works perfectly",
            "amazing quality fast shipping highly recommend",
            "exceeded expectations great value for money",
            "fantastic experience would buy again",
            "perfect exactly what I wanted",
            "outstanding service brilliant product",
            "very happy with this purchase",
            "excellent build quality feels premium",
            "best product I have ever bought",
            "incredible results works flawlessly",
            "really impressed great experience",
            "five stars no complaints whatsoever",
            "wonderful purchase delighted with everything",
            "superb quality and quick delivery",
            "highly satisfied with the result",
            "great value works as advertised",
            "love everything about this item",
            "top notch product strong recommend",
            "very pleased with my order",
            "exceptional quality and service",
        ]
        neg = [
            "terrible product complete waste of money",
            "broke after one day total disappointment",
            "awful quality do not buy",
            "worst purchase I have ever made",
            "scam product never arrived",
            "horrible experience customer service useless",
            "defective on arrival waste of time",
            "completely useless does not work",
            "garbage cheap rubbish avoid",
            "extremely disappointed refund please",
            "rip off product fell apart",
            "fraud company never delivered",
            "poor quality flimsy and broken",
            "stopped working within a week",
            "regret buying this overpriced junk",
            "absolutely terrible would not recommend",
            "fake product nothing like described",
            "dreadful customer service ignore me",
            "money back this is useless",
            "pathetic product does not work",
        ]
        neu = [
            "okay product nothing special",
            "average quality for the price",
            "decent but could be better",
            "fine just an ordinary item",
            "not bad not great either",
            "acceptable for the cost",
            "mediocre experience overall",
            "standard product as expected",
            "passable but unremarkable",
            "alright works as intended mostly",
            "moderate quality some flaws",
            "fair value normal product",
            "ordinary nothing to write home about",
            "so so experience meh",
            "does the job for the price",
            "neutral feelings on this one",
            "average performance reasonable cost",
            "expected quality nothing more",
            "fine for casual use",
            "decent but not amazing",
        ]
        texts = pos + neg + neu
        labels = ["positive"] * len(pos) + ["negative"] * len(neg) + ["neutral"] * len(neu)
        return {"text": texts, "label": labels}


ensemble = Ensemble()


# ─────────────────────────────────────────────────────────────────────
# Multi-source live scrapers  (async, with timeout, smart fallback)
# ─────────────────────────────────────────────────────────────────────
UA_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
]


def _ua():
    import random
    return random.choice(UA_POOL)


async def scrape_reddit(query: str, limit: int = 25) -> list[dict]:
    """Reddit JSON API — public, no auth, never blocks. Best primary source."""
    out = []
    headers = {"User-Agent": "sentiment-analyzer/2.0 (research project)"}
    # Search across all of reddit
    url = "https://www.reddit.com/search.json"
    params = {"q": query, "limit": min(limit, 100), "sort": "relevance", "t": "year"}
    try:
        async with httpx.AsyncClient(timeout=SCRAPE_TIMEOUT, headers=headers) as c:
            r = await c.get(url, params=params)
            if r.status_code != 200:
                log.warning("Reddit search %s -> %d", query, r.status_code)
                return out
            data = r.json()
            for child in data.get("data", {}).get("children", []):
                d = child.get("data", {})
                title = d.get("title", "")
                body = d.get("selftext", "") or ""
                text = (title + ". " + body).strip()
                if len(text) < 25:
                    continue
                out.append({
                    "text": text[:2000],
                    "rating": None,
                    "author": "u/" + d.get("author", "anon"),
                    "source": f"reddit:r/{d.get('subreddit', '?')}",
                    "url": "https://reddit.com" + d.get("permalink", ""),
                    "score": d.get("score", 0),
                    "ts": d.get("created_utc", 0),
                })
                if len(out) >= limit:
                    break
    except Exception as e:
        log.warning("scrape_reddit error: %s", e)
    return out


async def scrape_hn(query: str, limit: int = 25) -> list[dict]:
    """Hacker News Algolia API — public JSON, fast, never blocks."""
    out = []
    url = "https://hn.algolia.com/api/v1/search"
    params = {"query": query, "tags": "comment,story", "hitsPerPage": min(limit, 100)}
    try:
        async with httpx.AsyncClient(timeout=SCRAPE_TIMEOUT) as c:
            r = await c.get(url, params=params)
            if r.status_code != 200:
                return out
            for hit in r.json().get("hits", []):
                text = hit.get("comment_text") or hit.get("story_text") or hit.get("title") or ""
                # Strip HTML
                text = BeautifulSoup(text, "html.parser").get_text(" ", strip=True)
                if len(text) < 25:
                    continue
                out.append({
                    "text": text[:2000],
                    "rating": None,
                    "author": hit.get("author", "anon"),
                    "source": "hackernews",
                    "url": f"https://news.ycombinator.com/item?id={hit.get('objectID', '')}",
                    "score": hit.get("points") or 0,
                    "ts": hit.get("created_at_i", 0),
                })
    except Exception as e:
        log.warning("scrape_hn error: %s", e)
    return out


async def scrape_trustpilot(slug: str, pages: int = 2) -> list[dict]:
    """Trustpilot HTML — works sometimes; we try JSON-LD first, then HTML cards."""
    reviews = []
    headers = {
        "User-Agent": _ua(),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        async with httpx.AsyncClient(timeout=SCRAPE_TIMEOUT, headers=headers, follow_redirects=True) as c:
            for pg in range(1, pages + 1):
                url = f"https://www.trustpilot.com/review/{slug}?page={pg}"
                r = await c.get(url)
                if r.status_code != 200:
                    log.info("Trustpilot %s page %d -> %d", slug, pg, r.status_code)
                    continue
                soup = BeautifulSoup(r.text, "html.parser")

                # Method 1: JSON-LD (most reliable when present)
                for script in soup.find_all("script", type="application/ld+json"):
                    try:
                        ld = json.loads(script.string or "{}")
                    except Exception:
                        continue
                    blocks = ld.get("@graph") if isinstance(ld, dict) and "@graph" in ld else [ld]
                    for item in blocks if isinstance(blocks, list) else []:
                        if not isinstance(item, dict):
                            continue
                        for rev in item.get("review", []) or []:
                            text = (rev.get("reviewBody") or "").strip()
                            if not text or len(text) < 15:
                                continue
                            try:
                                rating = int(rev.get("reviewRating", {}).get("ratingValue", 3))
                            except Exception:
                                rating = 3
                            reviews.append({
                                "text": text[:2000],
                                "rating": rating,
                                "author": (rev.get("author", {}) or {}).get("name", "Trustpilot user"),
                                "source": f"trustpilot:{slug}",
                                "url": url,
                                "score": rating,
                                "ts": 0,
                            })

                # Method 2: HTML review cards (newer Trustpilot layout)
                for card in soup.select("[data-service-review-card-paper], article[data-review-id]"):
                    body = card.select_one("p[data-service-review-text-typography], [data-review-content]")
                    if not body:
                        continue
                    text = body.get_text(" ", strip=True)
                    if len(text) < 15:
                        continue
                    rating = 3
                    star_img = card.select_one("img[alt]")
                    if star_img:
                        m = re.search(r"Rated\s+(\d)", star_img.get("alt", "") or "")
                        if m:
                            rating = int(m.group(1))
                    reviews.append({
                        "text": text[:2000],
                        "rating": rating,
                        "author": "Trustpilot user",
                        "source": f"trustpilot:{slug}",
                        "url": url,
                        "score": rating,
                        "ts": 0,
                    })
                await asyncio.sleep(0.8)
    except Exception as e:
        log.warning("scrape_trustpilot error: %s", e)

    # Dedup
    seen, uniq = set(), []
    for r in reviews:
        key = r["text"][:120]
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)
    return uniq


async def scrape_smart(query: str, sources: list[str], limit: int = 30) -> AsyncIterator[dict]:
    """
    Yields review dicts as soon as each source returns. Async-streams.
    `sources` is a subset of: reddit, hackernews, trustpilot.
    For trustpilot, `query` should be the company slug (e.g. "amazon.com").
    """
    tasks = []
    if "reddit" in sources:
        tasks.append(("reddit", scrape_reddit(query, limit)))
    if "hackernews" in sources:
        tasks.append(("hackernews", scrape_hn(query, limit)))
    if "trustpilot" in sources:
        tasks.append(("trustpilot", scrape_trustpilot(query)))

    if not tasks:
        return

    results = await asyncio.gather(*[t[1] for t in tasks], return_exceptions=True)
    for (name, _), res in zip(tasks, results):
        if isinstance(res, Exception):
            log.warning("Source %s failed: %s", name, res)
            continue
        for rev in res:
            yield rev


def cache_key(query: str, sources: list[str]) -> str:
    return hashlib.md5(f"{query}|{','.join(sorted(sources))}".encode()).hexdigest()


# ─────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    log.info("DB ready: %s", DB_PATH)
    # Cold-start sklearn so /predict works immediately
    ensemble.cold_start()
    # Load RoBERTa in background — don't block startup if it fails
    try:
        roberta.load()
    except Exception as e:
        log.error("RoBERTa load deferred: %s", e)
    # Try to retrain from any prior data
    try:
        ensemble.retrain_from_db()
    except Exception as e:
        log.warning("Initial retrain skipped: %s", e)
    yield


app = FastAPI(title="Sentiment Analyzer v2", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


# ── helpers ──
def fuse(roberta_pred: dict, ensemble_preds: dict) -> dict:
    """Ensemble vote: RoBERTa weight=2, each sklearn model weight=1."""
    votes: Counter = Counter()
    confs: list[float] = []
    if roberta_pred and roberta_pred.get("sentiment"):
        votes[roberta_pred["sentiment"]] += 2
        confs.append(roberta_pred.get("confidence", 0))
    for v in ensemble_preds.values():
        votes[v["sentiment"]] += 1
        confs.append(v.get("confidence", 0))
    if not votes:
        return {"sentiment": "neutral", "confidence": 0.5}
    sent = votes.most_common(1)[0][0]
    conf = float(np.mean(confs)) if confs else 0.5
    return {"sentiment": sent, "confidence": round(conf, 4)}


def analyze_one(text: str) -> dict:
    rob = roberta.predict(text) if roberta.ready else {"sentiment": "neutral", "confidence": 0.0, "all_scores": {}, "model": "roberta_unloaded"}
    sk = ensemble.predict_all(text)
    fused = fuse(rob, sk)
    return {
        "roberta": rob,
        "models": sk,
        "ensemble": fused,
    }


def analyze_batch(texts: list[str]) -> list[dict]:
    """Vectorized batch path — much faster for scraped batches."""
    if not texts:
        return []
    rob_results = roberta.predict_batch(texts) if roberta.ready else [
        {"sentiment": "neutral", "confidence": 0.0, "all_scores": {}, "model": "roberta_unloaded"}
        for _ in texts
    ]
    out = []
    for t, rob in zip(texts, rob_results):
        sk = ensemble.predict_all(t)
        fused = fuse(rob, sk)
        out.append({"roberta": rob, "models": sk, "ensemble": fused})
    return out


# ─────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "roberta_ready": roberta.ready,
        "device": roberta.device,
        "sklearn_models": list(ensemble.models.keys()),
        "sklearn_metrics": ensemble.metrics,
        "trained_on": ensemble.trained_on,
        "ts": datetime.now().isoformat(),
    }


class PredictBody(BaseModel):
    text: str


@app.post("/api/predict")
def predict(body: PredictBody):
    if not body.text:
        raise HTTPException(400, "text required")
    return {"text": body.text, **analyze_one(body.text)}


@app.get("/api/scrape/stream")
async def scrape_stream(
    query: str = Query(..., min_length=2),
    sources: str = Query("reddit,hackernews,trustpilot"),
    limit: int = Query(30, ge=1, le=100),
    use_cache: bool = Query(True),
):
    """
    Server-Sent Events. Frontend gets review-by-review updates instead of
    waiting for the whole scrape to finish.
    """
    src_list = [s.strip() for s in sources.split(",") if s.strip()]
    ck = cache_key(query, src_list)

    async def event_stream():
        # Cache check
        if use_cache:
            cached = cache_get(ck)
            if cached:
                yield _sse({"type": "meta", "cached": True, "count": len(cached)})
                for rev in cached:
                    yield _sse({"type": "review", "data": rev})
                yield _sse({"type": "done", "count": len(cached), "cached": True})
                return

        yield _sse({"type": "meta", "cached": False, "sources": src_list, "query": query})

        # Stream scraped reviews; analyze each as it lands
        collected: list[dict] = []
        async for rev in scrape_smart(query, src_list, limit):
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

        # Cache the batch
        if collected:
            cache_set(ck, collected)
            # Retrain ensemble in background (cheap with our sizes)
            try:
                if ensemble.retrain_from_db():
                    log.info("Ensemble retrained after scrape (n=%d)", ensemble.trained_on)
            except Exception as e:
                log.warning("Retrain after scrape failed: %s", e)

        yield _sse({"type": "done", "count": len(collected), "cached": False})

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, default=str)}\n\n"


@app.get("/api/scrape")
async def scrape_blocking(
    query: str = Query(..., min_length=2),
    sources: str = Query("reddit,hackernews,trustpilot"),
    limit: int = Query(30, ge=1, le=100),
    use_cache: bool = Query(True),
):
    """Non-streaming variant — returns the whole batch once it's ready. Easier to consume from cURL/tests."""
    src_list = [s.strip() for s in sources.split(",") if s.strip()]
    ck = cache_key(query, src_list)

    if use_cache:
        cached = cache_get(ck)
        if cached:
            return {"reviews": cached, "count": len(cached), "cached": True}

    raw: list[dict] = []
    async for rev in scrape_smart(query, src_list, limit):
        raw.append(rev)
    if not raw:
        return {"reviews": [], "count": 0, "warning": "No reviews found across requested sources."}

    # Batch analyze for speed
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
    try:
        ensemble.retrain_from_db()
    except Exception as e:
        log.warning("Retrain after scrape failed: %s", e)
    return {"reviews": out, "count": len(out), "cached": False}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    """Upload a CSV/XLSX of reviews; analyze + persist + retrain."""
    ext = (file.filename or "").lower().split(".")[-1]
    raw = await file.read()
    path = f"/tmp/{file.filename}"
    with open(path, "wb") as f:
        f.write(raw)

    try:
        df = pd.read_csv(path, on_bad_lines="skip") if ext == "csv" else pd.read_excel(path)
    except Exception as e:
        raise HTTPException(400, f"Could not read file: {e}")

    # Find text/rating columns
    tcol = next((c for c in ["Text","text","review","Review","review_body","comment","review_text","Summary"] if c in df.columns), None)
    if not tcol:
        # Pick the longest-string column
        for c in df.columns:
            if df[c].dtype == "object" and df[c].astype(str).str.len().mean() > 30:
                tcol = c
                break
    if not tcol:
        raise HTTPException(400, f"No text column found in {list(df.columns)}")
    rcol = next((c for c in ["Score","score","rating","Rating","stars","star_rating","overall"] if c in df.columns), None)

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
            "text": text,
            "rating": rating,
            "author": "upload",
            "source": f"upload:{file.filename}",
            "url": "",
            "score": rating or 0,
            "ts": 0,
            "roberta": a["roberta"],
            "models": a["models"],
            "ensemble": a["ensemble"]["sentiment"],
            "sentiment": a["ensemble"]["sentiment"],
            "confidence": a["ensemble"]["confidence"],
        }
        save_review(f"upload:{file.filename}", rev)
        out.append(rev)

    retrained = False
    try:
        retrained = ensemble.retrain_from_db()
    except Exception as e:
        log.warning("Retrain after upload failed: %s", e)

    return {
        "reviews": out,
        "count": len(out),
        "total_rows": int(len(df)),
        "text_column": tcol,
        "rating_column": rcol,
        "retrained": retrained,
        "trained_on": ensemble.trained_on,
        "metrics": ensemble.metrics,
    }


@app.get("/api/history")
def history(query: Optional[str] = None, limit: int = 100):
    sql = "SELECT * FROM reviews"
    args = ()
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
            "SELECT sentiment, COUNT(*) c FROM reviews GROUP BY sentiment"
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


# ── static frontend (built React) ──
BUILD_DIR = "/app/build"
if os.path.isdir(BUILD_DIR):
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
            "msg": "Sentiment Analyzer v2 API running",
            "endpoints": [
                "/api/health",
                "/api/predict",
                "/api/scrape?query=...",
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
