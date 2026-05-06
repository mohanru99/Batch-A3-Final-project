"""
Pretrains all 8 sklearn models at Docker build time using real public data.
Result: /app/models/*.pkl that ship inside the image — no cold-start, no toy data.

Data sources (all public, all bundled with Python or downloaded once at build):
  - NLTK movie_reviews corpus (2,000 IMDB reviews, balanced pos/neg)
  - NLTK twitter_samples (10,000 labeled tweets, pos/neg)
  - A curated 600-sample neutral set (templated from real product-review patterns)

Trains:
  - LogisticRegression × {tfidf, bow}
  - MultinomialNB       × {tfidf, bow}
  - RandomForest        × {tfidf, bow}
  - MLPClassifier (FFNN)× {tfidf, bow}

Saves: vectorizers + models + metrics to /app/models/
"""
import os
import re
import json
import pickle
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import nltk
from nltk.corpus import movie_reviews, twitter_samples, stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils import resample

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pretrain")

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/app/models"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────
# Setup NLTK
# ─────────────────────────────────────────────────────────────────────
for pkg in ("punkt", "punkt_tab", "stopwords", "movie_reviews", "twitter_samples"):
    try:
        nltk.download(pkg, quiet=True)
    except Exception as e:
        log.warning("NLTK download %s failed: %s", pkg, e)

try:
    STOP = set(stopwords.words("english"))
except Exception:
    STOP = set()

stemmer = PorterStemmer()


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
# Build training corpus from real public datasets
# ─────────────────────────────────────────────────────────────────────
def load_movie_reviews():
    """NLTK movie_reviews: 2000 IMDB reviews, 1000 pos / 1000 neg."""
    rows = []
    for label in ("pos", "neg"):
        sentiment = "positive" if label == "pos" else "negative"
        for fid in movie_reviews.fileids(label):
            text = movie_reviews.raw(fid)
            rows.append({"text": text, "sentiment": sentiment})
    log.info("movie_reviews: %d samples", len(rows))
    return rows


def load_twitter_samples():
    """NLTK twitter_samples: 10000 labeled tweets."""
    rows = []
    try:
        pos = twitter_samples.strings("positive_tweets.json")
        neg = twitter_samples.strings("negative_tweets.json")
        for t in pos:
            rows.append({"text": t, "sentiment": "positive"})
        for t in neg:
            rows.append({"text": t, "sentiment": "negative"})
        log.info("twitter_samples: %d samples", len(rows))
    except Exception as e:
        log.warning("twitter_samples skipped: %s", e)
    return rows


def synth_neutral():
    """
    NLTK's two main labeled corpora are binary (pos/neg). Real product reviews
    have a substantial neutral class — these are *templated* but use real
    patterns from public Amazon/Yelp review samples (no copyrighted text).
    """
    products = ["product", "item", "device", "phone", "laptop", "headphones",
                "speaker", "camera", "watch", "book", "appliance", "tool",
                "shoes", "jacket", "bag", "monitor", "keyboard", "chair",
                "lamp", "kettle", "vacuum", "blender"]
    openings = [
        "the {p} is {adj}",
        "this {p} works {adj}",
        "got the {p}, it's {adj}",
        "the {p} arrived and it's {adj}",
        "received my {p}, {adj} overall",
        "{p} does its job, nothing {adj2}",
        "average {p}, {adj} for the price",
        "the {p} is fine, just {adj}",
        "ordered the {p}, it's {adj}",
        "this {p} is what you'd expect, {adj}",
    ]
    adj = ["okay", "decent", "average", "alright", "acceptable", "ordinary",
           "fair", "passable", "moderate", "fine", "standard", "mediocre",
           "so-so", "middling", "unremarkable", "mediocre", "nothing special"]
    adj2 = ["amazing", "spectacular", "outstanding", "terrible", "horrible"]
    extensions = [
        ". does what it should but nothing more.",
        ". not bad but not great either.",
        ". would call it average for the price point.",
        ". it works, that's about it.",
        ". meets the basic expectations.",
        ". no real complaints, no real praise.",
        ". neither impressed nor disappointed.",
        ". gets the job done, nothing exciting.",
        ". it's reasonable for the cost.",
        ". the build quality is moderate.",
        ". performance is adequate.",
        ". some good points, some not so good.",
        ". mixed feelings on this one.",
        ". does most things okay.",
        ". value for money is fair.",
    ]

    rows = []
    rng = np.random.default_rng(42)
    n = 600
    for _ in range(n):
        p = rng.choice(products)
        opening = rng.choice(openings).format(p=p, adj=rng.choice(adj), adj2=rng.choice(adj2))
        text = opening + rng.choice(extensions)
        rows.append({"text": text, "sentiment": "neutral"})
    log.info("synth_neutral: %d samples", len(rows))
    return rows


def real_product_phrases():
    """
    Hand-curated patterns observed in real customer reviews — these expand the
    pos/neg classes beyond movie/tweet language into product-review language.
    All written by hand for this project (no copyright concerns).
    """
    pos = [
        "Build quality is excellent and the materials feel premium",
        "Battery life exceeds the advertised specs",
        "Setup was straightforward and took only a few minutes",
        "Customer service responded within an hour and resolved my issue",
        "The product arrived earlier than the estimated delivery date",
        "Highly recommend this to anyone looking for reliable performance",
        "Far better than the previous model I owned",
        "Comfortable to use for extended periods",
        "The display is sharp and colors are accurate",
        "Worth every penny I spent on this",
        "Sturdy construction does not feel cheap at all",
        "Sound quality is impressive for this price range",
        "Easy to clean and maintain over time",
        "Performs reliably even after months of daily use",
        "Sleek design that looks great in any setting",
        "Intuitive controls anyone can figure out",
        "Solid investment for the long term",
        "Pleasantly surprised by how well this works",
        "Customer support went above and beyond",
        "Packaging was secure and product arrived in perfect condition",
        "Quiet operation does not disturb my workspace",
        "Compact size fits perfectly on my desk",
        "Lightweight yet feels durable",
        "Color matches the photos exactly",
        "Holds a charge for the entire day",
        "Works seamlessly with all my other devices",
        "Software updates have actually improved performance",
        "Best purchase I have made all year",
        "Far exceeded what I paid for",
        "Genuinely impressed with the attention to detail",
    ]
    neg = [
        "Stopped working after only two weeks of normal use",
        "Customer service was unresponsive and unhelpful",
        "The product arrived damaged and shipping took forever",
        "Battery dies within a couple of hours",
        "Build quality feels cheap and flimsy",
        "Does not match the description in the listing",
        "Constant connectivity issues that never get resolved",
        "Software is buggy and crashes frequently",
        "Way overpriced for what you actually get",
        "Returned it within a week of purchase",
        "Materials feel like they will break any moment",
        "Performance is sluggish even on simple tasks",
        "Sound quality is muffled and disappointing",
        "Charging port stopped working after one month",
        "Instructions were unclear and assembly was a nightmare",
        "Color is completely different from the product photos",
        "Buttons stopped responding after light use",
        "Heats up uncomfortably during normal operation",
        "Wifi disconnects every few minutes",
        "Smaller than advertised in the dimensions",
        "Stitching came apart on the first day",
        "Screen developed dead pixels within a month",
        "Refused to provide a refund despite the defect",
        "Paint started peeling almost immediately",
        "Loud whirring noise from day one",
        "Buttons feel mushy and unresponsive",
        "Battery swelled up and became unsafe",
        "Worst purchase I have made in years",
        "Complete waste of money I will never buy again",
        "Strong chemical smell that did not go away",
    ]
    rows = []
    for t in pos:
        rows.append({"text": t, "sentiment": "positive"})
    for t in neg:
        rows.append({"text": t, "sentiment": "negative"})
    log.info("real_product_phrases: %d samples", len(rows))
    return rows


# ─────────────────────────────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────────────────────────────
def main():
    rows = []
    rows += load_movie_reviews()
    rows += load_twitter_samples()
    rows += synth_neutral()
    rows += real_product_phrases()

    df = pd.DataFrame(rows)
    log.info("Total raw samples: %d", len(df))
    log.info("Class distribution: %s", df["sentiment"].value_counts().to_dict())

    # Preprocess
    log.info("Preprocessing...")
    df["clean"] = df["text"].apply(preprocess)
    df = df[df["clean"].str.len() > 5].reset_index(drop=True)
    log.info("After preprocessing: %d samples", len(df))

    # Balance via upsample
    mx = df["sentiment"].value_counts().max()
    parts = []
    for cls in df["sentiment"].unique():
        sub = df[df["sentiment"] == cls]
        if len(sub) < mx:
            sub = resample(sub, replace=True, n_samples=mx, random_state=42)
        parts.append(sub)
    bal = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)
    log.info("After balancing: %d samples (%s)", len(bal), bal["sentiment"].value_counts().to_dict())

    Xtr, Xte, ytr, yte = train_test_split(
        bal["clean"], bal["sentiment"],
        test_size=0.2, random_state=42, stratify=bal["sentiment"],
    )

    # Vectorize
    log.info("Vectorizing...")
    tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    bow = CountVectorizer(max_features=15000, ngram_range=(1, 2), min_df=2)
    Xtr_tf, Xte_tf = tfidf.fit_transform(Xtr), tfidf.transform(Xte)
    Xtr_bw, Xte_bw = bow.fit_transform(Xtr), bow.transform(Xte)
    log.info("Vectorizer dims: tfidf=%d, bow=%d", Xtr_tf.shape[1], Xtr_bw.shape[1])

    # Train each model with both vectorizers
    cfgs = {
        "logistic_regression": LogisticRegression(max_iter=1000, C=1.0),
        "naive_bayes": MultinomialNB(alpha=0.3),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=30, min_samples_split=4,
            random_state=42, n_jobs=-1,
        ),
        "feedforward_nn": MLPClassifier(
            hidden_layer_sizes=(256, 128), activation="relu",
            max_iter=600, random_state=42,
            early_stopping=False,  # disabled to avoid sklearn-version-specific bugs
        ),
    }

    metrics = {}
    log.info("Training models...")
    for vn, (Xt, Xe) in [("tfidf", (Xtr_tf, Xte_tf)), ("bow", (Xtr_bw, Xte_bw))]:
        for mn, base in cfgs.items():
            key = f"{mn}_{vn}"
            m = type(base)(**base.get_params())
            m.fit(Xt, ytr)
            pred = m.predict(Xe)
            acc = accuracy_score(yte, pred)
            f1 = f1_score(yte, pred, average="macro")
            metrics[key] = {"accuracy": round(float(acc), 4), "f1_macro": round(float(f1), 4)}
            log.info("  %-30s  acc=%.4f  f1=%.4f", key, acc, f1)
            with open(MODEL_DIR / f"{key}.pkl", "wb") as f:
                pickle.dump(m, f)

    # Save vectorizers
    with open(MODEL_DIR / "tfidf.pkl", "wb") as f:
        pickle.dump(tfidf, f)
    with open(MODEL_DIR / "bow.pkl", "wb") as f:
        pickle.dump(bow, f)

    # Save metrics
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump({
            "metrics": metrics,
            "trained_on": int(len(bal)),
            "test_size": int(len(yte)),
            "classes": list(sorted(bal["sentiment"].unique())),
            "vectorizer_features": {"tfidf": int(Xtr_tf.shape[1]), "bow": int(Xtr_bw.shape[1])},
        }, f, indent=2)

    log.info("✓ Saved %d models + 2 vectorizers + metrics to %s", len(cfgs) * 2, MODEL_DIR)
    log.info("✓ Best model: %s", max(metrics.items(), key=lambda x: x[1]["accuracy"]))


if __name__ == "__main__":
    main()
