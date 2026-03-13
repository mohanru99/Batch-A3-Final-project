"""
AI-Based Intelligent Customer Feedback Analyzer
Flask API — 8 Traditional ML models + optional RoBERTa
"""

import os, re, time, json
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

app = Flask(__name__, static_folder='build', static_url_path='')
CORS(app)

models = {}
vectorizers = {}
LABELS = ['negative','neutral','positive']

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))


# ===============================
# TEXT PREPROCESSING
# ===============================
def preprocess(text):
    if not isinstance(text,str):
        return ""

    text = text.lower()
    text = re.sub(r'<[^>]+>','',text)
    text = re.sub(r'http\S+','',text)
    text = re.sub(r'[^\w\s]','',text)
    text = re.sub(r'\d+','',text)

    tokens = word_tokenize(text)
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words and len(w)>2]

    return " ".join(tokens)


def score_to_sentiment(score):
    if score <=2:
        return "negative"
    elif score==3:
        return "neutral"
    else:
        return "positive"


# ===============================
# DATA LOADING (FIXED VERSION)
# ===============================
def load_data(filepath=None,data=None):

    df = pd.DataFrame(data) if data else pd.read_csv(filepath)

    df = df.dropna().drop_duplicates()

    # detect text column
    text_col=None
    for c in df.columns:
        if "text" in c.lower() or "review" in c.lower():
            text_col=c
            break

    if text_col is None:
        raise ValueError("No review/text column found")

    # detect rating column
    rating_col=None
    for c in df.columns:
        if "rating" in c.lower() or "score" in c.lower() or "stars" in c.lower():
            rating_col=c
            break

    # create sentiment labels
    if rating_col:
        df["sentiment"]=df[rating_col].apply(
            lambda x: score_to_sentiment(int(x)) if str(x).isdigit() else "neutral"
        )
    elif "sentiment" in df.columns:
        pass
    else:
        raise ValueError("Dataset must contain rating/score column")

    df["clean_text"]=df[text_col].apply(preprocess)

    df=df[df["clean_text"].str.len()>0]

    # balance dataset
    mx=df["sentiment"].value_counts().max()

    parts=[]
    for cls in LABELS:
        sub=df[df["sentiment"]==cls]

        if len(sub)==0:
            continue

        if len(sub)<mx:
            sub=resample(sub,replace=True,n_samples=mx,random_state=42)

        parts.append(sub)

    return pd.concat(parts)


# ===============================
# FEATURE EXTRACTION
# ===============================
def extract_features(X_train,X_test):

    tfidf=TfidfVectorizer(max_features=10000,ngram_range=(1,2))
    bow=CountVectorizer(max_features=10000,ngram_range=(1,2))

    feats={
        "tfidf":(tfidf.fit_transform(X_train),tfidf.transform(X_test)),
        "bow":(bow.fit_transform(X_train),bow.transform(X_test))
    }

    vectorizers["tfidf"]=tfidf
    vectorizers["bow"]=bow

    return feats


# ===============================
# TRAIN MODELS
# ===============================
def train_all(feats,y_train,y_test):

    configs={
        "logistic_regression":LogisticRegression(max_iter=1000),
        "naive_bayes":MultinomialNB(),
        "random_forest":RandomForestClassifier(n_estimators=100),
        "feedforward_nn":MLPClassifier(hidden_layer_sizes=(128,),max_iter=200)
    }

    results={}

    for vname,(Xtr,Xte) in feats.items():

        for mname,mdl in configs.items():

            key=f"{mname}_{vname}"

            model=type(mdl)(**mdl.get_params())

            model.fit(Xtr,y_train)

            y_pred=model.predict(Xte)

            acc=accuracy_score(y_test,y_pred)

            models[key]=model

            results[key]={
                "accuracy":round(acc,4),
                "vectorizer":vname,
                "model":mname
            }

    return results


# ===============================
# SINGLE PREDICTION
# ===============================
def predict_one(text,model_key):

    clean=preprocess(text)

    vtype="tfidf" if "tfidf" in model_key else "bow"

    vec=vectorizers.get(vtype)
    mdl=models.get(model_key)

    if not vec or not mdl:
        return None

    X=vec.transform([clean])

    pred=mdl.predict(X)[0]

    proba=mdl.predict_proba(X)[0]

    conf=float(np.max(proba))

    scores={LABELS[i]:round(float(proba[i]),4) for i in range(len(LABELS))}

    return {
        "sentiment":pred,
        "confidence":round(conf,4),
        "all_scores":scores
    }


def predict_all_models(text):

    preds={}

    for key in models:

        r=predict_one(text,key)

        if r:
            preds[key]=r

    return preds


# ===============================
# API ROUTES
# ===============================
@app.route("/api/health")
def health():

    return jsonify({
        "status":"ok",
        "models":list(models.keys()),
        "time":datetime.now().isoformat()
    })


@app.route("/api/train",methods=["POST"])
def train():

    try:

        if "file" in request.files:

            f=request.files["file"]

            path=f"/tmp/{f.filename}"

            f.save(path)

            df=load_data(filepath=path)

        else:
            return jsonify({"error":"upload csv"}),400


        X_train,X_test,y_train,y_test=train_test_split(
            df["clean_text"],
            df["sentiment"],
            test_size=0.3,
            stratify=df["sentiment"],
            random_state=42
        )

        feats=extract_features(X_train,X_test)

        results=train_all(feats,y_train,y_test)

        return jsonify({
            "status":"trained",
            "dataset_size":len(df),
            "results":results
        })

    except Exception as e:

        return jsonify({"error":str(e)}),500


@app.route("/api/predict",methods=["POST"])
def predict():

    data=request.json

    text=data.get("text","")

    if not text:
        return jsonify({"error":"no text"}),400

    preds=predict_all_models(text)

    sents=[p["sentiment"] for p in preds.values()]

    ens=Counter(sents).most_common(1)[0][0] if sents else "neutral"

    avg=float(np.mean([p["confidence"] for p in preds.values()]))

    return jsonify({
        "text":text,
        "ensemble":{"sentiment":ens,"confidence":round(avg,4)},
        "models":preds
    })


# ===============================
# SERVE REACT
# ===============================
@app.route("/",defaults={"path":""})
@app.route("/<path:path>")
def serve(path):

    if app.static_folder and os.path.exists(os.path.join(app.static_folder,path or "index.html")):
        return send_from_directory(app.static_folder,path if path else "index.html")

    return jsonify({"msg":"API running"})


if __name__=="__main__":

    port=int(os.environ.get("PORT",5000))

    print("Server starting...")

    app.run(host="0.0.0.0",port=port)
