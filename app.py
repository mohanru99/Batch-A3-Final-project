"""
AI-Based Intelligent Customer Feedback Analyzer
Flask API — 8 Traditional ML models + optional RoBERTa
Serves React frontend from /build
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

import requests as http_req
from bs4 import BeautifulSoup

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

app = Flask(__name__, static_folder='build', static_url_path='')
CORS(app)

models = {}
vectorizers = {}
transformer_pipe = None
LABELS = ['negative', 'neutral', 'positive']
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# ═══════════════════════════════════
# PREPROCESSING (Paper Section III)
# ═══════════════════════════════════
def preprocess(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

def score_to_sentiment(score):
    if score <= 2: return 'negative'
    if score == 3: return 'neutral'
    return 'positive'

# ═══════════════════════════════════
# DATA LOADING + BALANCING
# ═══════════════════════════════════
def load_data(filepath=None, data=None):
    df = pd.DataFrame(data) if data else pd.read_csv(filepath)
    df = df.dropna().drop_duplicates()
    if 'Score' in df.columns:
        df['sentiment'] = df['Score'].apply(score_to_sentiment)
        tcol = 'Text' if 'Text' in df.columns else 'text'
    elif 'score' in df.columns:
        df['sentiment'] = df['score'].apply(score_to_sentiment)
        tcol = 'text'
    elif 'sentiment' in df.columns:
        tcol = 'text' if 'text' in df.columns else df.columns[0]
    else:
        raise ValueError("Need Score/score/sentiment column")
    df['clean_text'] = df[tcol].apply(preprocess)
    df = df[df['clean_text'].str.len() > 0]
    mx = df['sentiment'].value_counts().max()
    parts = []
    for cls in LABELS:
        sub = df[df['sentiment'] == cls]
        if len(sub) < mx:
            sub = resample(sub, replace=True, n_samples=mx, random_state=42)
        parts.append(sub)
    return pd.concat(parts)

# ═══════════════════════════════════
# FEATURE EXTRACTION
# ═══════════════════════════════════
def extract_features(X_train, X_test):
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    bow = CountVectorizer(max_features=10000, ngram_range=(1, 2))
    feats = {
        'tfidf': (tfidf.fit_transform(X_train), tfidf.transform(X_test)),
        'bow': (bow.fit_transform(X_train), bow.transform(X_test)),
    }
    vectorizers['tfidf'] = tfidf
    vectorizers['bow'] = bow
    return feats

# ═══════════════════════════════════
# TRAIN ALL MODELS
# ═══════════════════════════════════
def train_all(feats, y_train, y_test):
    configs = {
        'logistic_regression': LogisticRegression(max_iter=1000, multi_class='multinomial'),
        'naive_bayes': MultinomialNB(alpha=1.0),
        'random_forest': RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42),
        'feedforward_nn': MLPClassifier(hidden_layer_sizes=(128,), activation='relu',
                                         max_iter=200, random_state=42, early_stopping=True),
    }
    results = {}
    for vname, (Xtr, Xte) in feats.items():
        for mname, mdl in configs.items():
            key = f"{mname}_{vname}"
            t0 = time.time()
            m = type(mdl)(**mdl.get_params())
            m.fit(Xtr, y_train)
            dt = time.time() - t0
            y_pred = m.predict(Xte)
            proba = m.predict_proba(Xte) if hasattr(m, 'predict_proba') else None
            avg_conf = float(np.max(proba, axis=1).mean()) if proba is not None else 1.0
            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred, labels=LABELS).tolist()
            report = classification_report(y_test, y_pred, labels=LABELS, output_dict=True)
            models[key] = m
            results[key] = {
                'accuracy': round(acc, 4), 'train_time': round(dt, 2),
                'avg_confidence': round(avg_conf, 4),
                'confusion_matrix': cm, 'classification_report': report,
                'vectorizer': vname, 'model': mname,
            }
    return results

# ═══════════════════════════════════
# ROBERTA (optional — needs torch)
# ═══════════════════════════════════
def load_transformer():
    global transformer_pipe
    try:
        from transformers import pipeline
        transformer_pipe = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
            top_k=None)
        print("RoBERTa loaded")
        return True
    except Exception as e:
        print(f"RoBERTa not available ({e}). Using traditional ML only.")
        return False

def predict_transformer(texts):
    if not transformer_pipe: return None
    lmap = {'negative':'negative','NEGATIVE':'negative','neutral':'neutral',
            'NEUTRAL':'neutral','positive':'positive','POSITIVE':'positive',
            'LABEL_0':'negative','LABEL_1':'neutral','LABEL_2':'positive'}
    out = []
    for text in texts:
        try:
            preds = transformer_pipe(text[:512])
            if isinstance(preds[0], list): preds = preds[0]
            best = max(preds, key=lambda x: x['score'])
            scores = {lmap.get(p['label'], p['label']): round(p['score'], 4) for p in preds}
            out.append({'sentiment': lmap.get(best['label'], 'neutral'),
                        'confidence': round(best['score'], 4), 'all_scores': scores})
        except:
            out.append({'sentiment':'neutral','confidence':0.0,'all_scores':{}})
    return out

# ═══════════════════════════════════
# SINGLE PREDICTION
# ═══════════════════════════════════
def predict_one(text, model_key):
    clean = preprocess(text)
    vtype = 'tfidf' if 'tfidf' in model_key else 'bow'
    vec, mdl = vectorizers.get(vtype), models.get(model_key)
    if not vec or not mdl: return None
    X = vec.transform([clean])
    pred = mdl.predict(X)[0]
    proba = mdl.predict_proba(X)[0] if hasattr(mdl, 'predict_proba') else None
    conf = float(np.max(proba)) if proba is not None else 1.0
    scores = {}
    if proba is not None:
        scores = {LABELS[i]: round(float(proba[i]), 4) for i in range(min(len(LABELS), len(proba)))}
    return {'sentiment': pred, 'confidence': round(conf, 4), 'all_scores': scores, 'model': model_key}

# ═══════════════════════════════════
# SCRAPING
# ═══════════════════════════════════
def scrape_reviews(url, source='amazon', pages=3):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    reviews = []
    for pg in range(1, pages + 1):
        try:
            full_url = f"{url}&pageNumber={pg}" if source == 'amazon' else url
            resp = http_req.get(full_url, headers=headers, timeout=10)
            soup = BeautifulSoup(resp.text, 'html.parser')
            if source == 'amazon':
                for div in soup.find_all('div', {'data-hook': 'review'}):
                    body = div.find('span', {'data-hook': 'review-body'})
                    if body: reviews.append({'text': body.get_text(strip=True), 'source': 'amazon'})
            time.sleep(1)
        except Exception as e:
            print(f"Scrape error: {e}")
    return reviews

# ═══════════════════════════════════
# API ROUTES
# ═══════════════════════════════════
@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'models': list(models.keys()),
                    'transformer': transformer_pipe is not None,
                    'timestamp': datetime.now().isoformat()})

@app.route('/api/train', methods=['POST'])
def train():
    try:
        if 'file' in request.files:
            f = request.files['file']
            path = f'/tmp/{f.filename}'; f.save(path)
            df = load_data(filepath=path)
        elif request.json and 'data' in request.json:
            df = load_data(data=request.json['data'])
        else:
            return jsonify({'error': 'No data provided'}), 400
        X_train, X_test, y_train, y_test = train_test_split(
            df['clean_text'], df['sentiment'], test_size=0.3, random_state=42, stratify=df['sentiment'])
        feats = extract_features(X_train, X_test)
        results = train_all(feats, y_train, y_test)
        return jsonify({'status': 'ok', 'results': results, 'dataset_size': len(df)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    if not text: return jsonify({'error': 'No text'}), 400
    all_preds = {}
    for key in models:
        r = predict_one(text, key)
        if r: all_preds[key] = r
    if transformer_pipe:
        t = predict_transformer([text])
        if t: all_preds['roberta_transformer'] = t[0]
    sents = [r['sentiment'] for r in all_preds.values()]
    ens = Counter(sents).most_common(1)[0][0] if sents else 'neutral'
    avg_c = float(np.mean([r['confidence'] for r in all_preds.values()])) if all_preds else 0
    return jsonify({'text': text, 'ensemble': {'sentiment': ens, 'confidence': round(avg_c, 4)},
                    'models': all_preds, 'timestamp': datetime.now().isoformat()})

@app.route('/api/scrape', methods=['POST'])
def scrape():
    data = request.json
    url = data.get('url', '')
    source = data.get('source', 'amazon')
    if not url: return jsonify({'error': 'No URL'}), 400
    reviews = scrape_reviews(url, source)
    analyzed = []
    for rev in reviews:
        preds = {}
        for key in models:
            r = predict_one(rev['text'], key)
            if r: preds[key] = r
        if transformer_pipe:
            t = predict_transformer([rev['text']])
            if t: preds['roberta_transformer'] = t[0]
        rev['predictions'] = preds
        analyzed.append(rev)
    return jsonify({'reviews': analyzed, 'count': len(analyzed)})

@app.route('/api/datasets')
def datasets():
    return jsonify({'datasets': [
        {'name': 'Amazon Fine Food Reviews', 'size': '568,454', 'src': 'Kaggle', 'primary': True},
        {'name': 'Amazon Customer Reviews', 'size': '130M+', 'src': 'Kaggle'},
        {'name': 'Yelp Open Dataset', 'size': '6.9M', 'src': 'Kaggle'},
        {'name': 'IMDB Movie Reviews', 'size': '50,000', 'src': 'Stanford'},
        {'name': 'Twitter Sentiment140', 'size': '1.6M', 'src': 'Kaggle'},
    ]})

# ═══════ Serve React ═══════
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if app.static_folder and os.path.exists(os.path.join(app.static_folder, path or 'index.html')):
        return send_from_directory(app.static_folder, path if path else 'index.html')
    if app.static_folder and os.path.exists(os.path.join(app.static_folder, 'index.html')):
        return send_from_directory(app.static_folder, 'index.html')
    return jsonify({'msg': 'API running', 'endpoints': ['/api/health','/api/train','/api/predict','/api/scrape','/api/datasets']})

if __name__ == '__main__':
    load_transformer()
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
