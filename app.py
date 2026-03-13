"""
AI-Based Intelligent Customer Feedback Analyzer
Flask API — 8 Traditional ML models + optional RoBERTa
Real scraping via Outscraper API (free tier: 100 requests/month)
CSV upload support for Amazon/Kaggle datasets
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
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=False)

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

@app.route("/api/<path:path>", methods=["OPTIONS"])
def options_handler(path):
    return "", 204

models = {}
vectorizers = {}
transformer_pipe = None
LABELS = ['negative', 'neutral', 'positive']
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Outscraper API key (free: 100 req/month at outscraper.com)
# Set as environment variable in Railway: OUTSCRAPER_API_KEY=your_key
OUTSCRAPER_KEY = os.environ.get('OUTSCRAPER_API_KEY', '')


# ═══════════════════════════════════
# PREPROCESSING (Paper Section III)
# ═══════════════════════════════════
def preprocess(text):
    if not isinstance(text, str):
        return ""
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
    if score <= 2:
        return 'negative'
    if score == 3:
        return 'neutral'
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
    elif 'review' in df.columns.str.lower():
        tcol = [c for c in df.columns if 'review' in c.lower()][0]
        if 'rating' in df.columns.str.lower():
            rcol = [c for c in df.columns if 'rating' in c.lower()][0]
            df['sentiment'] = df[rcol].apply(lambda x: score_to_sentiment(int(x)) if str(x).isdigit() else 'neutral')
        else:
            df['sentiment'] = 'neutral'
    else:
        raise ValueError("Need Score/score/sentiment/review column")
    df['clean_text'] = df[tcol].apply(preprocess)
    df = df[df['clean_text'].str.len() > 0]
    mx = df['sentiment'].value_counts().max()
    parts = []
    for cls in LABELS:
        sub = df[df['sentiment'] == cls]
        if len(sub) > 0 and len(sub) < mx:
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
        print(f"RoBERTa not available ({e}). Traditional ML only.")
        return False


def predict_transformer(texts):
    if not transformer_pipe:
        return None
    lmap = {'negative': 'negative', 'NEGATIVE': 'negative', 'neutral': 'neutral',
            'NEUTRAL': 'neutral', 'positive': 'positive', 'POSITIVE': 'positive',
            'LABEL_0': 'negative', 'LABEL_1': 'neutral', 'LABEL_2': 'positive'}
    out = []
    for text in texts:
        try:
            preds = transformer_pipe(text[:512])
            if isinstance(preds[0], list):
                preds = preds[0]
            best = max(preds, key=lambda x: x['score'])
            scores = {lmap.get(p['label'], p['label']): round(p['score'], 4) for p in preds}
            out.append({'sentiment': lmap.get(best['label'], 'neutral'),
                        'confidence': round(best['score'], 4), 'all_scores': scores})
        except:
            out.append({'sentiment': 'neutral', 'confidence': 0.0, 'all_scores': {}})
    return out


# ═══════════════════════════════════
# SINGLE PREDICTION
# ═══════════════════════════════════
def predict_one(text, model_key):
    clean = preprocess(text)
    vtype = 'tfidf' if 'tfidf' in model_key else 'bow'
    vec, mdl = vectorizers.get(vtype), models.get(model_key)
    if not vec or not mdl:
        return None
    X = vec.transform([clean])
    pred = mdl.predict(X)[0]
    proba = mdl.predict_proba(X)[0] if hasattr(mdl, 'predict_proba') else None
    conf = float(np.max(proba)) if proba is not None else 1.0
    scores = {}
    if proba is not None:
        scores = {LABELS[i]: round(float(proba[i]), 4) for i in range(min(len(LABELS), len(proba)))}
    return {'sentiment': pred, 'confidence': round(conf, 4), 'all_scores': scores, 'model': model_key}


def predict_all_models(text):
    """Run text through all available models"""
    all_preds = {}
    for key in models:
        r = predict_one(text, key)
        if r:
            all_preds[key] = r
    if transformer_pipe:
        t = predict_transformer([text])
        if t:
            all_preds['roberta_transformer'] = t[0]
    return all_preds


# ═══════════════════════════════════
# REAL SCRAPING — Multiple Methods
# ═══════════════════════════════════

def scrape_outscraper(query, limit=20):
    """
    Scrape Google Reviews using Outscraper API (free: 100 req/month)
    Sign up at https://outscraper.com — get API key — set as env var
    """
    if not OUTSCRAPER_KEY:
        return None, "Outscraper API key not set. Set OUTSCRAPER_API_KEY env variable in Railway."

    try:
        resp = http_req.get(
            'https://api.app.outscraper.com/maps/reviews-v3',
            params={'query': query, 'reviewsLimit': limit, 'language': 'en', 'sort': 'newest'},
            headers={'X-API-KEY': OUTSCRAPER_KEY},
            timeout=30
        )
        data = resp.json()
        if 'data' not in data or not data['data']:
            return [], "No reviews found for this query"

        reviews = []
        for place in data['data']:
            place_name = place.get('name', 'Unknown')
            for rev in place.get('reviews_data', []):
                reviews.append({
                    'text': rev.get('review_text', ''),
                    'rating': rev.get('review_rating', 3),
                    'author': rev.get('author_title', 'Anonymous'),
                    'date': rev.get('review_datetime_utc', ''),
                    'source': f'google:{place_name}',
                })
        reviews = [r for r in reviews if r['text'] and len(r['text']) > 10]
        return reviews, None
    except Exception as e:
        return None, str(e)


def scrape_trustpilot(company, pages=2):
    """Scrape Trustpilot reviews (works without API key)"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                       'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    reviews = []
    for pg in range(1, pages + 1):
        try:
            url = f'https://www.trustpilot.com/review/{company}?page={pg}'
            resp = http_req.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(resp.text, 'html.parser')

            # Find review cards
            for script in soup.find_all('script', type='application/ld+json'):
                try:
                    ld = json.loads(script.string)
                    if isinstance(ld, dict) and ld.get('@type') == 'LocalBusiness':
                        for rev in ld.get('review', []):
                            reviews.append({
                                'text': rev.get('reviewBody', ''),
                                'rating': int(rev.get('reviewRating', {}).get('ratingValue', 3)),
                                'author': rev.get('author', {}).get('name', 'Anonymous'),
                                'date': rev.get('datePublished', ''),
                                'source': f'trustpilot:{company}',
                            })
                except:
                    pass

            # Fallback: parse HTML directly
            if not reviews:
                for card in soup.find_all('article'):
                    body = card.find('p', {'data-service-review-text-typography': True})
                    if not body:
                        body = card.find('p')
                    if body:
                        txt = body.get_text(strip=True)
                        if len(txt) > 15:
                            reviews.append({
                                'text': txt,
                                'rating': 3,
                                'author': 'Unknown',
                                'date': '',
                                'source': f'trustpilot:{company}',
                            })
            time.sleep(1)
        except Exception as e:
            print(f"Trustpilot scrape error: {e}")
    return reviews


def scrape_amazon_basic(url, pages=2):
    """Basic Amazon review scraping (may be blocked)"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                       'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    reviews = []
    for pg in range(1, pages + 1):
        try:
            page_url = f"{url}&pageNumber={pg}" if '?' in url else f"{url}?pageNumber={pg}"
            resp = http_req.get(page_url, headers=headers, timeout=10)
            soup = BeautifulSoup(resp.text, 'html.parser')
            for div in soup.find_all('div', {'data-hook': 'review'}):
                body = div.find('span', {'data-hook': 'review-body'})
                title = div.find('a', {'data-hook': 'review-title'})
                rating_el = div.find('i', {'data-hook': 'review-star-rating'})

                txt = body.get_text(strip=True) if body else ''
                ttl = title.get_text(strip=True) if title else ''
                rating = 3
                if rating_el:
                    nums = re.findall(r'(\d+)', rating_el.get_text())
                    if nums:
                        rating = int(nums[0])

                if txt and len(txt) > 15:
                    reviews.append({
                        'text': f"{ttl} {txt}".strip(),
                        'rating': rating,
                        'author': 'Amazon Reviewer',
                        'date': '',
                        'source': 'amazon',
                    })
            time.sleep(1)
        except Exception as e:
            print(f"Amazon scrape error: {e}")
    return reviews


# ═══════════════════════════════════
# API ROUTES
# ═══════════════════════════════════
@app.route('/api/health')
def health():
    return jsonify({
        'status': 'ok',
        'models': list(models.keys()),
        'transformer': transformer_pipe is not None,
        'outscraper': bool(OUTSCRAPER_KEY),
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/train', methods=['POST'])
def train():
    try:
        if 'file' in request.files:
            f = request.files['file']
            path = f'/tmp/{f.filename}'
            f.save(path)
            df = load_data(filepath=path)
        elif request.json and 'data' in request.json:
            df = load_data(data=request.json['data'])
        else:
            return jsonify({'error': 'No data provided. Upload a CSV file.'}), 400

        X_train, X_test, y_train, y_test = train_test_split(
            df['clean_text'], df['sentiment'],
            test_size=0.3, random_state=42, stratify=df['sentiment']
        )
        feats = extract_features(X_train, X_test)
        results = train_all(feats, y_train, y_test)
        return jsonify({'status': 'ok', 'results': results, 'dataset_size': len(df)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text'}), 400

    all_preds = predict_all_models(text)
    sents = [r['sentiment'] for r in all_preds.values()]
    ens = Counter(sents).most_common(1)[0][0] if sents else 'neutral'
    avg_c = float(np.mean([r['confidence'] for r in all_preds.values()])) if all_preds else 0

    return jsonify({
        'text': text,
        'ensemble': {'sentiment': ens, 'confidence': round(avg_c, 4)},
        'models': all_preds,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/scrape', methods=['POST'])
def scrape():
    """
    Real scraping endpoint.
    Accepts: { "url": "...", "source": "google|trustpilot|amazon", "limit": 20 }
    For Google: uses Outscraper API (needs OUTSCRAPER_API_KEY env var)
    For Trustpilot: direct scraping (free, no API key needed)
    For Amazon: basic scraping (may be blocked by Amazon)
    """
    data = request.json
    url_or_query = data.get('url', '')
    source = data.get('source', 'google')
    limit = data.get('limit', 20)

    if not url_or_query:
        return jsonify({'error': 'No URL or query provided'}), 400

    reviews = []
    error_msg = None

    if source == 'google':
        reviews, error_msg = scrape_outscraper(url_or_query, limit)
        if reviews is None:
            reviews = []
    elif source == 'trustpilot':
        reviews = scrape_trustpilot(url_or_query)
    elif source == 'amazon':
        reviews = scrape_amazon_basic(url_or_query)
    else:
        return jsonify({'error': f'Unknown source: {source}'}), 400

    # Analyze each review with all models
    analyzed = []
    for rev in reviews:
        preds = predict_all_models(rev['text'])
        # Use ensemble for primary sentiment
        sents = [r['sentiment'] for r in preds.values()]
        ens_sent = Counter(sents).most_common(1)[0][0] if sents else 'neutral'
        avg_conf = float(np.mean([r['confidence'] for r in preds.values()])) if preds else 0

        rev['sentiment'] = ens_sent
        rev['confidence'] = round(avg_conf, 4)
        rev['predictions'] = preds
        analyzed.append(rev)

    result = {
        'reviews': analyzed,
        'count': len(analyzed),
        'source': source,
        'timestamp': datetime.now().isoformat()
    }
    if error_msg:
        result['warning'] = error_msg

    return jsonify(result)


@app.route('/api/upload-reviews', methods=['POST'])
def upload_reviews():
    """
    Upload a CSV file of reviews and analyze them all.
    CSV should have a 'text' or 'review' or 'Text' column.
    Optionally a 'score' or 'rating' or 'Score' column.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    f = request.files['file']
    if not f.filename.endswith('.csv'):
        return jsonify({'error': 'Only CSV files supported'}), 400

    try:
        path = f'/tmp/{f.filename}'
        f.save(path)
        df = pd.read_csv(path)

        # Find text column
        text_col = None
        for col in ['Text', 'text', 'review', 'Review', 'review_body', 'comment', 'Comment']:
            if col in df.columns:
                text_col = col
                break
        if not text_col:
            return jsonify({'error': f'No text column found. Columns: {list(df.columns)}'}), 400

        # Find rating column (optional)
        rating_col = None
        for col in ['Score', 'score', 'rating', 'Rating', 'star_rating', 'stars']:
            if col in df.columns:
                rating_col = col
                break

        df = df.dropna(subset=[text_col])
        # Limit to 100 reviews for performance
        df = df.head(100)

        analyzed = []
        for _, row in df.iterrows():
            text = str(row[text_col])
            if len(text) < 10:
                continue

            rating = int(row[rating_col]) if rating_col and pd.notna(row.get(rating_col)) else 3
            preds = predict_all_models(text)

            sents = [r['sentiment'] for r in preds.values()]
            ens_sent = Counter(sents).most_common(1)[0][0] if sents else 'neutral'
            avg_conf = float(np.mean([r['confidence'] for r in preds.values()])) if preds else 0

            analyzed.append({
                'text': text[:500],
                'rating': rating,
                'sentiment': ens_sent,
                'confidence': round(avg_conf, 4),
                'predictions': preds,
                'source': 'csv_upload',
            })

        return jsonify({
            'reviews': analyzed,
            'count': len(analyzed),
            'total_rows': len(df),
            'source': 'csv_upload',
            'columns_found': list(df.columns),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/datasets')
def datasets():
    return jsonify({'datasets': [
        {'name': 'Amazon Fine Food Reviews', 'size': '568,454', 'src': 'Kaggle', 'primary': True,
         'url': 'https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews'},
        {'name': 'Amazon Customer Reviews', 'size': '130M+', 'src': 'Kaggle',
         'url': 'https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset'},
        {'name': 'Yelp Open Dataset', 'size': '6.9M', 'src': 'Kaggle',
         'url': 'https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset'},
        {'name': 'IMDB Movie Reviews', 'size': '50,000', 'src': 'Stanford',
         'url': 'https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews'},
        {'name': 'Twitter Sentiment140', 'size': '1.6M', 'src': 'Kaggle',
         'url': 'https://www.kaggle.com/datasets/kazanova/sentiment140'},
    ]})


# ═══════ Serve React ═══════
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if app.static_folder and os.path.exists(os.path.join(app.static_folder, path or 'index.html')):
        return send_from_directory(app.static_folder, path if path else 'index.html')
    if app.static_folder and os.path.exists(os.path.join(app.static_folder, 'index.html')):
        return send_from_directory(app.static_folder, 'index.html')
    return jsonify({'msg': 'API running', 'endpoints': [
        '/api/health', '/api/train', '/api/predict', '/api/scrape',
        '/api/upload-reviews', '/api/datasets'
    ]})


if __name__ == '__main__':
    load_transformer()
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting on port {port}")
    print(f"Outscraper API: {'configured' if OUTSCRAPER_KEY else 'not set (Google scraping disabled)'}")
    print(f"Models loaded: {len(models)}")
    app.run(host='0.0.0.0', port=port, debug=False)
