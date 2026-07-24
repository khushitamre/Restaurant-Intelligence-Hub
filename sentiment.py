"""
sentiment.py
-------------
Review-level intelligence: sentiment classification, keyword extraction,
extractive review summarization, and frequency analysis (positive/negative
word lists, n-grams) that feed the WordCloud and bar charts in the app.

Sentiment engine: VADER (nltk.sentiment.SentimentIntensityAnalyzer).
VADER is chosen over a from-scratch model deliberately -- it's
lexicon-based, needs no training data or GPU, handles informal review
text (negation, intensifiers, punctuation emphasis) well out of the box,
and is the industry-standard baseline for review/social-text sentiment.
A production v2 would swap this for a fine-tuned transformer
(e.g. distilbert-base-uncased-finetuned-sst-2), noted in README future work.
"""

import re
from collections import Counter
import pandas as pd

from preprocessing import clean_text, STOPWORDS

try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
    _sia = SentimentIntensityAnalyzer()
    VADER_AVAILABLE = True
except Exception:
    VADER_AVAILABLE = False
    _sia = None

# Compact fallback lexicon used only if VADER/nltk truly can't load
# (e.g. fully offline grading environment with no cached lexicon).
_POS_WORDS = {"good", "great", "excellent", "amazing", "love", "best", "delicious",
              "fresh", "friendly", "perfect", "awesome", "tasty", "recommend"}
_NEG_WORDS = {"bad", "worst", "terrible", "slow", "cold", "rude", "disappointing",
              "overpriced", "dirty", "bland", "poor", "wrong", "awful"}


def analyze_sentiment(text: str) -> dict:
    """Returns {'label': 'Positive'|'Negative'|'Neutral', 'compound': float}"""
    if not text or not text.strip():
        return {"label": "Neutral", "compound": 0.0}

    if VADER_AVAILABLE:
        scores = _sia.polarity_scores(text)
        compound = scores["compound"]
    else:
        words = set(re.sub(r"[^a-z\s]", " ", text.lower()).split())
        pos_hits = len(words & _POS_WORDS)
        neg_hits = len(words & _NEG_WORDS)
        total_hits = pos_hits + neg_hits
        compound = (pos_hits - neg_hits) / total_hits if total_hits else 0.0

    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"
    return {"label": label, "compound": round(float(compound), 3)}


def analyze_restaurant_reviews(reviews: list) -> dict:
    """Aggregate sentiment across all reviews for one restaurant."""
    if not reviews:
        return {"label": "Neutral", "compound": 0.0, "positive_pct": 0, "negative_pct": 0, "neutral_pct": 0, "n_reviews": 0}

    results = [analyze_sentiment(r) for r in reviews]
    labels = [r["label"] for r in results]
    n = len(labels)
    counts = Counter(labels)
    avg_compound = sum(r["compound"] for r in results) / n

    overall = "Positive" if avg_compound >= 0.05 else ("Negative" if avg_compound <= -0.05 else "Neutral")
    return {
        "label": overall,
        "compound": round(avg_compound, 3),
        "positive_pct": round(100 * counts.get("Positive", 0) / n, 1),
        "negative_pct": round(100 * counts.get("Negative", 0) / n, 1),
        "neutral_pct": round(100 * counts.get("Neutral", 0) / n, 1),
        "n_reviews": n,
    }


def corpus_sentiment_distribution(df: pd.DataFrame) -> pd.Series:
    """Sentiment label counts across the ENTIRE dataset's reviews --
    powers the app-wide sentiment pie chart."""
    all_labels = []
    for reviews in df["reviews_parsed"]:
        for r in reviews:
            all_labels.append(analyze_sentiment(r)["label"])
    return pd.Series(all_labels).value_counts()


# ----------------------------------------------------------------------
# Keyword extraction (TF-IDF top terms) + frequency word lists
# ----------------------------------------------------------------------

def extract_keywords(text: str, top_n: int = 8) -> list:
    """Simple frequency-based keyword extraction over cleaned text.
    For a single restaurant's review set this is fast and interpretable;
    for corpus-wide keyword ranking, recommendation.py's TF-IDF
    vectorizer already does the heavier statistical version."""
    cleaned = clean_text(text)
    words = [w for w in cleaned.split() if w not in STOPWORDS]
    return [w for w, _ in Counter(words).most_common(top_n)]


def most_frequent_words(df: pd.DataFrame, sentiment_label: str, top_n: int = 25) -> dict:
    """Word frequency across all reviews of a given sentiment class
    (Positive / Negative) -- feeds the WordCloud and bar charts."""
    bag = []
    for reviews in df["reviews_parsed"]:
        for r in reviews:
            if analyze_sentiment(r)["label"] == sentiment_label:
                bag.extend(clean_text(r).split())
    return dict(Counter(bag).most_common(top_n))


def generate_ngrams(text_series: pd.Series, n: int = 2, top_k: int = 20) -> dict:
    """N-gram frequency analysis (bigrams by default) across a series
    of already-cleaned text documents."""
    counter = Counter()
    for doc in text_series.dropna():
        tokens = doc.split()
        grams = zip(*[tokens[i:] for i in range(n)])
        counter.update(" ".join(g) for g in grams)
    return dict(counter.most_common(top_k))


def summarize_reviews(reviews: list, max_sentences: int = 3) -> str:
    """
    Lightweight extractive summarizer: scores each review sentence by
    word-frequency salience (a mini TextRank-style heuristic without the
    graph overhead) and returns the top-scoring sentences. No external
    LLM call required, so the demo works fully offline -- a genuine
    design tradeoff worth stating in interviews (cost/latency vs. an
    LLM-based abstractive summary).
    """
    if not reviews:
        return "No reviews available yet for this restaurant."

    text = " ".join(reviews)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
    if not sentences:
        return "Reviews are too short to summarize."

    word_freq = Counter(clean_text(text).split())
    if not word_freq:
        return " ".join(sentences[:max_sentences])

    max_freq = max(word_freq.values())
    scores = {}
    for s in sentences:
        words = clean_text(s).split()
        if not words:
            continue
        scores[s] = sum(word_freq.get(w, 0) / max_freq for w in words) / len(words)

    top_sentences = sorted(scores, key=scores.get, reverse=True)[:max_sentences]
    # preserve original chronological order for readability
    ordered = [s for s in sentences if s in top_sentences]
    return " ".join(ordered)
