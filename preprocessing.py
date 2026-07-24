import os
import pandas as pd
import re
import nltk

# NLTK resources automatic download
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

STOPWORDS = set(stopwords.words('english'))
stop_words = STOPWORDS

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    filtered = [word for word in tokens if word not in STOPWORDS]
    return " ".join(filtered)

def parse_rate(x):
    """TypeError-safe rate parser"""
    if pd.isna(x):
        return None
    x = str(x).strip()
    if '/' in x:
        x = x.split('/')[0].strip()
    if x in ['NEW', '-', '']:
        return None
    try:
        return float(x)
    except ValueError:
        return None

def parse_cost(x):
    """TypeError-safe cost parser"""
    if pd.isna(x):
        return None
    x = str(x).replace(',', '').strip()
    try:
        return float(x)
    except ValueError:
        return None

def clean_dataframe(df):
    df = df.copy()
    if 'rate' in df.columns:
        df['rate'] = df['rate'].apply(parse_rate)
    if 'approx_cost(for two people)' in df.columns:
        df['approx_cost'] = df['approx_cost(for two people)'].apply(parse_cost)
    elif 'approx_cost' in df.columns:
        df['approx_cost'] = df['approx_cost'].apply(parse_cost)
    return df

def build_corpus_text(df):
    text_cols = []
    for col in ['cuisines', 'rest_type', 'dish_liked', 'reviews_list']:
        if col in df.columns:
            text_cols.append(df[col].fillna('').astype(str))
    if text_cols:
        combined = text_cols[0]
        for c in text_cols[1:]:
            combined = combined + " " + c
        return combined.apply(clean_text)
    return pd.Series([""] * len(df))

clean_data = clean_dataframe
build_corpus = build_corpus_text