import os
import pandas as pd
import re
import nltk

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
    if pd.isna(x):
        return None
    x = str(x).replace(',', '').strip()
    try:
        return float(x)
    except ValueError:
        return None

def clean_dataframe(df):
    df = df.copy()
    
    # 1. Ratings
    rate_series = None
    if 'rate' in df.columns:
        rate_series = df['rate'].apply(parse_rate)
    elif 'rating_clean' in df.columns:
        rate_series = df['rating_clean'].apply(parse_rate)
        
    if rate_series is not None:
        df['rate'] = rate_series
        df['rating_clean'] = rate_series
    else:
        df['rate'] = 3.5
        df['rating_clean'] = 3.5

    # 2. Cost
    cost_series = None
    if 'approx_cost(for two people)' in df.columns:
        cost_series = df['approx_cost(for two people)'].apply(parse_cost)
    elif 'approx_cost' in df.columns:
        cost_series = df['approx_cost'].apply(parse_cost)
    elif 'cost_clean' in df.columns:
        cost_series = df['cost_clean'].apply(parse_cost)

    if cost_series is not None:
        df['approx_cost'] = cost_series
        df['cost_clean'] = cost_series
    else:
        df['approx_cost'] = 500.0
        df['cost_clean'] = 500.0

    # 3. Reviews & Tokens count
    if 'reviews_list' in df.columns:
        df['reviews_parsed'] = df['reviews_list'].fillna('').astype(str)
        df['review_count'] = df['reviews_list'].fillna('').astype(str).apply(lambda x: len(x.split()))
    else:
        df['reviews_parsed'] = ""
        df['review_count'] = 100

    # Fill default NaNs
    df['rating_clean'] = df['rating_clean'].fillna(3.5)
    df['rate'] = df['rating_clean']
    df['cost_clean'] = df['cost_clean'].fillna(500.0)
    df['approx_cost'] = df['cost_clean']
    df['location'] = df['location'].fillna('Unknown') if 'location' in df.columns else 'Unknown'
    df['rest_type'] = df['rest_type'].fillna('Casual Dining') if 'rest_type' in df.columns else 'Casual Dining'
    df['cuisines'] = df['cuisines'].fillna('North Indian') if 'cuisines' in df.columns else 'North Indian'

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