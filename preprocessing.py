import os
import pandas as pd
import re
import nltk

# Download NLTK resources automatically if missing
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

# Define STOPWORDS clearly so sentiment.py can import it without error
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

def clean_dataframe(df):
    df = df.copy()
    if 'rate' in df.columns:
        df['rate'] = df['rate'].astype(str).apply(lambda x: x.split('/')[0].strip() if '/' in x else x)
        df['rate'] = pd.to_numeric(df['rate'], errors='coerce')
    if 'approx_cost(for two people)' in df.columns:
        df['approx_cost'] = df['approx_cost(for two people)'].astype(str).str.replace(',', '')
        df['approx_cost'] = pd.to_numeric(df['approx_cost'], errors='coerce')
    return df

def build_corpus_text(df):
    text_cols = []
    for col in ['cuisines', 'rest_type', 'dish_liked', 'reviews_list']:
        if col in df.columns:
            text_cols.append(df[col].astype(str))
    if text_cols:
        combined = text_cols[0]
        for c in text_cols[1:]:
            combined = combined + " " + c
        return combined.apply(clean_text)
    return pd.Series([""] * len(df))

# Functions aliases for smooth importing across app.py
clean_data = clean_dataframe
build_corpus = build_corpus_text