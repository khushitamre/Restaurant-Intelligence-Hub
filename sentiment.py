import pandas as pd
import re
from preprocessing import clean_text, STOPWORDS

def analyze_restaurant_reviews(df):
    """Fallback-safe review sentiment analysis helper"""
    if 'reviews_list' not in df.columns:
        df['sentiment'] = 'Neutral'
        return df
    
    # Simple rule-based scoring if VADER/TextBlob isn't initialized
    def quick_sentiment(text):
        if not isinstance(text, str):
            return 'Neutral'
        text_lower = text.lower()
        pos_words = ['good', 'great', 'excellent', 'delicious', 'amazing', 'best', 'love', 'nice']
        neg_words = ['bad', 'worst', 'horrible', 'terrible', 'slow', 'dirty', 'waste', 'poor']
        
        pos_score = sum(1 for w in pos_words if w in text_lower)
        neg_score = sum(1 for w in neg_words if w in text_lower)
        
        if pos_score > neg_score:
            return 'Positive'
        elif neg_score > pos_score:
            return 'Negative'
        return 'Neutral'

    df['sentiment'] = df['reviews_list'].apply(quick_sentiment)
    return df

def summarize_reviews(df, restaurant_name=None):
    """Provides key sentiment metrics for app dashboard"""
    if restaurant_name and 'name' in df.columns:
        sub_df = df[df['name'] == restaurant_name]
    else:
        sub_df = df
        
    if 'sentiment' not in sub_df.columns:
        sub_df = analyze_restaurant_reviews(sub_df)
        
    counts = sub_df['sentiment'].value_counts().to_dict()
    total = len(sub_df) if len(sub_df) > 0 else 1
    
    return {
        'total_reviews': total,
        'positive_pct': round((counts.get('Positive', 0) / total) * 100, 1),
        'negative_pct': round((counts.get('Negative', 0) / total) * 100, 1),
        'neutral_pct': round((counts.get('Neutral', 0) / total) * 100, 1)
    }