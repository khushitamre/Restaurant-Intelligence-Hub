"""
utils.py
---------
Shared utilities: data loading, caching helpers, and reusable
Plotly chart builders used across the Streamlit app.

Keeping chart construction here (instead of inline in app.py) means
app.py stays focused on layout/UX, and every chart follows one
consistent visual language -- this is what makes a Streamlit app look
like a product instead of a notebook dump.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------------------------------------------------
# Brand palette -- reused by every chart so the whole app feels designed,
# not default-matplotlib. Dark glassmorphism base + one accent gradient.
# ----------------------------------------------------------------------
COLORS = {
    "bg": "#0e1117",
    "card": "rgba(255,255,255,0.05)",
    "accent": "#22d3ee",       # cyan
    "accent2": "#a78bfa",      # violet
    "positive": "#34d399",     # green
    "negative": "#f87171",     # red
    "neutral": "#fbbf24",      # amber
    "text": "#e5e7eb",
}

PLOTLY_TEMPLATE = "plotly_dark"


def _style_fig(fig, title=None, height=380):
    """Apply consistent dark-glass styling to any Plotly figure."""
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"], family="Inter, sans-serif"),
        title=dict(text=title, x=0.02, font=dict(size=16)) if title else None,
        height=height,
        margin=dict(l=30, r=20, t=50, b=30),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def load_data(path: str) -> pd.DataFrame:
    """Load the raw Zomato CSV. Cached at the call site in app.py via
    @st.cache_data -- kept as a plain function here so it's testable
    outside Streamlit too."""
    return pd.read_csv(path)


# ----------------------------------------------------------------------
# Chart builders
# ----------------------------------------------------------------------

def chart_rating_distribution(df: pd.DataFrame):
    fig = px.histogram(
        df, x="rating_clean", nbins=25,
        color_discrete_sequence=[COLORS["accent"]],
    )
    fig.update_traces(marker_line_width=0)
    return _style_fig(fig, "Rating Distribution")


def chart_cost_distribution(df: pd.DataFrame):
    fig = px.histogram(
        df, x="cost_clean", nbins=30,
        color_discrete_sequence=[COLORS["accent2"]],
    )
    return _style_fig(fig, "Cost for Two — Distribution (₹)")


def chart_top_cuisines(df: pd.DataFrame, top_n: int = 12):
    all_cuisines = df["cuisines"].dropna().str.split(",").explode().str.strip()
    counts = all_cuisines.value_counts().head(top_n).reset_index()
    counts.columns = ["cuisine", "count"]
    fig = px.bar(
        counts.sort_values("count"), x="count", y="cuisine", orientation="h",
        color="count", color_continuous_scale=["#164e63", COLORS["accent"]],
    )
    fig.update_layout(coloraxis_showscale=False)
    return _style_fig(fig, f"Top {top_n} Cuisines")


def chart_top_locations(df: pd.DataFrame, top_n: int = 12):
    counts = df["location"].value_counts().head(top_n).reset_index()
    counts.columns = ["location", "count"]
    fig = px.bar(
        counts.sort_values("count"), x="count", y="location", orientation="h",
        color="count", color_continuous_scale=["#4c1d95", COLORS["accent2"]],
    )
    fig.update_layout(coloraxis_showscale=False)
    return _style_fig(fig, f"Top {top_n} Locations by Restaurant Count")


def chart_restaurant_type_count(df: pd.DataFrame):
    counts = df["rest_type"].value_counts().reset_index()
    counts.columns = ["type", "count"]
    fig = px.pie(
        counts, names="type", values="count", hole=0.55,
        color_discrete_sequence=px.colors.sequential.Tealgrn_r,
    )
    return _style_fig(fig, "Restaurant Type Mix")


def chart_word_frequency(word_freq: dict, title="Top Words", color=None):
    items = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
    words, counts = zip(*items) if items else ([], [])
    fig = px.bar(
        x=list(counts), y=list(words), orientation="h",
        color_discrete_sequence=[color or COLORS["accent"]],
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return _style_fig(fig, title)


def chart_sentiment_distribution(sentiment_counts: pd.Series):
    color_map = {"Positive": COLORS["positive"], "Negative": COLORS["negative"], "Neutral": COLORS["neutral"]}
    fig = px.pie(
        names=sentiment_counts.index, values=sentiment_counts.values, hole=0.5,
        color=sentiment_counts.index, color_discrete_map=color_map,
    )
    return _style_fig(fig, "Review Sentiment Distribution")


def chart_confidence_gauge(score: float):
    """0-100 gauge showing recommendation confidence for the AI explanation panel."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score, 1),
        number={"suffix": "%", "font": {"size": 28, "color": COLORS["text"]}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": COLORS["text"]},
            "bar": {"color": COLORS["accent"]},
            "bgcolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [0, 40], "color": "rgba(248,113,113,0.25)"},
                {"range": [40, 70], "color": "rgba(251,191,36,0.25)"},
                {"range": [70, 100], "color": "rgba(52,211,153,0.25)"},
            ],
        },
    ))
    return _style_fig(fig, height=220)
