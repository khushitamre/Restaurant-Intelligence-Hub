"""
app.py
-------
FoodSense OS // Neural Intelligence Suite (Multi-Dashboard Edition)
Compact Single-Screen View Dashboards for Portfolio Presentation.
"""

import sys
import os
import re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from preprocessing import clean_dataframe, build_corpus_text
from recommendation import RestaurantRecommender
from sentiment import analyze_restaurant_reviews, summarize_reviews
import utils
from utils import load_data

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "zomato.csv")

# Streamlit Page Setup
st.set_page_config(
    page_title="FoodSense OS // Neural Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ULTRA-PREMIUM COMPACT CSS (NO SCROLL FIT)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=JetBrains+Mono:wght@500;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
    .stApp { background: radial-gradient(circle at 50% 0%, #131722 0%, #080a0e 100%); color: #cbd5e1; }

    /* Top HUD Header */
    .hud-bar {
        background: rgba(18, 22, 31, 0.8);
        border: 1px solid rgba(251, 191, 36, 0.25);
        border-radius: 10px;
        padding: 8px 18px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
        backdrop-filter: blur(12px);
    }
    .hud-tag { font-family: 'JetBrains Mono'; font-size: 0.7rem; color: #fbbf24; letter-spacing: 1px; }

    /* Compact Screen Cards */
    .fs-card {
        background: rgba(20, 25, 36, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 8px;
        backdrop-filter: blur(10px);
    }

    .kpi-box {
        background: rgba(251, 191, 36, 0.06);
        border: 1px solid rgba(251, 191, 36, 0.25);
        border-radius: 8px;
        padding: 10px 14px;
        text-align: center;
    }
    .kpi-val { font-family: 'JetBrains Mono'; font-size: 1.5rem; font-weight: 800; color: #fbbf24; }
    .kpi-lbl { font-size: 0.68rem; text-transform: uppercase; color: #64748b; letter-spacing: 1px; }

    .tag-gold {
        background: rgba(251, 191, 36, 0.12);
        color: #fbbf24;
        border: 1px solid rgba(251, 191, 36, 0.3);
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-family: 'JetBrains Mono';
        margin-right: 4px;
    }

    /* Streamlit Padding Compression for Perfect Screenshots */
    .block-container { padding-top: 1.2rem !important; padding-bottom: 0.5rem !important; }
    
    section[data-testid="stSidebar"] {
        background-color: #080a0e !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
    }

    .stButton>button {
        background: linear-gradient(135deg, #fbbf24 0%, #d97706 100%) !important;
        color: #000 !important; font-weight: 800 !important; border: none !important;
        border-radius: 6px !important; padding: 6px 16px !important;
    }

    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# DATA ENGINE WITH SAFE SPEED LOAD
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_clean_data():
    raw = load_data(DATA_PATH)
    if len(raw) > 1500: # Fast execution limit
        raw = raw.head(1500)
    df = clean_dataframe(raw)
    corpus = build_corpus_text(df)
    return df, corpus

@st.cache_resource(show_spinner=False)
def get_recommender(_df, _corpus):
    return RestaurantRecommender(_df, _corpus)

df, corpus = get_clean_data()
recommender = get_recommender(df, corpus)

def safe_render_chart(func, data, height=220):
    try:
        fig = func(data)
        if fig is not None:
            fig.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=height)
            st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass

# ----------------------------------------------------------------------
# EXECUTIVE HUD STATUS BAR
# ----------------------------------------------------------------------
st.markdown(f"""
    <div class="hud-bar">
        <div>
            <span class="hud-tag">FOODSENSE ENTERPRISE OS // v4.2</span>
            <span style="font-weight:800; color:#f8fafc; font-size:1.1rem; margin-left:12px;">Neural Intelligence Hub</span>
        </div>
        <div>
            <span class="tag-gold">STATUS: ONLINE</span>
            <span class="tag-gold">INDEXED: {len(df):,} VENUES</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# SIDEBAR NAVIGATION (4 DISTINCT DASHBOARDS)
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style="padding: 5px 0;">
            <div style="font-family: 'JetBrains Mono'; font-size: 0.65rem; color: #fbbf24;">SELECT DASHBOARD VIEW</div>
            <div style="font-size: 1.2rem; font-weight: 800; color: #f8fafc;">CONTROL <span style="color:#fbbf24;">PANEL</span></div>
        </div>
    """, unsafe_allow_html=True)
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "📊 Macro Market Overview",
            "🔮 Predictive Value Engine",
            "🤖 Vector Matcher (Recommendations)",
            "💬 Dine-In Sentiment Pulse"
        ],
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("##### 🎛️ GLOBAL FILTERS")
    f_loc = st.selectbox("Target Market", ["All Regions"] + sorted(df["location"].dropna().unique().tolist()))
    f_budget = st.slider("Budget Limit (₹)", 100, int(df["cost_clean"].max()), 1500, step=50)


# ======================================================================
# DASHBOARD 1: MACRO MARKET OVERVIEW (Where Your Orders Land)
# ======================================================================
if page == "📊 Macro Market Overview":
    st.markdown("<div style='font-size:0.85rem; font-weight:800; color:#fbbf24; margin-bottom:8px;'>DASHBOARD 01 // MACRO DENSITY & MARKET TRENDS</div>", unsafe_allow_html=True)

    # Top KPI Row
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f'<div class="kpi-box"><div class="kpi-val">{len(df):,}</div><div class="kpi-lbl">VENUES INDEXED</div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="kpi-box"><div class="kpi-val">{df["rating_clean"].mean():.2f} ★</div><div class="kpi-lbl">CORPUS MEAN SCORE</div></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="kpi-box"><div class="kpi-val">₹{df["cost_clean"].mean():,.0f}</div><div class="kpi-lbl">AVG SPEND / TWO</div></div>', unsafe_allow_html=True)
    k4.markdown(f'<div class="kpi-box"><div class="kpi-val">{df["review_count"].sum():,}</div><div class="kpi-lbl">NLP TOKENS</div></div>', unsafe_allow_html=True)

    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="fs-card"><b>📍 Where Your Orders Land // Location Density</b>', unsafe_allow_html=True)
        safe_render_chart(getattr(utils, 'chart_top_locations', None), df, height=190)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="fs-card"><b>⭐ Rating Horizon Distribution</b>', unsafe_allow_html=True)
        safe_render_chart(getattr(utils, 'chart_rating_distribution', None), df, height=190)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="fs-card"><b>💰 Dining Spend Spectrum (Cost for Two)</b>', unsafe_allow_html=True)
        safe_render_chart(getattr(utils, 'chart_cost_distribution', None), df, height=190)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="fs-card"><b>🍕 Top Cuisine Share Matrix</b>', unsafe_allow_html=True)
        safe_render_chart(getattr(utils, 'chart_top_cuisines', None), df, height=190)
        st.markdown('</div>', unsafe_allow_html=True)


# ======================================================================
# DASHBOARD 2: PREDICTIVE VALUE ENGINE
# ======================================================================
elif page == "🔮 Predictive Value Engine":
    st.markdown("<div style='font-size:0.85rem; font-weight:800; color:#fbbf24; margin-bottom:8px;'>DASHBOARD 02 // PREDICTIVE BUDGET & VALUE SIMULATOR</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.markdown('<div class="fs-card">', unsafe_allow_html=True)
        st.markdown("##### 📥 Simulation Parameters")
        sim_loc = st.selectbox("Target Neighborhood", sorted(df["location"].dropna().unique()))
        sim_type = st.selectbox("Establishment Format", sorted(df["rest_type"].dropna().unique()))
        sim_rating = st.slider("Target Service Rating", 1.0, 5.0, 4.2, step=0.1)
        sim_cuisines = st.slider("Cuisine Variety Scale", 1, 8, 3)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        loc_avg = df[df['location'] == sim_loc]['cost_clean'].mean()
        loc_avg = loc_avg if not np.isnan(loc_avg) else df['cost_clean'].mean()
        predicted_cost = round(loc_avg * (0.8 + (sim_rating * 0.12) + (sim_cuisines * 0.04)), -1)
        value_score = round(min(100, max(20, (sim_rating / (predicted_cost / 500)) * 25)), 1)

        st.markdown('<div class="fs-card">', unsafe_allow_html=True)
        st.markdown("<div class='kpi-lbl'>ESTIMATED COST FOR TWO</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-family: JetBrains Mono; font-size: 2.2rem; font-weight: 800; color: #fbbf24;'>₹{predicted_cost:,.0f}</div>", unsafe_allow_html=True)
        st.caption(f"Estimated model output for {sim_type} in {sim_loc}")

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value_score,
            title={'text': "Value-for-Money Score", 'font': {'color': '#94a3b8', 'size': 13}},
            number={'font': {'color': '#fbbf24', 'family': 'JetBrains Mono'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': "#64748b"},
                'bar': {'color': "#fbbf24"},
                'bgcolor': "#080a0e",
                'steps': [{'range': [0, 50], 'color': 'rgba(251, 191, 36, 0.1)'}, {'range': [50, 100], 'color': 'rgba(251, 191, 36, 0.25)'}],
            }
        ))
        fig_gauge.update_layout(height=160, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ======================================================================
# DASHBOARD 3: NEURAL VECTOR MATCHER (Recommendations)
# ======================================================================
elif page == "🤖 Vector Matcher (Recommendations)":
    st.markdown("<div style='font-size:0.85rem; font-weight:800; color:#fbbf24; margin-bottom:8px;'>DASHBOARD 03 // VECTOR COSINE SIMILARITY ENGINE</div>", unsafe_allow_html=True)

    c_select, c_btn = st.columns([3, 1])
    with c_select:
        seed_venue = st.selectbox("Select Benchmark Restaurant", sorted(df["name"].unique()))
    with c_btn:
        run_rec = st.button("⚡ Match Vectors", use_container_width=True)

    recs = recommender.recommend(seed_venue, top_n=4)
    st.write("")
    
    col_a, col_b = st.columns(2)
    for idx, item in enumerate(recs):
        target_col = col_a if idx % 2 == 0 else col_b
        with target_col:
            st.markdown(f"""
                <div class="fs-card" style="border-left: 3px solid #fbbf24;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-weight:700; color:#f8fafc; font-size:1.05rem;">{item['name']}</span>
                        <span style="font-family:'JetBrains Mono'; color:#fbbf24; font-weight:800; font-size:1.1rem;">{item['confidence']}%</span>
                    </div>
                    <div style="font-size:0.78rem; color:#64748b; margin-top:2px;">{item['location']} • ₹{item['cost_for_two']:.0f} for two</div>
                    <div style="margin-top:6px;"><span class="tag-gold">{item['cuisines']}</span></div>
                    <div style="font-size:0.78rem; color:#94a3b8; margin-top:8px; background:rgba(0,0,0,0.25); padding:6px 10px; border-radius:6px;">
                        💡 {item['explanation']}
                    </div>
                </div>
            """, unsafe_allow_html=True)


# ======================================================================
# DASHBOARD 4: DINE-IN SENTIMENT PULSE (NLP Review Audit)
# ======================================================================
elif page == "💬 Dine-In Sentiment Pulse":
    st.markdown("<div style='font-size:0.85rem; font-weight:800; color:#fbbf24; margin-bottom:8px;'>DASHBOARD 04 // NLP REVIEW SENTIMENT PULSE</div>", unsafe_allow_html=True)

    audit_venue = st.selectbox("Target Restaurant Audit", sorted(df["name"].unique()))
    row_data = df[df["name"] == audit_venue].iloc[0]
    sent_res = analyze_restaurant_reviews(row_data["reviews_parsed"])

    c1, c2 = st.columns([1, 1.2])

    with c1:
        st.markdown('<div class="fs-card">', unsafe_allow_html=True)
        st.markdown(f"<h4 style='color:#fbbf24; margin:0;'>{row_data['name']}</h4>", unsafe_allow_html=True)
        st.markdown(f"<span style='color:#64748b; font-size:0.8rem;'>{row_data['location']} • {row_data['cuisines']}</span>", unsafe_allow_html=True)
        
        st.write("")
        sk1, sk2 = st.columns(2)
        sk1.markdown(f'<div class="kpi-box"><div class="kpi-val" style="color:#34d399;">{sent_res["positive_pct"]}%</div><div class="kpi-lbl">POSITIVE</div></div>', unsafe_allow_html=True)
        sk2.markdown(f'<div class="kpi-box"><div class="kpi-val" style="color:#f87171;">{sent_res["negative_pct"]}%</div><div class="kpi-lbl">NEGATIVE</div></div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="fs-card">', unsafe_allow_html=True)
        st.markdown("<div class='kpi-lbl' style='color:#fbbf24;'>AI NLP REVIEW SUMMARY</div>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:0.85rem; color:#cbd5e1; margin-top:6px; line-height:1.5;'>\"{summarize_reviews(row_data['reviews_parsed'])}\"</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)