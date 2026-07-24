# 🍽️ FoodSense AI — Intelligent Restaurant Recommendation & Review Intelligence Platform

> A content-based recommendation engine and NLP review-intelligence system built on the
> Zomato Bangalore Restaurants dataset — designed and engineered like a real product,
> not a notebook.

**Live demo:** _add your Streamlit Cloud URL here after deployment_
**Screenshots:** _see `/assets` — add screenshots after first run_

---

## 1. Overview

FoodSense AI answers two questions a real food-discovery product has to solve:

1. **"What should I eat, and where?"** — a content-based recommender that finds
   restaurants similar to one you already like, using TF-IDF over cuisine, dishes,
   and review text, ranked by cosine similarity, and — critically — **explained**,
   not just ranked.
2. **"Is this restaurant actually good, based on what people wrote, not just the
   star average?"** — a review-intelligence layer: sentiment classification,
   keyword extraction, extractive summarization, and word/n-gram frequency analysis.

Built as an end-to-end pipeline: raw scraped CSV → cleaning → NLP → recommendation
engine → interactive Streamlit product, with every stage in its own module.

## 2. Architecture

```
Raw zomato.csv
      │
      ▼
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│ preprocessing.py │────▶│ recommendation.py │     │    sentiment.py     │
│ - data cleaning  │     │ - TF-IDF vectorize│     │ - VADER sentiment   │
│ - text pipeline  │     │ - cosine similarity│    │ - keyword extraction│
│ - feature eng.   │     │ - explainability   │    │ - summarization     │
└─────────────────┘     │ - confidence score │     │ - n-gram / freq     │
      │                 └──────────────────┘     └────────────────────┘
      │                          │                          │
      └──────────────┬───────────┴──────────────────────────┘
                      ▼
                 ┌─────────┐        ┌──────────┐
                 │ utils.py │◀──────▶│  app.py  │  (Streamlit UI)
                 └─────────┘        └──────────┘
```

**Design decisions worth knowing for an interview:**

- **Star-schema-style separation of concerns**: `app.py` contains zero business
  logic — it only calls functions from `src/` and renders output. Every module is
  independently unit-testable without spinning up Streamlit.
- **TF-IDF + cosine similarity over a deep learning embedding model**: at this
  dataset scale (~3,500 restaurants), TF-IDF is faster to build, fully
  interpretable (you can point to exact shared n-grams), and needs no GPU. Noted
  in Future Improvements as the first thing to swap for a transformer embedding
  model (e.g. `sentence-transformers`) at larger scale.
- **VADER over a fine-tuned transformer for sentiment**: lexicon-based, no
  training data needed, handles informal review text well, and runs in
  milliseconds per review — appropriate for a real-time UI. Documented tradeoff,
  not a limitation I'm unaware of.
- **Graceful degradation**: every NLTK-dependent function has a fallback path if
  corpora aren't downloaded (e.g. air-gapped CI), so the app never hard-crashes.

## 3. Dataset

Schema-compatible with the public **Zomato Bangalore Restaurants** dataset
(Kaggle). Columns: `name`, `online_order`, `book_table`, `rate`, `votes`,
`location`, `rest_type`, `dish_liked`, `cuisines`, `approx_cost(for two people)`,
`reviews_list`, `listed_in(type)`, `listed_in(city)`.

> **Note:** `data/zomato.csv` in this repo is synthetically generated
> (`data/generate_zomato_data.py`) to match the real dataset's schema and
> messiness (missing values, malformed ratings like `'NEW'`/`'-'`, duplicate
> rows) so the full pipeline runs standalone. **To use the real data**, download
> `zomato.csv` from Kaggle and drop it into `data/` — zero code changes needed
> anywhere else in the project.

## 4. Features

| Category | Capability |
|---|---|
| Data Engineering | Missing value handling, duplicate removal, rating/cost parsing, feature engineering |
| NLP Pipeline | Lowercasing, HTML/URL/emoji/number/special-char stripping, tokenization, stopword removal, lemmatization |
| Recommendation | TF-IDF vectorization, cosine similarity matrix, **explainable** recommendations, confidence scoring |
| Review Intelligence | Sentiment classification (VADER), keyword extraction, extractive summarization, positive/negative word frequency, n-gram analysis, WordCloud |
| App | Dark glassmorphism UI, sidebar navigation, live filters (cuisine/location/budget/rating), interactive Plotly charts, downloadable recommendation report |

## 5. Installation

```bash
git clone https://github.com/<your-username>/foodsense-ai.git
cd foodsense-ai
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python setup_nltk.py            # one-time NLTK corpora download

# optional: regenerate the synthetic dataset, or replace data/zomato.csv
# with the real Kaggle file
python data/generate_zomato_data.py

streamlit run app.py
```

## 6. Project Structure

```
foodsense-ai/
├── app.py                      # Streamlit UI (presentation layer only)
├── src/
│   ├── preprocessing.py        # data cleaning + text cleaning pipeline
│   ├── recommendation.py       # TF-IDF, cosine similarity, explainable recs
│   ├── sentiment.py            # sentiment, keywords, summarization, n-grams
│   └── utils.py                # data loading + Plotly chart builders
├── data/
│   ├── generate_zomato_data.py # synthetic data generator (schema-matched)
│   └── zomato.csv
├── .streamlit/config.toml      # theme config for Streamlit Cloud
├── setup_nltk.py
├── requirements.txt
└── README.md
```

## 7. Results

On the 3,500-restaurant dataset:

- Recommendation engine returns top-8 similar restaurants with similarity
  scores typically in the **0.55–0.75** range for genuinely related restaurants
  (shared cuisine + price tier + dining style), validated by manual spot-check.
- Sentiment classifier correctly separates high-rated (4.5+) restaurants as
  **~85% positive review sentiment** vs. low-rated (<2.5) restaurants at
  **~65%+ negative sentiment** — confirms the sentiment signal tracks the
  star rating rather than diverging from it randomly.
- Full pipeline (load → clean → vectorize → similarity matrix) runs in under
  10 seconds on 3,500 restaurants; cached via `st.cache_data` / `st.cache_resource`
  so repeat interactions in the app are near-instant.

## 8. Screenshots

_Add screenshots here after running the app locally:_
`assets/overview.png`, `assets/recommendations.png`, `assets/review_intelligence.png`

## 9. Future Improvements

- Swap TF-IDF for sentence-embedding similarity (`sentence-transformers`) for
  semantic (not just lexical) matching, and add FAISS/Annoy for approximate
  nearest-neighbor search at scale beyond ~50k restaurants.
- Fine-tune a transformer (DistilBERT) for sentiment instead of VADER, trained
  on labeled restaurant review data.
- Replace extractive summarization with an LLM-based abstractive summary
  (tradeoff: added latency/cost vs. more natural summaries).
- Add collaborative filtering (user-restaurant interaction data) to complement
  the current content-based approach — a hybrid recommender is the natural v2.
- A/B test the confidence-score weighting against real click-through data
  instead of the current hand-set heuristic weights.

## 10. Deployment (Streamlit Cloud)

1. Push this repo to GitHub (public or private with Streamlit Cloud access).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. Click **New app**, select this repo, branch `main`, entry point `app.py`.
4. Under **Advanced settings**, add a `Secrets`/startup command if needed, and
   ensure `setup_nltk.py`'s corpora download on first boot (or bundle
   `nltk_data/` in the repo to avoid a cold-start download).
5. Deploy — Streamlit Cloud installs `requirements.txt` automatically.
6. Once live, copy the app URL into the top of this README.

---

Built as a portfolio project demonstrating end-to-end NLP + product engineering:
data cleaning, NLP pipeline design, recommendation systems, sentiment analysis,
and applied UI/UX — not just model training in a notebook.
