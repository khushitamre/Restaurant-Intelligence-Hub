"""
recommendation.py
-------------------
Content-based restaurant recommender.

Pipeline:
  TF-IDF vectorize the combined text corpus (cuisine + dishes + reviews)
  -> cosine similarity matrix between every pair of restaurants
  -> for a query restaurant, rank all others by similarity
  -> attach a plain-English EXPLANATION (shared cuisines, similar price
     tier, comparable rating) and a CONFIDENCE SCORE, because a bare
     ranked list is not a product -- "why" is the feature that makes
     this look like something a startup shipped, not a class exercise.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RestaurantRecommender:
    def __init__(self, df: pd.DataFrame, corpus: pd.Series):
        """
        df      : cleaned dataframe, one row per restaurant (post clean_dataframe)
        corpus  : cleaned text per restaurant, aligned by position to df
        """
        self.df = df.reset_index(drop=True)
        self.corpus = corpus.reset_index(drop=True)
        self.name_to_idx = pd.Series(self.df.index, index=self.df["name"].str.lower())

        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),   # unigrams + bigrams capture phrases like "north indian"
            min_df=2,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)

        # NOTE: for large catalogs (100k+ restaurants) a full pairwise
        # cosine matrix is O(n^2) memory and won't scale -- in production
        # you'd use approximate nearest neighbors (e.g. Annoy/FAISS) instead.
        # At this dataset's scale (~3.5k rows) the dense matrix is fine
        # and keeps the code simple and inspectable, which matters more
        # for a portfolio piece an interviewer will actually read.
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)

    # ------------------------------------------------------------------
    def _get_index(self, restaurant_name: str):
        key = restaurant_name.lower()
        if key not in self.name_to_idx:
            return None
        match = self.name_to_idx[key]
        return int(match.iloc[0]) if isinstance(match, pd.Series) else int(match)

    def _explain(self, query_idx: int, cand_idx: int, sim_score: float) -> str:
        """Builds a human-readable reason for the recommendation."""
        q = self.df.iloc[query_idx]
        c = self.df.iloc[cand_idx]

        q_cuisines = set(x.strip() for x in str(q["cuisines"]).split(","))
        c_cuisines = set(x.strip() for x in str(c["cuisines"]).split(","))
        shared = q_cuisines & c_cuisines

        reasons = []
        if shared:
            reasons.append(f"shares {', '.join(list(shared)[:3])} cuisine with {q['name']}")
        cost_diff_pct = abs(c["cost_clean"] - q["cost_clean"]) / max(q["cost_clean"], 1)
        if cost_diff_pct < 0.25:
            reasons.append("similar price range")
        if abs(c["rating_clean"] - q["rating_clean"]) < 0.3:
            reasons.append("comparable rating")
        if c["rest_type"] == q["rest_type"]:
            reasons.append(f"same dining style ({c['rest_type']})")

        if not reasons:
            reasons.append("overlapping menu and review vocabulary")

        return f"Recommended because it {', '.join(reasons)}. Text similarity score: {sim_score:.2f}."

    @staticmethod
    def _confidence_score(sim_score: float, rating: float, review_count: int) -> float:
        """
        Blends three signals into one 0-100 confidence score:
          - text similarity (primary signal, 60% weight)
          - restaurant's own rating, normalized to 0-1 (25% weight)
          - review volume as a reliability proxy, log-scaled (15% weight)
        This is a deliberately simple, explainable weighting -- a real
        production system would learn these weights from click-through
        data, but a transparent heuristic is the right call for v1 and
        is easy to defend in an interview.
        """
        sim_component = sim_score
        rating_component = np.clip(rating / 5.0, 0, 1)
        volume_component = np.clip(np.log1p(review_count) / np.log1p(50), 0, 1)
        score = 0.60 * sim_component + 0.25 * rating_component + 0.15 * volume_component
        return round(float(score) * 100, 1)

    # ------------------------------------------------------------------
    def recommend(self, restaurant_name: str, top_n: int = 8) -> list:
        """Returns a list of dicts, each a recommended restaurant with
        name, location, cuisines, cost, rating, similarity, confidence,
        and a natural-language explanation. Empty list if not found."""
        idx = self._get_index(restaurant_name)
        if idx is None:
            return []

        sims = self.similarity_matrix[idx]
        ranked = np.argsort(-sims)
        ranked = [i for i in ranked if i != idx][:top_n]

        results = []
        for cand_idx in ranked:
            row = self.df.iloc[cand_idx]
            sim_score = float(sims[cand_idx])
            results.append({
                "name": row["name"],
                "location": row["location"],
                "cuisines": row["cuisines"],
                "rest_type": row["rest_type"],
                "cost_for_two": row["cost_clean"],
                "rating": row["rating_clean"],
                "similarity": round(sim_score, 3),
                "confidence": self._confidence_score(sim_score, row["rating_clean"], row["review_count"]),
                "explanation": self._explain(idx, cand_idx, sim_score),
            })
        return results

    def find_similar_by_filters(self, cuisine=None, location=None, budget_max=None, min_rating=None) -> pd.DataFrame:
        """Filter-based search (not similarity-based) -- powers the
        sidebar filter panel in the Streamlit app."""
        out = self.df.copy()
        if cuisine and cuisine != "All":
            out = out[out["cuisines"].str.contains(cuisine, case=False, na=False)]
        if location and location != "All":
            out = out[out["location"] == location]
        if budget_max:
            out = out[out["cost_clean"] <= budget_max]
        if min_rating:
            out = out[out["rating_clean"] >= min_rating]
        return out.sort_values("rating_clean", ascending=False)
