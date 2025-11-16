# model.py – FINAL SAFE VERSION
# Uses item-based CF when user_item_matrix exists,
# otherwise popularity + sentiment fallback.

import os
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict

# ---------- Load artifacts ----------
_ITEM_SIM_PATH = "item_sim_df.pkl"
_PROD_STATS_PATH = "product_stats_ext.pkl"
_USER_ITEM_PATH = "user_item_matrix.pkl"  # optional

_item_sim_df = joblib.load(_ITEM_SIM_PATH)

_product_stats = pd.read_pickle(_PROD_STATS_PATH)
if "id" in _product_stats.columns:
    _product_stats = _product_stats.set_index("id")

_user_item = None
if os.path.exists(_USER_ITEM_PATH):
    try:
        _user_item = pd.read_pickle(_USER_ITEM_PATH)
    except Exception:
        _user_item = None


# ---------- Helpers ----------

def _popularity_fallback(top_k: int = 20) -> pd.DataFrame:
    """
    Popularity + sentiment fallback used when user is unknown or
    user_item matrix is not available.
    """
    cols = ["product_name", "avg_rating", "n_ratings",
            "pct_pos", "mean_pos_prob"]
    cols = [c for c in cols if c in _product_stats.columns]

    df = _product_stats[cols].copy()

    sort_cols = [c for c in ["pct_pos", "mean_pos_prob",
                             "avg_rating", "n_ratings"]
                 if c in df.columns]

    df = df.sort_values(sort_cols, ascending=False).head(top_k)
    df = df.reset_index()
    df["recomm_score"] = 0.0
    return df


# ---------- Top-20 recommender (item-based + fallback) ----------

def get_top20_products(username: str, top_k: int = 20) -> pd.DataFrame:
    """
    Item-based recommendations if possible; otherwise popularity fallback.
    This function DOES NOT add sentiment columns – only metadata.
    """
    if _user_item is None or username not in _user_item.index:
        return _popularity_fallback(top_k=top_k)

    user_ratings = _user_item.loc[username]
    rated_items = user_ratings[user_ratings > 0].index.tolist()

    scores = {}
    for item in _item_sim_df.columns:
        if item in rated_items:
            continue
        if not rated_items:
            scores[item] = 0.0
            continue

        sims = _item_sim_df.loc[item, rated_items]
        ratings = user_ratings[rated_items]

        if sims.sum() == 0:
            scores[item] = 0.0
        else:
            scores[item] = float(np.dot(sims.values, ratings.values) /
                                 (sims.sum() + 1e-9))

    ranked = sorted(scores.items(),
                    key=lambda x: x[1],
                    reverse=True)[:top_k]

    rows = []
    for pid, score in ranked:
        meta = _product_stats.loc[pid] if pid in _product_stats.index else {}
        rows.append({
            "id": pid,
            "product_name": meta.get("product_name"),
            "avg_rating": float(meta.get("avg_rating", np.nan))
                          if not pd.isna(meta.get("avg_rating", np.nan))
                          else None,
            "n_ratings": int(meta.get("n_ratings", 0)),
            "recomm_score": float(score),
        })

    return pd.DataFrame(rows)


# ---------- Final Top-5 with sentiment ----------

def get_final_recommendations(username: str, top_k_candidates: int = 100, final_k: int = 5):
    """
    Improved final recommendations:
    - get a larger candidate pool (default 100) to increase variety
    - merge only sentiment columns with safety
    - create a blended score = w1*recomm_score + w2*pct_pos + w3*mean_pos_prob
    - sort by blended score then avg_rating
    """
    cand_df = get_top20_products(username, top_k_candidates).copy()

    # ensure id index
    if 'id' in cand_df.columns:
        cand_df = cand_df.set_index('id')

    # sentiment-only columns
    sent_cols = [c for c in ['pct_pos', 'mean_pos_prob'] if c in _product_stats.columns]
    prod_sent = _product_stats.reindex(cand_df.index)[sent_cols].copy() if sent_cols else pd.DataFrame(index=cand_df.index)

    # drop any overlapping columns from prod_sent to avoid join collisions
    overlap = [c for c in prod_sent.columns if c in cand_df.columns]
    if overlap:
        prod_sent = prod_sent.drop(columns=overlap, errors='ignore')

    merged = cand_df.join(prod_sent, how='left')

    # ensure sentiment cols exist and fillna
    for c in ['pct_pos', 'mean_pos_prob']:
        if c not in merged.columns:
            merged[c] = 0.0
        else:
            merged[c] = merged[c].fillna(0.0)

    # ensure recomm_score exists (some fallbacks may not compute it)
    if 'recomm_score' not in merged.columns:
        merged['recomm_score'] = 0.0

    # Compute blended score: tweak weights to prefer personalization or sentiment
    # Recommended initial weights:
    #   recomm_score (personalization) = 0.6
    #   pct_pos (sentiment prevalence)   = 0.3
    #   mean_pos_prob (sentiment strength)= 0.1
    merged['blend'] = 0.6 * merged['recomm_score'] + 0.3 * merged['pct_pos'] + 0.1 * merged['mean_pos_prob']

    # sort by blend then avg_rating
    sort_cols = ['blend']
    if 'avg_rating' in merged.columns:
        sort_cols.append('avg_rating')

    merged = merged.reset_index().sort_values(by=sort_cols, ascending=False)

    # optional: filter out items user already rated -- generally already excluded by recommend logic
    # but if you want to be safe:
    try:
        if getattr(_user_item, 'shape', None) and username in _user_item.index:
            already = set(_user_item.loc[username][_user_item.loc[username] > 0].index.tolist())
            merged = merged[~merged['id'].isin(already)]
    except Exception:
        pass

    # return top final_k
    out = []
    for _, r in merged.head(final_k).iterrows():
        out.append({
            'id': r['id'],
            'product_name': r.get('product_name'),
            'avg_rating': float(r['avg_rating']) if 'avg_rating' in r and not pd.isna(r.get('avg_rating')) else None,
            'n_ratings': int(r['n_ratings']) if 'n_ratings' in r and not pd.isna(r.get('n_ratings')) else 0,
            'pct_pos': float(r.get('pct_pos', 0.0)),
            'mean_pos_prob': float(r.get('mean_pos_prob', 0.0)),
            'blend': float(r.get('blend', 0.0))
        })
    return out
