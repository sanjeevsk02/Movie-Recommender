"""
Correctly merges TMDB 5000 datasets using:
    movies.id == credits.movie_id

Extracts:
  - genres
  - keywords (top 500)
  - cast (top 10 actors)
  - director (top 200)
  - numeric features
  - overview TF-IDF (500)

Outputs:
  data/tmdb/tmdb_features.npy
  data/tmdb/tmdb_index_map.json
  data/tmdb/tmdb_feature_columns.json
"""

import json
import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from ast import literal_eval


# ----------------------------------------------------------
# Safe JSON decode
# ----------------------------------------------------------

def safe_json(x):
    if pd.isna(x):
        return []
    try:
        return literal_eval(x)
    except Exception:
        return []


# ----------------------------------------------------------
# Load TMDB 5000 movies + credits
# ----------------------------------------------------------

def load_tmdb(tmdb_folder="data/tmdb"):
    movies_path = os.path.join(tmdb_folder, "tmdb_5000_movies.csv")
    credits_path = os.path.join(tmdb_folder, "tmdb_5000_credits.csv")

    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)

    # FIX: use correct join key
    credits = credits.rename(columns={"movie_id": "id"})

    # Merge on TMDB movie ID (id)
    df = movies.merge(credits, on="id", how="left")

    print("Loaded TMDB rows:", len(df))
    return df


# ----------------------------------------------------------
# Extract Genres
# ----------------------------------------------------------

def extract_genres(df):
    df["genres"] = df["genres"].apply(safe_json)
    all_genres = sorted({g["name"] for lst in df["genres"] for g in lst})

    idx_map = {g: i for i, g in enumerate(all_genres)}
    mat = np.zeros((len(df), len(all_genres)), dtype=np.float32)

    for i, lst in enumerate(df["genres"]):
        for g in lst:
            mat[i, idx_map[g["name"]]] = 1.0

    return mat, all_genres


# ----------------------------------------------------------
# Extract Keywords (Top 500)
# ----------------------------------------------------------

def extract_keywords(df, top_k=500):
    df["keywords"] = df["keywords"].apply(safe_json)

    freq = {}
    for lst in df["keywords"]:
        for kw in lst:
            name = kw["name"]
            freq[name] = freq.get(name, 0) + 1

    top_keywords = sorted(freq, key=freq.get, reverse=True)[:top_k]
    idx_map = {kw: i for i, kw in enumerate(top_keywords)}

    mat = np.zeros((len(df), top_k), dtype=np.float32)
    for i, lst in enumerate(df["keywords"]):
        for kw in lst:
            name = kw["name"]
            if name in idx_map:
                mat[i, idx_map[name]] = 1.0

    return mat, top_keywords


# ----------------------------------------------------------
# Extract Cast (Top 10 actors)
# ----------------------------------------------------------

def extract_cast(df, top_k=10):
    df["cast"] = df["cast"].apply(safe_json)

    freq = {}
    for lst in df["cast"]:
        for entry in lst[:10]:  # top-billed actors
            name = entry["name"]
            freq[name] = freq.get(name, 0) + 1

    top_actors = sorted(freq, key=freq.get, reverse=True)[:top_k]
    idx_map = {a: i for i, a in enumerate(top_actors)}

    mat = np.zeros((len(df), top_k), dtype=np.float32)
    for i, lst in enumerate(df["cast"]):
        for entry in lst[:10]:
            name = entry["name"]
            if name in idx_map:
                mat[i, idx_map[name]] = 1.0

    return mat, top_actors


# ----------------------------------------------------------
# Extract Director (Top 200)
# ----------------------------------------------------------

def extract_director(df, top_k=200):
    df["crew"] = df["crew"].apply(safe_json)

    freq = {}
    for lst in df["crew"]:
        for entry in lst:
            if entry.get("job") == "Director":
                name = entry["name"]
                freq[name] = freq.get(name, 0) + 1

    top_directors = sorted(freq, key=freq.get, reverse=True)[:top_k]
    idx_map = {d: i for i, d in enumerate(top_directors)}

    mat = np.zeros((len(df), top_k), dtype=np.float32)
    for i, lst in enumerate(df["crew"]):
        for entry in lst:
            if entry.get("job") == "Director":
                name = entry["name"]
                if name in idx_map:
                    mat[i, idx_map[name]] = 1.0

    return mat, top_directors


# ----------------------------------------------------------
# Numeric fields
# ----------------------------------------------------------

NUMERIC_FIELDS = ["budget", "revenue", "runtime",
                  "popularity", "vote_average", "vote_count"]

def extract_numeric(df):
    return df[NUMERIC_FIELDS].fillna(0.0).astype(np.float32).to_numpy(), NUMERIC_FIELDS


# ----------------------------------------------------------
# Overview TF-IDF (500)
# ----------------------------------------------------------

def extract_overview(df, dim=500):
    df["overview"] = df["overview"].fillna("")
    tfidf = TfidfVectorizer(max_features=dim, stop_words="english")
    mat = tfidf.fit_transform(df["overview"]).toarray().astype(np.float32)
    return mat, tfidf.get_feature_names_out().tolist()


# ----------------------------------------------------------
# Main function
# ----------------------------------------------------------

def main():
    df = load_tmdb()

    genres, genre_list = extract_genres(df)
    keywords, keyword_list = extract_keywords(df)
    cast, actors_list = extract_cast(df)
    directors, director_list = extract_director(df)
    numeric, numeric_fields = extract_numeric(df)
    overview, tfidf_words = extract_overview(df)

    # Combine features
    features = np.concatenate(
        [genres, keywords, cast, directors, numeric, overview],
        axis=1
    )

    print("Final metadata shape =", features.shape)

    # TMDB ID → row index
    tmdb_ids = df["id"].astype(int).tolist()
    index_map = {tmdb_id: i for i, tmdb_id in enumerate(tmdb_ids)}

    os.makedirs("data/tmdb", exist_ok=True)
    np.save("data/tmdb/tmdb_features.npy", features)

    with open("data/tmdb/tmdb_index_map.json", "w") as f:
        json.dump(index_map, f)

    # Save columns for reproducibility
    with open("data/tmdb/tmdb_feature_columns.json", "w") as f:
        json.dump({
            "genres": genre_list,
            "keywords": keyword_list,
            "cast": actors_list,
            "director": director_list,
            "numeric_fields": numeric_fields,
            "tfidf_words": tfidf_words
        }, f)

    print("TMDB preprocessing complete.")


if __name__ == "__main__":
    main()
