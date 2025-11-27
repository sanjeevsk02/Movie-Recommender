import os
import json
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class MovieDataset(Dataset):
    def __init__(self, ratings_df):
        self.users = ratings_df["userId"].values.astype(np.int64)
        self.movies = ratings_df["movieId"].values.astype(np.int64)
        self.ratings = ratings_df["rating"].values.astype(np.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]


def build_user_features_cpu(ratings_df, movie_metadata, cache_path="data/user_features.npy"):
    if os.path.exists(cache_path):
        print("Loading user metadata from cache...")
        return np.load(cache_path)

    print("Building user metadata on CPU safely...")

    users = ratings_df["userId"].values
    movies = ratings_df["movieId"].values
    ratings = ratings_df["rating"].values

    num_users = int(users.max()) + 1
    feat_dim = movie_metadata.shape[1]

    user_pref_sum = np.zeros((num_users, feat_dim), dtype=np.float32)
    user_pref_count = np.zeros((num_users, feat_dim), dtype=np.float32)
    rating_sum = np.zeros(num_users, dtype=np.float32)
    rating_count = np.zeros(num_users, dtype=np.float32)

    total = len(users)
    chunk_size = 250000

    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        u = users[start:end]
        m = movies[start:end]
        r = ratings[start:end]

        md = movie_metadata[m]

        user_pref_sum[u] += md * r[:, None]
        user_pref_count[u] += (md > 0).astype(np.float32)
        rating_sum[u] += r
        rating_count[u] += 1

        if start % 1000000 == 0:
            print(f"Processed {start:,} / {total:,}")

    user_pref = user_pref_sum / (user_pref_count + 1e-6)
    mean_rating = (rating_sum / (rating_count + 1e-6)).reshape(-1, 1)
    activity = np.log1p(rating_count).reshape(-1, 1)

    user_features = np.concatenate([user_pref, mean_rating, activity], axis=1)
    os.makedirs("data", exist_ok=True)
    np.save(cache_path, user_features)

    print(f"User metadata cached: {cache_path}")
    return user_features


def load_data(
    data_path="data/ml-20m/",
    tmdb_path="data/tmdb/",
    test_size=0.2,
    min_user_ratings=20,
    min_movie_ratings=10,
    use_tmdb=True,
):

    # Load and filter ratings
    ratings = pd.read_csv(os.path.join(data_path, "ratings.csv"))
    user_counts = ratings["userId"].value_counts()
    movie_counts = ratings["movieId"].value_counts()

    ratings = ratings[
        ratings["userId"].isin(user_counts[user_counts >= min_user_ratings].index)
    ]
    ratings = ratings[
        ratings["movieId"].isin(movie_counts[movie_counts >= min_movie_ratings].index)
    ]

    # Remap IDs
    unique_users = sorted(ratings["userId"].unique())
    unique_movies = sorted(ratings["movieId"].unique())

    user_id_map = {uid: i for i, uid in enumerate(unique_users)}
    movie_id_map = {mid: i for i, mid in enumerate(unique_movies)}

    ratings["userId"] = ratings["userId"].map(user_id_map)
    ratings["movieId"] = ratings["movieId"].map(movie_id_map)

    num_users = len(unique_users)
    num_movies = len(unique_movies)

    print(f"Filtered users: {num_users:,}, movies: {num_movies:,}, ratings: {len(ratings):,}")

    # Load TMDB metadata
    tmdb_feats = np.load(os.path.join(tmdb_path, "tmdb_features.npy"))
    with open(os.path.join(tmdb_path, "tmdb_index_map.json")) as f:
        tmdb_map = json.load(f)

    links = pd.read_csv(os.path.join(data_path, "links.csv"))

    movie_metadata = np.zeros((num_movies, tmdb_feats.shape[1]), dtype=np.float32)
    for old_id in unique_movies:
        new_idx = movie_id_map[old_id]
        tmdb_id = links.loc[links["movieId"] == old_id, "tmdbId"].fillna(0).values[0]
        tmdb_id = int(tmdb_id)

        if str(tmdb_id) in tmdb_map:
            movie_metadata[new_idx] = tmdb_feats[tmdb_map[str(tmdb_id)]]

    # Split
    train_df, test_df = train_test_split(ratings, test_size=test_size, random_state=42)

    # Build or load cached user metadata
    user_features = build_user_features_cpu(train_df, movie_metadata)

    return (
        train_df,
        test_df,
        num_users,
        num_movies,
        movie_metadata,
        user_id_map,
        movie_id_map,
        user_features,
    )
