import streamlit as st
import torch
import pandas as pd
import numpy as np

from model import PMF
from data_loader import load_data

# ---------------------------------------------------
# CACHE EVERYTHING 
# ---------------------------------------------------
@st.cache_resource
def load_all():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    (
        train_df, _, num_users, num_movies,
        movie_metadata_np, user_id_map, movie_id_map, user_features_np
    ) = load_data()

    movie_metadata = torch.tensor(movie_metadata_np, dtype=torch.float32, device=device)
    user_features = torch.tensor(user_features_np, dtype=torch.float32, device=device)

    model = PMF(
        num_users=num_users,
        num_movies=num_movies,
        embed_dim=128,
        use_metadata=True,
        movie_metadata_dim=movie_metadata.shape[1],
        user_metadata_dim=user_features.shape[1]
    ).to(device)

    model.load_state_dict(torch.load("pmf_model.pth", map_location=device))
    model.eval()

    movies_df = pd.read_csv("data/ml-20m/movies.csv")
    movies_df = movies_df[movies_df["movieId"].isin(movie_id_map.keys())]

    return (
        model, device, train_df, movies_df, movie_metadata,
        user_features, user_id_map, movie_id_map
    )


# ---------------------------------------------------
# Load everything once
# ---------------------------------------------------
(
    model, device, train_df, movies_df, movie_metadata,
    user_features, user_id_map, movie_id_map
) = load_all()


# ---------------------------------------------------
# Predict rating for a single user–movie pair
# ---------------------------------------------------
def predict_single_vectorized(user_idx, movie_indices):
    user_tensor = torch.tensor([user_idx] * len(movie_indices), device=device)
    movie_tensor = torch.tensor(movie_indices, device=device)

    with torch.no_grad():
        preds = model(user_tensor, movie_tensor, movie_metadata, user_features)

    preds = preds.cpu().numpy()
    return np.clip(preds, 0.5, 5.0)


def predict_single(user_id, movie_id):
    uid = user_id_map[user_id]
    mid = movie_id_map[movie_id]
    return float(predict_single_vectorized(uid, [mid])[0])


# ---------------------------------------------------
# Top-K Recommendations
# ---------------------------------------------------
def recommend_for_user(user_id, top_n=10):
    uid = user_id_map[user_id]

    all_movie_ids = list(movie_id_map.keys())
    mapped_indices = [movie_id_map[m] for m in all_movie_ids]

    preds = predict_single_vectorized(uid, mapped_indices)

    df = pd.DataFrame({
        "movieId": all_movie_ids,
        "pred_rating": preds
    })

    df = df.merge(movies_df, on="movieId")
    df = df.sort_values("pred_rating", ascending=False)

    return df.head(top_n)[["title", "pred_rating"]].values.tolist()


# ---------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------
st.set_page_config(page_title="CineMatch+", layout="wide")

st.title("🎬 CineMatch+ — Hybrid Movie Recommender")
st.write("Powered by PMF + Metadata + Vectorized Ranking")


# TABS
tab1, tab2 = st.tabs([" Predict Rating", "🔥 Recommendations"])


# ---------------------------------------------------
# TAB 1 – Predict rating
# ---------------------------------------------------
with tab1:
    st.subheader("🔍 Predict a user–movie rating")

    user = st.selectbox("Select User", sorted(user_id_map.keys()))
    search = st.text_input("Search movie")

    if search:
        matches = movies_df[movies_df["title"].str.contains(search, case=False)]
    else:
        matches = movies_df.sample(50)

    movie_title = st.selectbox("Movie", matches["title"].tolist())

    if st.button("Predict Rating", key="predict_rating"):
        movie_id = movies_df[movies_df["title"] == movie_title]["movieId"].iloc[0]
        pred = predict_single(user, movie_id)

        st.success(f"⭐ **Predicted Rating:** {pred:.2f} / 5")


# ---------------------------------------------------
# TAB 2 – Recommendations
# ---------------------------------------------------
with tab2:
    st.subheader("🔥 Personalized Top 10 Recommendations")

    user2 = st.selectbox("User", sorted(user_id_map.keys()), key="rec_user")

    recs = recommend_for_user(user2, 10)

    for title, score in recs:
        st.write(f"**{title}** — ⭐ {score:.2f}")
