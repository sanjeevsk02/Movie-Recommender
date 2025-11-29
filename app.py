import streamlit as st
import torch
import pandas as pd
import numpy as np

from model import PMF
from data_loader import load_data


# ---------------------------------------------------
# Load everything once (simple global cache)
# ---------------------------------------------------
_loaded = False
model = None


def init():
    global _loaded, model, movies_df, movie_metadata, user_features
    global user_id_map, movie_id_map, device

    if _loaded:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    (
        train_df, test_df, num_users, num_movies,
        movie_metadata_np, user_id_map, movie_id_map, user_features_np
    ) = load_data()

    movie_metadata = torch.tensor(movie_metadata_np, dtype=torch.float32, device=device)
    user_features = torch.tensor(user_features_np, dtype=torch.float32, device=device)

    # Model setup
    m = PMF(
        num_users=num_users,
        num_movies=num_movies,
        embed_dim=128,
        use_metadata=True,
        movie_metadata_dim=movie_metadata.shape[1],
        user_metadata_dim=user_features.shape[1]
    ).to(device)

    m.load_state_dict(torch.load("pmf_model.pth", map_location=device))
    m.eval()
    model = m

    # Movie titles lookup
    movies_df = pd.read_csv("data/ml-20m/movies.csv")
    movies_df = movies_df[movies_df["movieId"].isin(movie_id_map.keys())]

    _loaded = True


# ---------------------------------------------------
# Predict rating for one pair
# ---------------------------------------------------
def predict_single(user_id, movie_id):
    uid = user_id_map[user_id]
    mid = movie_id_map[movie_id]

    u = torch.tensor([uid], device=device)
    m = torch.tensor([mid], device=device)

    with torch.no_grad():
        pred = model(u, m, movie_metadata, user_features).item()

    return float(np.clip(pred, 0.5, 5.0))


# ---------------------------------------------------
# Simple recommender (looping)
# ---------------------------------------------------
def recommend_for_user(user_id, top_n=10):
    scored = []

    for _, row in movies_df.iterrows():
        try:
            r = predict_single(user_id, row["movieId"])
            scored.append((row["title"], r))
        except:
            continue

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]


# ---------------------------------------------------
# Streamlit UI
# ---------------------------------------------------
st.set_page_config(page_title="CineMatch+", layout="wide")

st.markdown("""
# 🎬 CineMatch+
A simple movie recommendation powered by a hybrid PMF model  
""")

init()

st.markdown("---")


# ======= LAYOUT START (Two Column UI) =======
left, right = st.columns([1.1, 1])


# ---------------------------------------------------
# USER SELECT (LEFT)
# ---------------------------------------------------
with left:
    st.subheader("👤 Choose a User")
    user = st.selectbox("User ID", sorted(user_id_map.keys()))

    st.markdown("### 🔍 Search a Movie")
    search = st.text_input("Type part of the movie title...")

    if search:
        matches = movies_df[movies_df["title"].str.contains(search, case=False)]
    else:
        matches = movies_df.sample(50)  # random small list

    movie_choice = st.selectbox("Select Movie", matches["title"].tolist())

    if st.button("Predict Rating", use_container_width=True):
        movie_id = movies_df[movies_df["title"] == movie_choice]["movieId"].iloc[0]
        pred = predict_single(user, movie_id)

        st.success(f"### ⭐ Predicted rating: **{pred:.2f} / 5**")


# ---------------------------------------------------
# RECOMMENDATIONS (RIGHT)
# ---------------------------------------------------
with right:
    st.subheader("🔥 Top 10 Recommendations")
    recs = recommend_for_user(user, top_n=10)

    for title, score in recs:
        st.write(f"**{title}**  ⭐ {score:.2f}")

st.markdown("---")


