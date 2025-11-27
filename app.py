import streamlit as st
import torch
import pandas as pd
from model import PMF
from data_loader import load_data

# === LOAD DATA & MODEL ONCE ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

(
    train_df, test_df, num_users, num_movies,
    movie_metadata, user_id_map, movie_id_map, user_features
) = load_data()

movie_metadata = torch.tensor(movie_metadata, dtype=torch.float32, device=device)
user_features = torch.tensor(user_features, dtype=torch.float32, device=device)

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
movies_df = movies_df[movies_df["movieId"].isin(movie_id_map.keys())].reset_index(drop=True)


# === UI ===
st.title("CineMatch — Movie Rating Predictor")
st.write("Using ML-20M + TMDB Metadata + PMF Hybrid Model")

user = st.selectbox("Select User", sorted(user_id_map.keys()))
movie = st.selectbox("Select Movie", movies_df["title"].tolist())

if st.button("Predict Rating"):
    u_idx = torch.tensor([user_id_map[user]], device=device)
    m_id = movies_df[movies_df["title"] == movie]["movieId"].iloc[0]
    m_idx = torch.tensor([movie_id_map[m_id]], device=device)

    with torch.no_grad():
        pred = model(u_idx, m_idx, movie_metadata, user_features).item()

    pred = max(0.5, min(5.0, pred))
    st.success(f"Predicted Rating: **{pred:.2f} / 5.0** ⭐")
