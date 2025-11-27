import torch
import pandas as pd
from model import PMF
from data_loader import load_data


def predict_rating(user_id, movie_title, model_path="pmf_model.pth"):
    (
        _, _, num_users, num_movies,
        movie_metadata, user_id_map, movie_id_map, user_features
    ) = load_data()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    movie_metadata = torch.tensor(movie_metadata, dtype=torch.float32, device=device)
    user_features = torch.tensor(user_features, dtype=torch.float32, device=device)

    model = PMF(
        num_users=num_users,
        num_movies=num_movies,
        embed_dim=128,
        use_metadata=True,
        movie_metadata_dim=movie_metadata.shape[1],
        user_metadata_dim=user_features.shape[1],
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    movies_df = pd.read_csv("data/ml-20m/movies.csv")

    # Find movie by title
    movie_row = movies_df[movies_df["title"].str.contains(movie_title, case=False)]
    if movie_row.empty:
        raise ValueError("Movie not found.")
    orig_movie_id = movie_row["movieId"].values[0]

    if orig_movie_id not in movie_id_map:
        raise ValueError("Movie not in training data.")

    user_idx = torch.tensor([user_id_map[user_id]], device=device)
    movie_idx = torch.tensor([movie_id_map[orig_movie_id]], device=device)

    with torch.no_grad():
        pred = model(user_idx, movie_idx, movie_metadata, user_features).item()

    return max(0.5, min(5.0, pred))


if __name__ == "__main__":
    print(predict_rating(1, "Toy Story"))
