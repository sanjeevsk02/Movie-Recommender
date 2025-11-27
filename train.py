import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os

from data_loader import load_data, MovieDataset
from model import PMF


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    (
        train_df, test_df, num_users, num_movies,
        movie_metadata, user_id_map, movie_id_map, user_features
    ) = load_data()

    train_loader = DataLoader(MovieDataset(train_df), batch_size=4096, shuffle=True)
    test_loader = DataLoader(MovieDataset(test_df), batch_size=4096)

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

    # Stable training hyperparams
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    criterion = nn.MSELoss()

    best_rmse = float("inf")

    for epoch in range(1, 16):
        model.train()
        running_loss = 0

        for users, movies, ratings in tqdm(train_loader, desc=f"Epoch {epoch:02d}"):
            users = users.to(device)
            movies = movies.to(device)
            ratings = ratings.to(device)

            preds = model(users, movies, movie_metadata, user_features)
            loss = criterion(preds, ratings)

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

        # Evaluate
        model.eval()
        preds_list, actual_list = [], []

        with torch.no_grad():
            for users, movies, ratings in test_loader:
                users = users.to(device)
                movies = movies.to(device)
                ratings = ratings.to(device)

                p = model(users, movies, movie_metadata, user_features)
                preds_list.extend(p.cpu().numpy())
                actual_list.extend(ratings.cpu().numpy())

        rmse = np.sqrt(np.mean((np.array(preds_list) - actual_list) ** 2))
        print(f"Epoch {epoch:02d} | Loss={running_loss/len(train_loader):.4f} | RMSE={rmse:.4f}")

        if rmse < best_rmse:
            best_rmse = rmse
            torch.save(model.state_dict(), "pmf_model.pth")
            print(" → Saved new best model.")

    print("Training complete. Best RMSE:", best_rmse)


if __name__ == "__main__":
    main()
