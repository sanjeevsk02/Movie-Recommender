import torch
import torch.nn as nn

class PMF(nn.Module):
    def __init__(self, num_users, num_movies, embed_dim=128,
                 use_metadata=True, movie_metadata_dim=None, user_metadata_dim=None):
        super().__init__()

        self.use_metadata = use_metadata

        # Base latent factor embeddings
        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.movie_emb = nn.Embedding(num_movies, embed_dim)

        # Bias terms (VERY IMPORTANT)
        self.user_bias = nn.Embedding(num_users, 1)
        self.movie_bias = nn.Embedding(num_movies, 1)

        # Initialize embeddings small
        nn.init.normal_(self.user_emb.weight, mean=0, std=0.01)
        nn.init.normal_(self.movie_emb.weight, mean=0, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.movie_bias.weight)

        if use_metadata:
            # Process movie metadata → embedding correction
            self.movie_meta_net = nn.Sequential(
                nn.Linear(movie_metadata_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim)
            )
            # Process user metadata → embedding correction
            self.user_meta_net = nn.Sequential(
                nn.Linear(user_metadata_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim)
            )

            # Xavier initialization for metadata nets
            for net in [self.movie_meta_net, self.user_meta_net]:
                for layer in net:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        nn.init.zeros_(layer.bias)

        self.global_bias = nn.Parameter(torch.zeros(1))

    def forward(self, user_ids, movie_ids, movie_metadata=None, user_metadata=None):
        u = self.user_emb(user_ids)
        m = self.movie_emb(movie_ids)

        user_b = self.user_bias(user_ids).squeeze()
        movie_b = self.movie_bias(movie_ids).squeeze()

        if self.use_metadata:
            # Normalize metadata before feeding to network
            movie_meta = movie_metadata[movie_ids]
            movie_meta = movie_meta / (movie_meta.norm(dim=1, keepdim=True) + 1e-6)
            m = m + self.movie_meta_net(movie_meta)

            user_meta = user_metadata[user_ids]
            user_meta = user_meta / (user_meta.norm(dim=1, keepdim=True) + 1e-6)
            u = u + self.user_meta_net(user_meta)

        pred = (u * m).sum(dim=1) + user_b + movie_b + self.global_bias
        return pred  # no clamp during training
