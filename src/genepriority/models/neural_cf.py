import torch
import torch.nn as nn
from typing import List

class NeuralCF(nn.Module):
    """
    Neural Collaborative Filtering model with user and item side information.

    This model learns embeddings for users and items, projects side-information
    through linear layers, and combines all representations via a multi-layer
    perceptron (MLP) to predict ratings.

    Mathematical formulation:
        p_u = Embedding(user_index) ∈ R^k
        q_i = Embedding(item_index) ∈ R^k
        s_u = W_u x_u + b_u           where x_u ∈ R^{d_u}
        t_i = W_i y_i + b_i           where y_i ∈ R^{d_i}

        z_ui = [p_u; q_i; s_u; t_i]    ∈ R^{4k}

        h^(1) = ReLU(W^(1) z_ui + b^(1))
        h^(2) = ReLU(W^(2) h^(1) + b^(2))
        ...
        h^(L) = ReLU(W^(L) h^(L-1) + b^(L))

        ŕ_ui = w^T h^(L) + b

    Loss:
        L = (1/|D|) Σ_{(u,i)∈D} (r_ui - ŕ_ui)^2 + λ ||Θ||^2

    Args:
        num_users (int): Number of unique users.
        num_items (int): Number of unique items.
        embedding_dim (int): Dimensionality k of user/item embeddings.
        user_feat_dim (int): Dimensionality d_u of user side-information.
        item_feat_dim (int): Dimensionality d_i of item side-information.
        hidden_dims (List[int]): List of sizes for MLP hidden layers [h_1, ..., h_L].
    """
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        user_feat_dim: int,
        item_feat_dim: int,
        hidden_dims: List[int]
    ):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        # Linear projection for side information
        self.user_feat_fc = nn.Linear(user_feat_dim, embedding_dim)
        self.item_feat_fc = nn.Linear(item_feat_dim, embedding_dim)
        # MLP
        layers = []
        input_dim = embedding_dim * 4
        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        self.mlp = nn.Sequential(*layers)
        self.output = nn.Linear(input_dim, 1)

    def forward(
        self,
        user: torch.LongTensor,
        item: torch.LongTensor,
        user_feat: torch.FloatTensor,
        item_feat: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Forward pass for rating prediction.

        Args:
            user (LongTensor): Tensor of user indices, shape (batch_size,).
            item (LongTensor): Tensor of item indices, shape (batch_size,).
            user_feat (FloatTensor): User side-information, shape (batch_size, d_u).
            item_feat (FloatTensor): Item side-information, shape (batch_size, d_i).

        Returns:
            FloatTensor: Predicted ratings, shape (batch_size,).
        """
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        user_side = self.user_feat_fc(user_feat)
        item_side = self.item_feat_fc(item_feat)
        x = torch.cat([user_emb, item_emb, user_side, item_side], dim=-1)
        x = self.mlp(x)
        out = self.output(x)
        return out.squeeze()