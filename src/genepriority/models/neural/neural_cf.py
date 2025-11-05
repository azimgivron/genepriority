from typing import List

import torch
from torch import nn


class NeuralCF(nn.Module):
    """
    Neural Collaborative Filtering for gene-disease prioritization.

    This model learns embeddings for genes and diseases, projects their side-information
    through linear layers, concatenates all representations, and passes them through
    an MLP (with dropout) to predict association scores.

    Mathematical formulation:
        s_g = W_g x_g + b_g        where x_g ∈ R^{d_g}
        s_d = W_d y_d + b_d        where y_d ∈ R^{d_d}

        z_gd = [s_g; s_d] ∈ R^{2k}

        h^(1) = ReLU(W^(1) z_gd + b^(1))
        …
        h^(L) = ReLU(W^(L) h^(L-1) + b^(L))

        ŷ_gd = w^T h^(L) + b

    Args:
        embedding_dim (int): Dimensionality k of gene/disease embeddings.
        gene_feat_dim (int): Dimensionality d_g of gene side-information.
        disease_feat_dim (int): Dimensionality d_d of disease side-information.
        hidden_dims (List[int]): List of MLP hidden layer sizes [h_1, …, h_L].
        dropout (float): Dropout probability to apply after each ReLU (default: 0.0).
    """

    def __init__(
        self,
        embedding_dim: int,
        gene_feat_dim: int,
        disease_feat_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.3,
    ):
        super().__init__()
        # Side‐feature projections
        self.gene_feat_fc = nn.Linear(gene_feat_dim, embedding_dim)
        self.disease_feat_fc = nn.Linear(disease_feat_dim, embedding_dim)

        # MLP on concatenated vector [s_g; s_d] with dropout
        layers: List[nn.Module] = []
        input_dim = embedding_dim * 2
        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            input_dim = h
        self.mlp = nn.Sequential(*layers)

        # Final regression to a single score
        self.output = nn.Linear(input_dim, 1)
        self._init_weights()

    def _init_weights(self):
        """
        Initializes the weights of all layers in the model.

        Linear layers:
            - Weights are initialized using Xavier uniform initialization.
            - Biases are initialized to zero.
            - This is suitable for layers followed by ReLU activations, ensuring
            variance is preserved across layers during forward and backward passes.

        Embedding layers:
            - Weights are initialized from a normal distribution with mean 0.0 and
            standard deviation 0.01.
            - This produces small initial values, preventing large initial activations
            and helping stabilize training when embeddings are combined with other inputs.

        This method is called once during model construction to promote stable and
        efficient training.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)

    def forward(
        self,
        gene_feat: torch.FloatTensor,
        disease_feat: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Forward pass.

        Args:
            gene_feat (FloatTensor): shape (B, d_g) gene side features
            disease_feat (FloatTensor): shape (B, d_d) disease side features

        Returns:
            FloatTensor: shape (B,) predicted association scores
        """
        # Project side features
        s_g = self.gene_feat_fc(gene_feat)
        s_d = self.disease_feat_fc(disease_feat)
        # Concatenate and MLP
        x = torch.cat([s_g, s_d], dim=-1)
        h = self.mlp(x)
        return self.output(h).squeeze(-1)
