"""
Hierarchical Vocabulary Embedding

Instead of a flat embedding table (50K independent vectors), token embeddings
are computed by traversing a semantic hierarchy:

    Level 0 (root): general linguistic features shared by all tokens
    Level 1 (group): semantic cluster centroid (~128 groups via k-means)
    Level 2 (token): per-token residual (what distinguishes this token from its group)

    Token embedding = root + group_centroid[group_id] + residual[token_id]

Storage: 1*d + 128*d + 50257*d_residual (where d_residual can be < d for compression).
For the PoC, d_residual = d (no compression yet), but the structure is in place.

The key property: tokens in the same semantic group share their level-1 embedding.
New tokens can be placed in the hierarchy by group assignment and immediately
get a meaningful embedding without retraining.
"""

import torch
import torch.nn as nn


class HierarchicalEmbedding(nn.Module):
    """
    2-level hierarchical embedding: group centroids + per-token residuals.

    Build the hierarchy from an existing embedding table via k-means clustering.
    """

    def __init__(self, vocab_size: int, d_model: int, n_groups: int = 128):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_groups = n_groups

        # Level 0: root embedding (shared by all tokens)
        self.root = nn.Parameter(torch.zeros(d_model))

        # Level 1: group centroids
        self.group_centroids = nn.Parameter(torch.randn(n_groups, d_model) * 0.02)

        # Level 2: per-token residuals
        self.token_residuals = nn.Parameter(torch.randn(vocab_size, d_model) * 0.02)

        # Group assignments (not a parameter — fixed after hierarchy construction)
        self.register_buffer("group_ids", torch.zeros(vocab_size, dtype=torch.long))

        self._hierarchy_built = False

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Token embedding = root + group_centroid + token_residual."""
        group_ids = self.group_ids[token_ids]          # (..., )
        group_emb = self.group_centroids[group_ids]    # (..., d_model)
        token_emb = self.token_residuals[token_ids]    # (..., d_model)
        return self.root + group_emb + token_emb

    def build_hierarchy(self, base_embeddings: nn.Embedding, n_iters: int = 20):
        """
        Build the hierarchy from an existing flat embedding table via k-means.

        Args:
            base_embeddings: the model's current nn.Embedding to decompose
            n_iters: k-means iterations
        """
        with torch.no_grad():
            weights = base_embeddings.weight.data  # (vocab_size, d_model)

            # Root = global mean
            root = weights.mean(dim=0)
            self.root.data.copy_(root)

            # K-means clustering
            centered = weights - root
            centroids, assignments = self._kmeans(centered, self.n_groups, n_iters)

            self.group_centroids.data.copy_(centroids)
            self.group_ids.copy_(assignments)

            # Residuals = token embedding - root - group centroid
            group_emb = centroids[assignments]
            residuals = centered - group_emb
            self.token_residuals.data.copy_(residuals)

        self._hierarchy_built = True

    def _kmeans(
        self, data: torch.Tensor, k: int, n_iters: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Simple k-means clustering."""
        n = data.shape[0]

        # Initialize centroids with k-means++ style: random selection
        indices = torch.randperm(n)[:k]
        centroids = data[indices].clone()

        for _ in range(n_iters):
            # Assign each point to nearest centroid
            dists = torch.cdist(data, centroids)  # (n, k)
            assignments = dists.argmin(dim=1)      # (n,)

            # Update centroids
            for j in range(k):
                mask = assignments == j
                if mask.any():
                    centroids[j] = data[mask].mean(dim=0)

        return centroids, assignments

    def get_stats(self) -> dict:
        """Stats for demo output."""
        group_sizes = torch.bincount(self.group_ids, minlength=self.n_groups)
        return {
            "n_groups": self.n_groups,
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "hierarchy_built": self._hierarchy_built,
            "avg_group_size": group_sizes.float().mean().item(),
            "min_group_size": group_sizes.min().item(),
            "max_group_size": group_sizes.max().item(),
            "total_parameters": self.root.numel() + self.group_centroids.numel() + self.token_residuals.numel(),
            "flat_equivalent": self.vocab_size * self.d_model,
        }
