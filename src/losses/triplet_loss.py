"""Batch-Hard Triplet Loss for Metric Learning in Person Re-ID.

Reference:
Hermans et al., 'In Defense of the Triplet Loss for Person Re-Identification', 2017.

For each anchor sample in a mini-batch:
- Select the hardest positive: max distance between samples of the same person.
- Select the hardest negative: min distance between samples of different persons.
- Loss = max(0, d(anchor, pos) - d(anchor, neg) + margin)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def euclidean_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute pairwise Euclidean distance between two matrices of vectors.

    Args:
        x: (m, d) tensor
        y: (n, d) tensor

    Returns:
        dist: (m, n) tensor of Euclidean distances
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy - 2 * torch.matmul(x, y.t())
    # Numerical stability clamp: avoid sqrt of negative / near-zero numbers
    dist = dist.clamp(min=1e-12).sqrt()
    return dist


class TripletLoss(nn.Module):
    """Batch-Hard Triplet Loss with Euclidean Distance."""

    def __init__(self, margin: float = 0.3, normalize_feature: bool = False):
        """Initialize the loss module.

        Args:
            margin: Triplet margin distance threshold (default: 0.3).
            normalize_feature: Whether to L2-normalize features before distance computation.
        """
        super().__init__()
        self.margin = margin
        self.normalize_feature = normalize_feature
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate batch-hard triplet loss.

        Args:
            features: (N, D) batch embeddings.
            targets: (N,) ground-truth identity labels.

        Returns:
            Scalar loss tensor.
        """
        if self.normalize_feature:
            features = F.normalize(features, p=2, dim=-1)

        # Pairwise distance matrix of shape (N, N)
        dist_mat = euclidean_dist(features, features)

        N = dist_mat.size(0)
        # Create masks for positive and negative pairs
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t())
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t())

        # Extract hardest positive (max distance among identical identities)
        dist_ap = []
        for i in range(N):
            pos_dists = dist_mat[i][is_pos[i]]
            dist_ap.append(pos_dists.max().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)

        # Extract hardest negative (min distance among different identities)
        dist_an = []
        for i in range(N):
            neg_dists = dist_mat[i][is_neg[i]]
            dist_an.append(neg_dists.min().unsqueeze(0))
        dist_an = torch.cat(dist_an)

        # Target label 1 indicates dist_an should be greater than dist_ap by at least margin
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss

