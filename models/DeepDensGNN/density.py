"""Differentiable cluster-conditioned MRD for DeepDensGNN.

``hdbscan_min_samples`` belongs to the non-differentiable HDBSCAN refresh.
``candidate_k`` instead supplies the k-neighbor ranks used here. A single
top-k operation up to max(candidate_k) produces all candidate core distances.
``Pi`` learns a cluster-specific mixture of those distances; ``k_eff`` is only
a diagnostic expectation and never enters the loss directly. Consequently
MRD_j(x, y) is conditioned on cluster j, not one global graph distance.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from graph_benchmark.utils.densgnn_loss import ClusterSupport


def validate_candidate_k(candidate_k: Sequence[int]) -> tuple[int, ...]:
    if not candidate_k or any(isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in candidate_k):
        raise ValueError("candidate_k must be a nonempty sequence of positive integers")
    if len(set(candidate_k)) != len(candidate_k):
        raise ValueError("candidate_k values must be unique")
    return tuple(candidate_k)


def multiscale_core_distances(
    embeddings: torch.Tensor,
    candidate_k: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return [N, M] core distances and one reusable [N, N] distance matrix.

    As in the original single-k DensGNN, each point itself is excluded and
    ranks above N-1 are clipped to N-1. The neighbor ranking is discrete, but
    selected distances and all cluster-scale logits remain in the autograd graph.
    """

    ranks = validate_candidate_k(candidate_k)
    n = embeddings.size(0)
    if n < 2:
        raise ValueError("Multi-scale MRD requires at least two nodes")
    pairwise = torch.cdist(embeddings, embeddings, p=2)
    masked = pairwise.clone()
    masked.fill_diagonal_(float("inf"))
    distances = torch.topk(masked, k=min(max(ranks), n - 1), dim=1, largest=False).values
    indices = torch.tensor([min(k, n - 1) - 1 for k in ranks], device=embeddings.device)
    core = distances.index_select(1, indices)
    return core, pairwise


def cluster_conditioned_distances(
    pairwise: torch.Tensor,
    core: torch.Tensor,
    supports: list[ClusterSupport],
    pi: torch.Tensor,
) -> torch.Tensor:
    """Compute [N, C] support-averaged MRD_j without an [N, N, C] tensor."""

    if not supports or pi.shape != (len(supports), core.size(1)):
        raise ValueError("Pi must have one row per support cluster and one column per k")
    effective = core @ pi.T  # [N, C], differentiable with respect to Pi.
    distances = []
    for j, support in enumerate(supports):
        mrd = torch.maximum(pairwise[:, support.node_ids], effective[:, j:j + 1])
        mrd = torch.maximum(mrd, effective[support.node_ids, j].unsqueeze(0))
        if support.node_weights is None:
            distances.append(mrd.mean(dim=1))
        else:
            weights = support.node_weights.to(device=mrd.device, dtype=mrd.dtype)
            if weights.ndim != 1 or weights.numel() != support.node_ids.numel():
                raise ValueError("Support weights must match support node ids")
            if not torch.isfinite(weights).all() or torch.any(weights < 0) or weights.sum() <= 0:
                raise ValueError("Support weights must be finite, nonnegative and nonzero")
            distances.append((mrd * (weights / weights.sum()).unsqueeze(0)).sum(dim=1))
    return torch.stack(distances, dim=1)
