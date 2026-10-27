from __future__ import annotations

"""Loss utilities for DensGNN.

DensGNN replaces centroid-based cluster representations with support sets
obtained from HDBSCAN. Each cluster is represented by a subset of nodes
selected from the latent space. The clustering signal is then computed from the
average mutual reachability distance (MRD) between a node embedding and the
support nodes of every cluster.

This file keeps the math explicit because the implementation is meant to be
read, reviewed and iterated on as part of the benchmark project.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch


@dataclass(frozen=True)
class ClusterSupport:
    """Support nodes used to represent one HDBSCAN cluster inside DensGNN.

    Parameters
    ----------
    label:
        Original cluster label returned by HDBSCAN.
    node_ids:
        Node ids chosen to represent the cluster in the loss.
    node_weights:
        Optional normalized weights for the support nodes. When omitted, every
        support contributes equally, preserving the original core/border
        behavior. Adaptive selection uses these weights to combine the mean
        core and border distances without changing the MRD definition.
    """

    label: int
    node_ids: torch.Tensor
    node_weights: torch.Tensor | None = None


def canonical_point_selection(point_selection: str) -> str:
    """Normalize and validate the configured support-selection strategy."""

    normalized = point_selection.strip().lower()
    if normalized == "adaptative":
        normalized = "adaptive"
    if normalized not in {"core", "border", "adaptive"}:
        raise ValueError(
            "point_selection must be 'core', 'border', or 'adaptive' "
            "('adaptative' is accepted as an alias)."
        )
    return normalized


def canonical_support_distance(support_distance: str) -> str:
    """Validate the distance used from a node to each cluster support point."""

    normalized = support_distance.strip().lower()
    if normalized not in {"mrd", "euclidean"}:
        raise ValueError("support_distance must be 'mrd' or 'euclidean'.")
    return normalized


def embedding_core_distances(embeddings: np.ndarray, k: int) -> np.ndarray:
    """Return the distance to each embedding's k-th nearest other point."""

    num_nodes = int(embeddings.shape[0])
    if num_nodes < 2:
        return np.zeros(num_nodes, dtype=np.float64)

    effective_k = max(1, min(int(k), num_nodes - 1))
    neighbors = NearestNeighbors(
        n_neighbors=effective_k + 1,
        metric="euclidean",
    ).fit(embeddings)
    distances, _ = neighbors.kneighbors(embeddings, return_distance=True)
    return distances[:, -1].astype(np.float64, copy=False)


def robust_density_alphas(
    cluster_radii: dict[int, float],
    global_radius: float,
    *,
    eps: float = 1e-12,
) -> tuple[dict[int, float], dict[int, float]]:
    """Map relative cluster densities to alpha values in [0, 1].

    A cluster radius is the median core distance of its members. Relative
    density is ``global_radius / cluster_radius``. Its logarithm is centered
    and scaled with MAD; IQR and standard deviation are robust fallbacks for
    tied or very small collections of clusters.
    """

    if not cluster_radii:
        return {}, {}

    labels = list(cluster_radii)
    relative_values = np.asarray(
        [max(global_radius, eps) / max(cluster_radii[label], eps) for label in labels],
        dtype=np.float64,
    )
    log_densities = np.log(np.maximum(relative_values, eps))
    center = float(np.median(log_densities))
    mad = float(np.median(np.abs(log_densities - center)))
    scale = 1.4826 * mad

    if scale <= eps:
        q25, q75 = np.percentile(log_densities, [25.0, 75.0])
        scale = float((q75 - q25) / 1.349)
    if scale <= eps:
        scale = float(np.std(log_densities))

    if scale <= eps:
        alpha_values = np.full(log_densities.shape, 0.5, dtype=np.float64)
    else:
        robust_scores = np.clip((log_densities - center) / scale, -60.0, 60.0)
        alpha_values = 1.0 / (1.0 + np.exp(-robust_scores))

    relative_densities = {
        label: float(value) for label, value in zip(labels, relative_values, strict=True)
    }
    alphas = {
        label: float(value) for label, value in zip(labels, alpha_values, strict=True)
    }
    return relative_densities, alphas


def adaptive_cluster_support(
    *,
    label: int,
    core_node_ids: np.ndarray,
    border_node_ids: np.ndarray,
    alpha: float,
    embeddings: torch.Tensor,
) -> ClusterSupport:
    """Create support weights implementing the core/border distance mixture."""

    core_weight = (1.0 - alpha) / core_node_ids.size
    border_weight = alpha / border_node_ids.size
    node_ids = np.concatenate((core_node_ids, border_node_ids))
    node_weights = np.concatenate(
        (
            np.full(core_node_ids.size, core_weight, dtype=np.float64),
            np.full(border_node_ids.size, border_weight, dtype=np.float64),
        )
    )

    # Fallback selections can overlap. Merge repeated ids while retaining the
    # exact total core and border contribution from the mixture equation.
    unique_node_ids, inverse = np.unique(node_ids, return_inverse=True)
    merged_weights = np.zeros(unique_node_ids.size, dtype=np.float64)
    np.add.at(merged_weights, inverse, node_weights)

    return ClusterSupport(
        label=label,
        node_ids=torch.as_tensor(unique_node_ids, dtype=torch.long, device=embeddings.device),
        node_weights=torch.as_tensor(
            merged_weights,
            dtype=embeddings.dtype,
            device=embeddings.device,
        ),
    )


def _kth_neighbor_distances(
    pairwise_distances: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Approximate the core distance used inside mutual reachability distance."""

    num_nodes = pairwise_distances.size(0)
    if num_nodes < 2:
        return torch.zeros(num_nodes, dtype=pairwise_distances.dtype, device=pairwise_distances.device)

    effective_k = max(1, min(k, num_nodes - 1))
    masked = pairwise_distances.clone()
    masked.fill_diagonal_(float("inf"))
    knn_distances, _ = torch.topk(masked, k=effective_k, dim=1, largest=False)
    return knn_distances[:, -1]


def support_cluster_distances(
    embeddings: torch.Tensor,
    supports: list[ClusterSupport],
    mrd_k: int,
    support_distance: str = "mrd",
) -> torch.Tensor:
    """Compute the DensGNN cluster distance matrix.

    For node i and cluster j, the default DensGNN distance is:

        d_ij = (1 / |Omega_j|) * sum_{l in Omega_j} MRD(z_i, z_l)

    where Omega_j is the selected support set of cluster j. With
    support_distance="euclidean", the same support points and weights are used
    but MRD(z_i, z_l) is replaced by ||z_i - z_l||_2.
    """

    if not supports:
        raise ValueError("At least one support cluster is required.")

    distance_mode = canonical_support_distance(support_distance)
    pairwise_distances = torch.cdist(embeddings, embeddings, p=2)
    if distance_mode == "mrd":
        core_distances = _kth_neighbor_distances(pairwise_distances, k=mrd_k)
        node_core_distances = core_distances.unsqueeze(1)
    per_cluster_distances: list[torch.Tensor] = []

    for support in supports:
        point_to_support = pairwise_distances[:, support.node_ids]
        if distance_mode == "mrd":
            support_core_distances = core_distances[support.node_ids].unsqueeze(0)
            distances = torch.maximum(point_to_support, node_core_distances)
            distances = torch.maximum(distances, support_core_distances)
        else:
            distances = point_to_support
        if support.node_weights is None:
            per_cluster_distances.append(distances.mean(dim=1))
            continue

        weights = support.node_weights.to(device=distances.device, dtype=distances.dtype)
        if weights.ndim != 1 or weights.numel() != support.node_ids.numel():
            raise ValueError(
                "Cluster support weights must be one-dimensional and match node_ids."
            )
        if not torch.isfinite(weights).all() or torch.any(weights < 0):
            raise ValueError("Cluster support weights must be finite and non-negative.")
        weight_sum = weights.sum()
        if weight_sum <= 0:
            raise ValueError("Cluster support weights must have a positive sum.")
        normalized_weights = weights / weight_sum
        per_cluster_distances.append(
            (distances * normalized_weights.unsqueeze(0)).sum(dim=1)
        )

    return torch.stack(per_cluster_distances, dim=1)


def inverse_distance_assignments(
    cluster_distances: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Convert cluster distances into DensGNN soft assignments Q."""

    inverse_distances = 1.0 / cluster_distances.clamp_min(eps)
    return inverse_distances / inverse_distances.sum(dim=1, keepdim=True).clamp_min(eps)
