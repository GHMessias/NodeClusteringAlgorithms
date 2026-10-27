from __future__ import annotations

# Optional benchmark result selection; standalone/default training is unchanged.
import os as _benchmark_os
if _benchmark_os.environ.get("BENCHMARK_SELECT_BEST_NMI") == "1":
    from benchmark_selection import selection as _nmi_selection
else:
    _nmi_selection = None


"""Implementation of DensGNN.

DensGNN is a density-guided deep graph clustering baseline built on top of a
Graph Autoencoder (GAE) whose encoder defaults to the original two-layer GCN
(with optional GAT or GraphSAGE) and whose decoder is the standard inner-product
decoder from PyG's ``GAE`` wrapper.

Training follows an alternating scheme:

1. Warm-up the autoencoder using only graph reconstruction.
2. Run HDBSCAN on the current latent embeddings.
3. Select core, border, or density-adaptive support nodes from each cluster.
4. Compute DensGNN soft assignments Q from average MRD to those supports.
5. Refresh the target distribution P every ``update_p_interval`` epochs and
   keep P fixed between refreshes, while Q is recomputed every epoch.

This keeps the clustering signal tied to dense support regions instead of
centroids.
"""

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GAE

from graph_benchmark.models.encoders import build_densgnn_encoder
from graph_benchmark.models.gae import resolve_torch_device
from graph_benchmark.utils.clustering_loss import kl_clustering_loss, target_distribution
from graph_benchmark.utils.densgnn_loss import (
    ClusterSupport,
    adaptive_cluster_support,
    canonical_point_selection,
    canonical_support_distance,
    embedding_core_distances,
    inverse_distance_assignments,
    robust_density_alphas,
    support_cluster_distances,
)
from graph_benchmark.utils.graph_ops import compute_clustering_accuracy, compute_modularity


@dataclass(frozen=True)
class DensGNNConfig:
    input_channels: int
    hidden_channels: int
    embedding_channels: int
    epochs: int = 100
    warmup_epochs: int = 10
    learning_rate: float = 0.01
    weight_decay: float = 0.0
    dropout: float = 0.0
    clustering_loss_gamma: float = 10.0
    update_p_interval: int = 5
    point_selection: str = "core"
    point_probability_threshold: float = 0.5
    hdbscan_min_cluster_size: int = 10
    hdbscan_min_samples: int = 5
    hdbscan_cluster_selection_method: str = "eom"
    mrd_k: int = 5
    support_distance: str = "mrd"
    random_state: int = 42
    device: str = "auto"
    verbose: bool = False
    log_interval: int = 1
    evaluation_interval: int = 5
    capture_evaluation_assignments: bool = False
    encoder_type: str = "gcn"


@dataclass(frozen=True)
class DensGNNClusterState:
    """Frozen HDBSCAN output reused across several training epochs."""

    supports: list[ClusterSupport]
    raw_labels: np.ndarray
    raw_probabilities: np.ndarray
    noise_ratio: float
    support_points: int
    selection_alphas: dict[int, float]
    relative_densities: dict[int, float]
    cluster_support_stats: list[dict[str, int | float | None]]


def _select_support_node_ids(
    cluster_node_ids: np.ndarray,
    cluster_probabilities: np.ndarray,
    point_selection: str,
    probability_threshold: float,
) -> np.ndarray:
    """Choose support points from one HDBSCAN cluster.

    ``sklearn.cluster.HDBSCAN`` does not expose explicit core/border tags. To
    keep the implementation compatible with the requested API, we approximate:

    - ``core``: members whose HDBSCAN membership strength is at least the
      configured threshold;
    - ``border``: non-noise members below that threshold.

    When the requested subset is empty, we fall back to a small non-empty set so
    the loss remains computable.
    """

    if point_selection not in {"core", "border"}:
        raise ValueError("point_selection must be either 'core' or 'border'.")

    if point_selection == "core":
        mask = cluster_probabilities >= probability_threshold
        if np.any(mask):
            return cluster_node_ids[mask]

        fallback_count = min(max(1, cluster_node_ids.size // 4), cluster_node_ids.size)
        top_indices = np.argsort(cluster_probabilities)[-fallback_count:]
        return cluster_node_ids[top_indices]

    mask = cluster_probabilities < probability_threshold
    if np.any(mask):
        return cluster_node_ids[mask]

    fallback_count = min(max(1, cluster_node_ids.size // 4), cluster_node_ids.size)
    bottom_indices = np.argsort(cluster_probabilities)[:fallback_count]
    return cluster_node_ids[bottom_indices]


def _build_cluster_state(
    embeddings: torch.Tensor,
    config: DensGNNConfig,
) -> DensGNNClusterState | None:
    """Run HDBSCAN on detached embeddings and prepare DensGNN support sets."""

    embeddings_np = embeddings.detach().cpu().numpy()
    clusterer = HDBSCAN(
        min_cluster_size=config.hdbscan_min_cluster_size,
        min_samples=config.hdbscan_min_samples,
        cluster_selection_method=config.hdbscan_cluster_selection_method,
        copy=True,
    )
    clusterer.fit(embeddings_np)

    raw_labels = clusterer.labels_.astype(np.int64)
    raw_probabilities = clusterer.probabilities_.astype(np.float32)
    cluster_labels = sorted(int(value) for value in np.unique(raw_labels) if value >= 0)
    point_selection = canonical_point_selection(config.point_selection)
    supports: list[ClusterSupport] = []
    selection_alphas: dict[int, float] = {}
    relative_densities: dict[int, float] = {}
    cluster_support_stats: list[dict[str, int | float | None]] = []

    if point_selection == "adaptive" and cluster_labels:
        core_distances = embedding_core_distances(embeddings_np, config.mrd_k)
        non_noise_mask = raw_labels >= 0
        global_radius = float(np.median(core_distances[non_noise_mask]))
        cluster_radii = {
            label: float(np.median(core_distances[raw_labels == label]))
            for label in cluster_labels
        }
        relative_densities, selection_alphas = robust_density_alphas(
            cluster_radii,
            global_radius,
        )

    for label in cluster_labels:
        cluster_node_ids = np.flatnonzero(raw_labels == label)
        cluster_probabilities = raw_probabilities[cluster_node_ids]
        support_stats: dict[str, int | float | None] = {
            "cluster_id": label,
            "cluster_size": int(cluster_node_ids.size),
            "core_candidates": int(
                np.count_nonzero(
                    cluster_probabilities >= config.point_probability_threshold
                )
            ),
            "border_candidates": int(
                np.count_nonzero(
                    cluster_probabilities < config.point_probability_threshold
                )
            ),
            "core_support_points": 0,
            "border_support_points": 0,
            "unique_support_points": 0,
            "alpha_j": None,
            "relative_density": None,
        }

        if point_selection == "adaptive":
            core_node_ids = _select_support_node_ids(
                cluster_node_ids=cluster_node_ids,
                cluster_probabilities=cluster_probabilities,
                point_selection="core",
                probability_threshold=config.point_probability_threshold,
            )
            border_node_ids = _select_support_node_ids(
                cluster_node_ids=cluster_node_ids,
                cluster_probabilities=cluster_probabilities,
                point_selection="border",
                probability_threshold=config.point_probability_threshold,
            )
            support = adaptive_cluster_support(
                label=label,
                core_node_ids=core_node_ids,
                border_node_ids=border_node_ids,
                alpha=selection_alphas[label],
                embeddings=embeddings,
            )
            supports.append(support)
            support_stats.update(
                {
                    "core_support_points": int(core_node_ids.size),
                    "border_support_points": int(border_node_ids.size),
                    "unique_support_points": int(support.node_ids.numel()),
                    "alpha_j": float(selection_alphas[label]),
                    "relative_density": float(relative_densities[label]),
                }
            )
            cluster_support_stats.append(support_stats)
            continue

        support_node_ids = _select_support_node_ids(
            cluster_node_ids=cluster_node_ids,
            cluster_probabilities=cluster_probabilities,
            point_selection=point_selection,
            probability_threshold=config.point_probability_threshold,
        )
        if support_node_ids.size == 0:
            continue

        selection_alphas[label] = 0.0 if point_selection == "core" else 1.0

        support = ClusterSupport(
            label=label,
            node_ids=torch.as_tensor(
                support_node_ids,
                dtype=torch.long,
                device=embeddings.device,
            ),
        )
        supports.append(support)
        support_stats[f"{point_selection}_support_points"] = int(
            support_node_ids.size
        )
        support_stats["unique_support_points"] = int(support.node_ids.numel())
        cluster_support_stats.append(support_stats)

    if len(supports) < 2:
        return None

    return DensGNNClusterState(
        supports=supports,
        raw_labels=raw_labels,
        raw_probabilities=raw_probabilities,
        noise_ratio=float(np.mean(raw_labels == -1)),
        support_points=int(sum(int(support.node_ids.numel()) for support in supports)),
        selection_alphas=selection_alphas,
        relative_densities=relative_densities,
        cluster_support_stats=cluster_support_stats,
    )


def _compute_densgnn_assignments(
    embeddings: torch.Tensor,
    cluster_state: DensGNNClusterState,
    config: DensGNNConfig,
) -> torch.Tensor:
    """Compute DensGNN soft assignments Q for the current embeddings."""

    cluster_distances = support_cluster_distances(
        embeddings=embeddings,
        supports=cluster_state.supports,
        mrd_k=config.mrd_k,
        support_distance=config.support_distance,
    )
    return inverse_distance_assignments(cluster_distances)


def _log_hdbscan_refresh(epoch: int, cluster_state: DensGNNClusterState | None) -> None:
    if cluster_state is None:
        print(f"[densgnn] epoch={epoch:03d} hdbscan_update=invalid_clustering")
        return

    alpha_values = list(cluster_state.selection_alphas.values())
    alpha_summary = ""
    if alpha_values:
        alpha_summary = (
            f" alpha_mean={np.mean(alpha_values):.4f}"
            f" alpha_min={np.min(alpha_values):.4f}"
            f" alpha_max={np.max(alpha_values):.4f}"
        )

    print(
        "[densgnn] "
        f"epoch={epoch:03d} "
        f"hdbscan_clusters={len(cluster_state.supports)} "
        f"support_points={cluster_state.support_points} "
        f"noise_ratio={cluster_state.noise_ratio:.4f}"
        f"{alpha_summary}"
    )


def _should_run_evaluation(epoch: int, config: DensGNNConfig) -> bool:
    """Return whether this epoch should trigger a diagnostic evaluation."""

    interval = max(1, config.evaluation_interval)
    return (epoch + 1) % interval == 0 or (epoch + 1) == config.epochs


def _format_cluster_distribution(assignments: np.ndarray) -> str:
    unique_labels, counts = np.unique(assignments, return_counts=True)
    parts = [f"{int(label)}:{int(count)}" for label, count in zip(unique_labels, counts, strict=True)]
    return "{" + ", ".join(parts) + "}"


def _selection_alpha_summary(
    cluster_state: DensGNNClusterState | None,
) -> tuple[float, float, float] | None:
    if cluster_state is None or not cluster_state.selection_alphas:
        return None
    values = np.asarray(list(cluster_state.selection_alphas.values()), dtype=np.float64)
    return float(values.mean()), float(values.min()), float(values.max())


def _internal_clustering_metrics(
    features: np.ndarray,
    assignments: np.ndarray,
    feature_space: str,
) -> dict[str, float]:
    """Compute internal diagnostics in one explicitly named feature space."""

    scorers = {
        "silhouette": silhouette_score,
        "calinski_harabasz": calinski_harabasz_score,
        "davies_bouldin": davies_bouldin_score,
    }
    results = {
        f"{metric}_{feature_space}": math.nan
        for metric in scorers
    }
    num_clusters = int(np.unique(assignments).size)
    if num_clusters < 2 or num_clusters >= len(assignments):
        return results

    for metric, scorer in scorers.items():
        try:
            results[f"{metric}_{feature_space}"] = float(
                scorer(features, assignments)
            )
        except ValueError:
            pass
    return results


def _epoch_evaluation_entry(
    *,
    epoch: int,
    assignments: np.ndarray,
    embeddings: np.ndarray,
    original_features: np.ndarray,
    noise_ratio: float,
) -> dict[str, float]:
    entry = {
        "epoch": float(epoch),
        "num_clusters": float(len(np.unique(assignments))),
        "noise_ratio": float(noise_ratio),
    }
    entry.update(
        _internal_clustering_metrics(embeddings, assignments, "embedding")
    )
    entry.update(
        _internal_clustering_metrics(original_features, assignments, "features")
    )
    return entry


def train_densgnn(
    data: Data,
    labels: np.ndarray | None,
    config: DensGNNConfig,
    *,
    warmup_callback: Callable[[np.ndarray, int], None] | None = None,
) -> dict[str, Any]:
    """Train DensGNN, optionally exporting embeddings at the warm-up boundary.

    The callback receives an independent CPU array in evaluation mode after
    exactly ``warmup_epochs`` optimizer steps, before any clustering step.
    With zero warm-up epochs it receives the untrained encoder's embeddings.
    """

    if config.epochs < 1:
        raise ValueError("DensGNN requires 'epochs' to be at least 1.")
    if warmup_callback is not None and not 0 <= config.warmup_epochs <= config.epochs:
        raise ValueError("Snapshot requires 0 <= warmup_epochs <= epochs.")
    canonical_point_selection(config.point_selection)
    canonical_support_distance(config.support_distance)
    if not 0.0 <= config.point_probability_threshold <= 1.0:
        raise ValueError("point_probability_threshold must be between 0 and 1.")

    device = resolve_torch_device(config.device)
    model = GAE(
        encoder=build_densgnn_encoder(
            encoder_type=config.encoder_type,
            input_channels=config.input_channels,
            hidden_channels=config.hidden_channels,
            embedding_channels=config.embedding_channels,
            dropout=config.dropout,
        )
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    working_data = data.clone().to(device)

    cluster_state: DensGNNClusterState | None = None
    target_assignments: torch.Tensor | None = None
    loss_history: list[dict[str, float]] = []
    evaluation_history: list[dict[str, float]] = []
    evaluation_assignments: list[dict[str, Any]] = []
    original_features = working_data.x
    if original_features.is_sparse:
        original_features = original_features.to_dense()
    original_features_np = original_features.detach().cpu().numpy()

    def capture_warmup(completed_epochs: int) -> None:
        if warmup_callback is None:
            return
        was_training = model.training
        try:
            model.eval()
            with torch.no_grad():
                snapshot = model.encode(working_data.x.float(), working_data.edge_index)
                snapshot_np = snapshot.detach().cpu().numpy().copy()
        finally:
            model.train(was_training)
        warmup_callback(snapshot_np, completed_epochs)

    if config.warmup_epochs == 0:
        capture_warmup(0)

    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad()
        embeddings = model.encode(working_data.x.float(), working_data.edge_index)
        reconstruction_loss = model.recon_loss(embeddings, working_data.edge_index)
        clustering_loss = torch.zeros((), device=device, dtype=embeddings.dtype)
        soft_assignments: torch.Tensor | None = None

        if epoch >= config.warmup_epochs:
            should_refresh_targets = (
                cluster_state is None
                or target_assignments is None
                or (epoch - config.warmup_epochs) % config.update_p_interval == 0
            )
            if should_refresh_targets:
                refreshed_state = _build_cluster_state(embeddings, config)
                if refreshed_state is not None:
                    cluster_state = refreshed_state
                    target_assignments = None
                if config.verbose:
                    _log_hdbscan_refresh(epoch, cluster_state)

            if cluster_state is not None:
                soft_assignments = _compute_densgnn_assignments(embeddings, cluster_state, config)
                if target_assignments is None:
                    target_assignments = target_distribution(soft_assignments).detach()
                clustering_loss = kl_clustering_loss(soft_assignments, target_assignments)

        total_loss = reconstruction_loss + config.clustering_loss_gamma * clustering_loss
        total_loss.backward()
        optimizer.step()

        if epoch + 1 == config.warmup_epochs:
            capture_warmup(epoch + 1)

        reconstruction_loss_value = float(reconstruction_loss.item())
        clustering_loss_value = float(clustering_loss.item())
        total_loss_value = float(total_loss.item())
        selected_clusters = len(cluster_state.supports) if cluster_state is not None else 0
        alpha_summary = _selection_alpha_summary(cluster_state)

        history_entry = {
            "epoch": float(epoch + 1),
            "reconstruction_loss": reconstruction_loss_value,
            "clustering_loss": clustering_loss_value,
            "total_loss": total_loss_value,
            "selected_clusters": float(selected_clusters),
        }
        if alpha_summary is not None:
            history_entry["selection_alpha_mean"] = alpha_summary[0]
            history_entry["selection_alpha_min"] = alpha_summary[1]
            history_entry["selection_alpha_max"] = alpha_summary[2]
        loss_history.append(history_entry)
        if _nmi_selection is not None and soft_assignments is not None:
            _nmi_selection.consider(soft_assignments.detach().argmax(1), embeddings, epoch + 1)

        if (
            (labels is not None or config.capture_evaluation_assignments)
            and soft_assignments is not None
            and cluster_state is not None
            and _should_run_evaluation(epoch, config)
        ):
            eval_assignments = soft_assignments.argmax(dim=1).detach().cpu().numpy()
            evaluation_entry = _epoch_evaluation_entry(
                epoch=epoch + 1,
                assignments=eval_assignments,
                embeddings=embeddings.detach().cpu().numpy(),
                original_features=original_features_np,
                noise_ratio=cluster_state.noise_ratio,
            )
            if labels is not None:
                evaluation_entry["nmi"] = float(
                    normalized_mutual_info_score(labels, eval_assignments)
                )
            evaluation_history.append(evaluation_entry)

            if config.capture_evaluation_assignments:
                evaluation_assignments.append(
                    {
                        **evaluation_entry,
                        "assignments": eval_assignments.copy(),
                    }
                )

            if config.verbose and labels is not None:
                print(
                    "[densgnn] "
                    f"epoch={epoch + 1:03d}/{config.epochs:03d} "
                    f"eval_nmi={evaluation_entry['nmi']:.6f} "
                    f"eval_clusters={int(evaluation_entry['num_clusters'])} "
                    f"cluster_sizes={_format_cluster_distribution(eval_assignments)} "
                    f"hdbscan_outliers={int(np.sum(cluster_state.raw_labels == -1))}"
                )

        if config.verbose and (
            epoch == 0
            or (epoch + 1) % max(1, config.log_interval) == 0
            or epoch + 1 == config.epochs
        ):
            print(
                "[densgnn] "
                f"epoch={epoch + 1:03d}/{config.epochs:03d} "
                f"recon_loss={reconstruction_loss_value:.6f} "
                f"cluster_loss={clustering_loss_value:.6f} "
                f"total_loss={total_loss_value:.6f} "
                f"selected_clusters={selected_clusters}"
            )

    model.eval()

    with torch.no_grad():
        embeddings = model.encode(working_data.x.float(), working_data.edge_index)
        reconstruction_loss = model.recon_loss(embeddings, working_data.edge_index)

    final_cluster_state = _build_cluster_state(embeddings, config)
    if final_cluster_state is None:
        hard_assignments = np.zeros(data.num_nodes, dtype=np.int64)
        final_clustering_loss_value = 0.0
        final_num_clusters = 1
        final_noise_ratio = 1.0
        final_support_points = 0
    else:
        final_soft_assignments = _compute_densgnn_assignments(embeddings, final_cluster_state, config)
        final_targets = target_distribution(final_soft_assignments)
        final_clustering_loss_value = float(
            kl_clustering_loss(final_soft_assignments, final_targets).item()
        )
        hard_assignments = final_soft_assignments.argmax(dim=1).detach().cpu().numpy()
        final_num_clusters = len(final_cluster_state.supports)
        final_noise_ratio = final_cluster_state.noise_ratio
        final_support_points = final_cluster_state.support_points

    if config.capture_evaluation_assignments:
        final_evaluation_entry: dict[str, Any] = _epoch_evaluation_entry(
            epoch=config.epochs,
            assignments=hard_assignments,
            embeddings=embeddings.detach().cpu().numpy(),
            original_features=original_features_np,
            noise_ratio=final_noise_ratio,
        )
        final_evaluation_entry["assignments"] = hard_assignments.copy()
        if evaluation_assignments and evaluation_assignments[-1]["epoch"] == float(config.epochs):
            evaluation_assignments[-1] = final_evaluation_entry
        else:
            evaluation_assignments.append(final_evaluation_entry)

    extra_metrics: dict[str, float] = {
        "reconstruction_loss": float(reconstruction_loss.item()),
        "clustering_loss": final_clustering_loss_value,
        "total_loss": float(reconstruction_loss.item() + config.clustering_loss_gamma * final_clustering_loss_value),
        "hdbscan_num_clusters": float(final_num_clusters),
        "hdbscan_noise_ratio": float(final_noise_ratio),
        "support_points": float(final_support_points),
    }

    if _nmi_selection is not None:
        _nmi_selection.consider(hard_assignments, embeddings, config.epochs, stage="final_inference")
    final_alpha_summary = _selection_alpha_summary(final_cluster_state)
    if final_alpha_summary is not None:
        extra_metrics["selection_alpha_mean"] = final_alpha_summary[0]
        extra_metrics["selection_alpha_min"] = final_alpha_summary[1]
        extra_metrics["selection_alpha_max"] = final_alpha_summary[2]

    if labels is not None:
        extra_metrics["clustering_accuracy"] = compute_clustering_accuracy(labels, hard_assignments)

    extra_metrics["modularity"] = compute_modularity(data, hard_assignments)

    return {
        "embeddings": embeddings.detach().cpu().numpy(),
        "assignments": hard_assignments,
        "loss_history": loss_history,
        "evaluation_history": evaluation_history,
        "evaluation_assignments": evaluation_assignments,
        "extra_metrics": extra_metrics,
        "selection_alphas": (
            dict(final_cluster_state.selection_alphas)
            if final_cluster_state is not None
            else {}
        ),
        "relative_densities": (
            dict(final_cluster_state.relative_densities)
            if final_cluster_state is not None
            else {}
        ),
        "cluster_support_stats": (
            list(final_cluster_state.cluster_support_stats)
            if final_cluster_state is not None
            else []
        ),
        "device": str(device),
    }
