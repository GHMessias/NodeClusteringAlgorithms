"""DeepDensGNN training with multi-scale MRD or Euclidean support distance.

The original DensGNN implementation is imported, never patched. HDBSCAN's
``min_samples`` controls cluster discovery; ``candidate_k`` controls only the
learnable MRD scales. The legacy scalar ``mrd_k`` is retained solely for the
original adaptive core/border support-selection heuristic, when requested.
"""

from __future__ import annotations

# Optional benchmark result selection; standalone/default training is unchanged.
import os as _benchmark_os
if _benchmark_os.environ.get("BENCHMARK_SELECT_BEST_NMI") == "1":
    from benchmark_selection import selection as _nmi_selection
else:
    _nmi_selection = None


from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GAE

from graph_benchmark.models.densgnn import (
    DensGNNConfig,
    DensGNNClusterState,
    _build_cluster_state,
    _epoch_evaluation_entry,
    _selection_alpha_summary,
    _should_run_evaluation,
)
from graph_benchmark.models.gae import resolve_torch_device
from graph_benchmark.utils.clustering_loss import kl_clustering_loss, target_distribution
from graph_benchmark.utils.densgnn_loss import (
    canonical_point_selection,
    canonical_support_distance,
    inverse_distance_assignments,
    support_cluster_distances,
)
from graph_benchmark.utils.graph_ops import compute_modularity

from cluster_scale import ClusterScaleManager
from density import cluster_conditioned_distances, multiscale_core_distances, validate_candidate_k
from encoders import build_encoder


@dataclass(frozen=True)
class DeepDensGNNConfig(DensGNNConfig):
    candidate_k: tuple[int, ...] = (3, 5, 8, 12, 16, 24)
    density_temperature: float = 1.0
    log_density_scales: bool = False
    encoder_type: str = "gcn"


def _soft_assignments(
    embeddings: torch.Tensor,
    state: DensGNNClusterState,
    scales: ClusterScaleManager | None,
    config: DeepDensGNNConfig,
) -> torch.Tensor:
    if canonical_support_distance(config.support_distance) == "euclidean":
        distances = support_cluster_distances(
            embeddings, state.supports, mrd_k=config.mrd_k, support_distance="euclidean"
        )
    else:
        if scales is None:
            raise RuntimeError("MRD requires an initialized cluster scale manager")
        core, pairwise = multiscale_core_distances(embeddings, config.candidate_k)
        distances = cluster_conditioned_distances(pairwise, core, state.supports, scales.pi())
    return inverse_distance_assignments(distances)


def train_deep_densgnn(data: Data, config: DeepDensGNNConfig) -> dict[str, Any]:
    """Train without ground-truth labels; return final-epoch assignments."""

    if config.epochs < 1 or config.warmup_epochs < 0 or config.update_p_interval < 1:
        raise ValueError("epochs and update_p_interval must be positive; warmup_epochs nonnegative")
    if not 0 <= config.point_probability_threshold <= 1:
        raise ValueError("point_probability_threshold must be in [0, 1]")
    canonical_point_selection(config.point_selection)
    support_distance = canonical_support_distance(config.support_distance)
    candidate_k = validate_candidate_k(config.candidate_k)
    device = resolve_torch_device(config.device)
    model = GAE(build_encoder(
        config.encoder_type,
        config.input_channels,
        config.hidden_channels,
        config.embedding_channels,
        dropout=config.dropout,
    )).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scales = (
        ClusterScaleManager(candidate_k, config.density_temperature)
        if support_distance == "mrd" else None
    )
    working_data = data.clone().to(device)
    original_features = working_data.x.to_dense() if working_data.x.is_sparse else working_data.x
    original_features_np = original_features.detach().cpu().numpy()

    state: DensGNNClusterState | None = None
    target_assignments: torch.Tensor | None = None
    loss_history: list[dict[str, float]] = []
    evaluation_assignments: list[dict[str, Any]] = []
    scale_history: list[dict[str, Any]] = []

    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad()
        embeddings = model.encode(working_data.x.float(), working_data.edge_index)
        reconstruction_loss = model.recon_loss(embeddings, working_data.edge_index)
        clustering_loss = torch.zeros((), device=device, dtype=embeddings.dtype)
        soft_assignments: torch.Tensor | None = None

        if epoch >= config.warmup_epochs:
            refresh = (
                state is None
                or target_assignments is None
                or (epoch - config.warmup_epochs) % config.update_p_interval == 0
            )
            if refresh:
                # HDBSCAN detaches only for discrete cluster discovery. The
                # MRD path below keeps Q -> Pi -> logits differentiable;
                # Euclidean mode does not create or optimize scale logits.
                new_state = _build_cluster_state(embeddings, config)
                if new_state is not None:
                    matches = (
                        scales.refresh(
                            new_state.raw_labels,
                            new_state.supports,
                            optimizer,
                            device=device,
                            learning_rate=config.learning_rate,
                        )
                        if scales is not None else {}
                    )
                    state = new_state
                    target_assignments = None
                    if config.log_density_scales and scales is not None:
                        scale_history.append({
                            "epoch": epoch + 1,
                            "matched_prior_labels": matches,
                            **scales.get_density_scale_info(),
                        })
                    if config.verbose:
                        print(
                            f"[deep_densgnn] epoch={epoch + 1} "
                            f"hdbscan_clusters={len(state.supports)} matched={len(matches)}"
                        )
                elif config.verbose:
                    print(f"[deep_densgnn] epoch={epoch + 1} hdbscan_update=invalid_clustering")

            if state is not None:
                soft_assignments = _soft_assignments(embeddings, state, scales, config)
                if target_assignments is None:
                    target_assignments = target_distribution(soft_assignments).detach()
                clustering_loss = kl_clustering_loss(soft_assignments, target_assignments)

        total_loss = reconstruction_loss + config.clustering_loss_gamma * clustering_loss
        total_loss.backward()
        optimizer.step()

        history_entry = {
            "epoch": float(epoch + 1),
            "reconstruction_loss": float(reconstruction_loss.item()),
            "clustering_loss": float(clustering_loss.item()),
            "total_loss": float(total_loss.item()),
            "selected_clusters": float(len(state.supports) if state else 0),
        }
        alpha_summary = _selection_alpha_summary(state)
        if alpha_summary is not None:
            history_entry.update({
                "selection_alpha_mean": alpha_summary[0],
                "selection_alpha_min": alpha_summary[1],
                "selection_alpha_max": alpha_summary[2],
            })
        loss_history.append(history_entry)
        if _nmi_selection is not None and soft_assignments is not None:
            _nmi_selection.consider(soft_assignments.detach().argmax(1), embeddings, epoch + 1)

        if config.capture_evaluation_assignments and soft_assignments is not None and state is not None and _should_run_evaluation(epoch, config):
            assignments = soft_assignments.argmax(dim=1).detach().cpu().numpy()
            evaluation_assignments.append({
                **_epoch_evaluation_entry(
                    epoch=epoch + 1,
                    assignments=assignments,
                    embeddings=embeddings.detach().cpu().numpy(),
                    original_features=original_features_np,
                    noise_ratio=state.noise_ratio,
                ),
                "assignments": assignments.copy(),
            })

        if config.verbose and (epoch == 0 or (epoch + 1) % max(1, config.log_interval) == 0 or epoch + 1 == config.epochs):
            print(
                f"[deep_densgnn] epoch={epoch + 1}/{config.epochs} "
                f"recon_loss={history_entry['reconstruction_loss']:.6f} "
                f"cluster_loss={history_entry['clustering_loss']:.6f} "
                f"total_loss={history_entry['total_loss']:.6f}"
            )

    model.eval()
    with torch.no_grad():
        embeddings = model.encode(working_data.x.float(), working_data.edge_index)
        reconstruction_loss = model.recon_loss(embeddings, working_data.edge_index)
        final_state = _build_cluster_state(embeddings, config)
        if final_state is None:
            hard_assignments = np.zeros(data.num_nodes, dtype=np.int64)
            final_cluster_loss = 0.0
            final_noise_ratio = 1.0
            final_support_points = 0
            final_num_clusters = 1
            scale_info = {
                "active": scales is not None,
                "cluster_labels": [], "candidate_k": list(candidate_k), "pi": [], "k_eff": [],
            }
        else:
            if scales is not None:
                scales.refresh(
                    final_state.raw_labels,
                    final_state.supports,
                    optimizer,
                    device=device,
                    learning_rate=config.learning_rate,
                )
            soft_assignments = _soft_assignments(embeddings, final_state, scales, config)
            hard_assignments = soft_assignments.argmax(dim=1).cpu().numpy()
            final_cluster_loss = float(kl_clustering_loss(soft_assignments, target_distribution(soft_assignments)).item())
            final_noise_ratio = final_state.noise_ratio
            final_support_points = final_state.support_points
            final_num_clusters = len(final_state.supports)
            scale_info = (
                {"active": True, **scales.get_density_scale_info()}
                if scales is not None else {
                    "active": False,
                    "cluster_labels": [int(s.label) for s in final_state.supports],
                    "candidate_k": list(candidate_k),
                    "pi": [],
                    "k_eff": [],
                }
            )

    if config.capture_evaluation_assignments:
        final_entry = {
            **_epoch_evaluation_entry(
                epoch=config.epochs,
                assignments=hard_assignments,
                embeddings=embeddings.detach().cpu().numpy(),
                original_features=original_features_np,
                noise_ratio=final_noise_ratio,
            ),
            "assignments": hard_assignments.copy(),
        }
        if evaluation_assignments and evaluation_assignments[-1]["epoch"] == float(config.epochs):
            evaluation_assignments[-1] = final_entry
        else:
            evaluation_assignments.append(final_entry)

    if _nmi_selection is not None:
        _nmi_selection.consider(hard_assignments, embeddings, config.epochs, stage="final_inference")
    extra_metrics = {
        "reconstruction_loss": float(reconstruction_loss.item()),
        "clustering_loss": final_cluster_loss,
        "total_loss": float(reconstruction_loss.item() + config.clustering_loss_gamma * final_cluster_loss),
        "hdbscan_num_clusters": float(final_num_clusters),
        "hdbscan_noise_ratio": float(final_noise_ratio),
        "support_points": float(final_support_points),
        "modularity": compute_modularity(data, hard_assignments),
    }
    alpha_summary = _selection_alpha_summary(final_state)
    if alpha_summary is not None:
        extra_metrics.update({
            "selection_alpha_mean": alpha_summary[0],
            "selection_alpha_min": alpha_summary[1],
            "selection_alpha_max": alpha_summary[2],
        })
    return {
        "embeddings": embeddings.detach().cpu().numpy(),
        "assignments": hard_assignments,
        "loss_history": loss_history,
        "evaluation_assignments": evaluation_assignments,
        "scale_history": scale_history,
        "density_scale_info": scale_info,
        "cluster_support_stats": list(final_state.cluster_support_stats) if final_state else [],
        "selection_alphas": dict(final_state.selection_alphas) if final_state else {},
        "relative_densities": dict(final_state.relative_densities) if final_state else {},
        "extra_metrics": extra_metrics,
        "device": str(device),
    }
