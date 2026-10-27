from __future__ import annotations

"""YAML-facing runner for the DensGNN baseline."""

from typing import Any, Callable

import numpy as np
from sklearn.metrics import normalized_mutual_info_score

from graph_benchmark.models.densgnn import DensGNNConfig, train_densgnn
from graph_benchmark.registry import ALGORITHM_REGISTRY
from graph_benchmark.utils.densgnn_loss import canonical_point_selection, canonical_support_distance


def _posthoc_evaluation_history(
    data: Any,
    evaluation_assignments: list[dict[str, Any]],
) -> list[dict[str, float | None]]:
    """Add label-based diagnostics only after training has finished."""

    raw_labels = getattr(data, "y", None)
    true_labels: np.ndarray | None = None
    if raw_labels is not None:
        if hasattr(raw_labels, "detach"):
            raw_labels = raw_labels.detach().cpu().numpy()
        true_labels = np.asarray(raw_labels)
        if true_labels.ndim == 2 and true_labels.shape[1] == 1:
            true_labels = true_labels.reshape(-1)
        elif true_labels.ndim > 1:
            true_labels = true_labels.argmax(axis=-1)
        true_labels = true_labels.reshape(-1)

    history: list[dict[str, float | None]] = []
    for snapshot in evaluation_assignments:
        predicted_labels = np.asarray(snapshot["assignments"]).reshape(-1)
        if true_labels is not None and true_labels.size != predicted_labels.size:
            raise ValueError(
                "Ground-truth and predicted label counts differ while building "
                "the DensGNN epoch NMI history."
            )
        evaluation_entry = {
            key: value
            for key, value in snapshot.items()
            if key != "assignments"
        }
        evaluation_entry["nmi"] = (
            float(normalized_mutual_info_score(true_labels, predicted_labels))
            if true_labels is not None
            else None
        )
        history.append(evaluation_entry)
    return history


@ALGORITHM_REGISTRY.register("densgnn")
def run_densgnn(
    dataset_bundle: dict[str, Any],
    algorithm_config: dict[str, Any],
    *,
    warmup_callback: Callable[[np.ndarray, int], None] | None = None,
) -> dict[str, Any]:
    dataset = dataset_bundle["dataset"]
    data = dataset_bundle["data"]
    params = algorithm_config.get("params", {})
    point_selection = canonical_point_selection(
        str(params.get("point_selection", "core"))
    )
    support_distance = canonical_support_distance(
        str(params.get("support_distance", "mrd"))
    )
    save_epoch_nmi = bool(params.get("save_epoch_nmi", False))

    output = train_densgnn(
        data=data,
        labels=None,
        config=DensGNNConfig(
            input_channels=int(dataset.num_features),
            hidden_channels=int(params.get("encoder_hidden_channels", 128)),
            embedding_channels=int(params.get("embedding_channels", 32)),
            epochs=int(params.get("epochs", 100)),
            warmup_epochs=int(params.get("warmup_epochs", 10)),
            learning_rate=float(params.get("learning_rate", 0.01)),
            weight_decay=float(params.get("weight_decay", 0.0)),
            dropout=float(params.get("dropout", 0.0)),
            clustering_loss_gamma=float(params.get("clustering_loss_gamma", 10.0)),
            update_p_interval=int(params.get("update_p_interval", 5)),
            point_selection=point_selection,
            point_probability_threshold=float(params.get("point_probability_threshold", 0.5)),
            hdbscan_min_cluster_size=int(params.get("hdbscan_min_cluster_size", 10)),
            hdbscan_min_samples=int(params.get("hdbscan_min_samples", 5)),
            hdbscan_cluster_selection_method=str(params.get("hdbscan_cluster_selection_method", "eom")),
            mrd_k=int(params.get("mrd_k", params.get("hdbscan_min_samples", 5))),
            support_distance=support_distance,
            random_state=int(params.get("random_state", 42)),
            device=str(params.get("device", "auto")),
            verbose=bool(params.get("verbose", False)),
            log_interval=int(params.get("log_interval", 1)),
            evaluation_interval=int(params.get("evaluation_interval", 5)),
            capture_evaluation_assignments=save_epoch_nmi,
            encoder_type=str(params.get("encoder_type", "gcn")),
        ),
        warmup_callback=warmup_callback,
    )
    evaluation_history = (
        _posthoc_evaluation_history(data, output["evaluation_assignments"])
        if save_epoch_nmi
        else output["evaluation_history"]
    )

    return {
        "assignments": output["assignments"],
        "features": output["embeddings"],
        "inertia": None,
        "metadata": {
            "name": "densgnn",
            "input": "graph_topology_and_node_features",
            "uses_topology": True,
            "result_selection": "last_epoch",
            "random_state": int(params.get("random_state", 42)),
            "encoder_type": str(params.get("encoder_type", "gcn")),
            "encoder_hidden_channels": int(params.get("encoder_hidden_channels", 128)),
            "embedding_channels": int(params.get("embedding_channels", 32)),
            "epochs": int(params.get("epochs", 100)),
            "warmup_epochs": int(params.get("warmup_epochs", 10)),
            "learning_rate": float(params.get("learning_rate", 0.01)),
            "weight_decay": float(params.get("weight_decay", 0.0)),
            "dropout": float(params.get("dropout", 0.0)),
            "clustering_loss_gamma": float(params.get("clustering_loss_gamma", 10.0)),
            "update_p_interval": int(params.get("update_p_interval", 5)),
            "point_selection": point_selection,
            "point_probability_threshold": float(params.get("point_probability_threshold", 0.5)),
            "hdbscan_min_cluster_size": int(params.get("hdbscan_min_cluster_size", 10)),
            "hdbscan_min_samples": int(params.get("hdbscan_min_samples", 5)),
            "hdbscan_cluster_selection_method": str(params.get("hdbscan_cluster_selection_method", "eom")),
            "mrd_k": int(params.get("mrd_k", params.get("hdbscan_min_samples", 5))),
            "support_distance": support_distance,
            "device": output["device"],
            "verbose": bool(params.get("verbose", False)),
            "log_interval": int(params.get("log_interval", 1)),
            "evaluation_interval": int(params.get("evaluation_interval", 5)),
            "save_epoch_nmi": save_epoch_nmi,
            "loss_history": output["loss_history"],
            "evaluation_history": evaluation_history,
            "selection_alphas": output["selection_alphas"],
            "relative_densities": output["relative_densities"],
            "cluster_support_stats": output["cluster_support_stats"],
        },
        "extra_metrics": output["extra_metrics"],
    }
