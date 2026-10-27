"""YAML-facing adapter for DeepDensGNN in the existing graph benchmark."""

from __future__ import annotations

from typing import Any

from graph_benchmark.clustering.densgnn import _posthoc_evaluation_history
from graph_benchmark.registry import ALGORITHM_REGISTRY
from graph_benchmark.utils.densgnn_loss import canonical_point_selection, canonical_support_distance

from model import DeepDensGNNConfig, train_deep_densgnn


@ALGORITHM_REGISTRY.register("deep_densgnn")
def run_deep_densgnn(
    dataset_bundle: dict[str, Any],
    algorithm_config: dict[str, Any],
) -> dict[str, Any]:
    dataset = dataset_bundle["dataset"]
    data = dataset_bundle["data"]
    params = algorithm_config.get("params", {})
    save_epoch_nmi = bool(params.get("save_epoch_nmi", False))
    candidate_k = params.get("candidate_k", [3, 5, 8, 12, 16, 24])
    if not isinstance(candidate_k, (list, tuple)):
        raise ValueError("candidate_k must be a list of positive integers")
    config = DeepDensGNNConfig(
        encoder_type=params.get("encoder_type", "gcn"),
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
        point_selection=canonical_point_selection(str(params.get("point_selection", "core"))),
        point_probability_threshold=float(params.get("point_probability_threshold", 0.5)),
        hdbscan_min_cluster_size=int(params.get("hdbscan_min_cluster_size", 10)),
        hdbscan_min_samples=int(params.get("hdbscan_min_samples", 5)),
        hdbscan_cluster_selection_method=str(params.get("hdbscan_cluster_selection_method", "eom")),
        mrd_k=int(params.get("mrd_k", params.get("hdbscan_min_samples", 5))),
        support_distance=canonical_support_distance(str(params.get("support_distance", "mrd"))),
        random_state=int(params.get("random_state", 42)),
        device=str(params.get("device", "auto")),
        verbose=bool(params.get("verbose", False)),
        log_interval=int(params.get("log_interval", 1)),
        evaluation_interval=int(params.get("evaluation_interval", 5)),
        capture_evaluation_assignments=save_epoch_nmi,
        candidate_k=tuple(candidate_k),
        density_temperature=float(params.get("density_temperature", 1.0)),
        log_density_scales=bool(params.get("log_density_scales", False)),
    )
    output = train_deep_densgnn(data, config)
    evaluation_history = (
        _posthoc_evaluation_history(data, output["evaluation_assignments"])
        if save_epoch_nmi else []
    )
    return {
        "assignments": output["assignments"],
        "features": output["embeddings"],
        "inertia": None,
        "metadata": {
            "name": "deep_densgnn",
            "input": "graph_topology_and_node_features",
            "uses_topology": True,
            "result_selection": "last_epoch",
            "random_state": config.random_state,
            "encoder_type": config.encoder_type,
            "encoder_hidden_channels": config.hidden_channels,
            "embedding_channels": config.embedding_channels,
            "epochs": config.epochs,
            "warmup_epochs": config.warmup_epochs,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "dropout": config.dropout,
            "clustering_loss_gamma": config.clustering_loss_gamma,
            "update_p_interval": config.update_p_interval,
            "point_selection": config.point_selection,
            "point_probability_threshold": config.point_probability_threshold,
            "hdbscan_min_cluster_size": config.hdbscan_min_cluster_size,
            "hdbscan_min_samples": config.hdbscan_min_samples,
            "hdbscan_cluster_selection_method": config.hdbscan_cluster_selection_method,
            "mrd_k": config.mrd_k,
            "support_distance": config.support_distance,
            "candidate_k": list(config.candidate_k),
            "density_temperature": config.density_temperature,
            "log_density_scales": config.log_density_scales,
            "density_scale_info": output["density_scale_info"],
            "scale_history": output["scale_history"],
            "loss_history": output["loss_history"],
            "evaluation_history": evaluation_history,
            "selection_alphas": output["selection_alphas"],
            "relative_densities": output["relative_densities"],
            "cluster_support_stats": output["cluster_support_stats"],
            "device": output["device"],
        },
        "extra_metrics": output["extra_metrics"],
    }
