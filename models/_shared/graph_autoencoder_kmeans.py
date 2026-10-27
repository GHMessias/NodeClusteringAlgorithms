"""Shared I/O, validation, and KMeans utilities for graph autoencoders."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch_geometric.utils import coalesce, remove_self_loops, to_undirected


def torch_load(path: str | Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def get_attr(data: Any, name: str) -> Any:
    if isinstance(data, dict):
        return data.get(name)
    return getattr(data, name, None)


def is_graph_data(value: Any) -> bool:
    return (
        not isinstance(value, type)
        and get_attr(value, "x") is not None
        and get_attr(value, "edge_index") is not None
    )


def load_data(path: str | Path) -> Any:
    value = torch_load(path)
    if isinstance(value, dict) and "data" in value:
        value = value["data"]
    if isinstance(value, (tuple, list)):
        candidates = [item for item in value if is_graph_data(item)]
        if len(candidates) != 1:
            raise ValueError("data_pt must contain exactly one graph")
        value = candidates[0]
    if not is_graph_data(value):
        raise ValueError("data_pt must provide data.x and data.edge_index")
    return value


def as_features(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor) and value.is_sparse:
        value = value.to_dense()
    elif hasattr(value, "toarray"):
        value = value.toarray()
    features = torch.as_tensor(value, dtype=torch.float32)
    if features.ndim != 2:
        raise ValueError("data.x must have shape [num_nodes, num_features]")
    if features.shape[0] < 2 or features.shape[1] < 1:
        raise ValueError("data.x must contain at least two nodes and one feature")
    if not torch.isfinite(features).all():
        raise ValueError("data.x contains NaN or infinite values")
    return features.contiguous()


def as_edge_index(value: Any, num_nodes: int) -> torch.Tensor:
    edge_index = torch.as_tensor(value, dtype=torch.long)
    if edge_index.ndim != 2:
        raise ValueError("data.edge_index must be two-dimensional")
    if edge_index.shape[0] != 2:
        edge_index = edge_index.t().contiguous()
    if edge_index.shape[0] != 2:
        raise ValueError("data.edge_index must have shape [2, num_edges]")
    if edge_index.numel() == 0:
        raise ValueError("graph autoencoders require at least one edge")
    if int(edge_index.min()) < 0 or int(edge_index.max()) >= num_nodes:
        raise ValueError("data.edge_index contains a node outside [0, num_nodes)")

    edge_index, _ = remove_self_loops(edge_index)
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    edge_index = coalesce(edge_index, num_nodes=num_nodes)
    if edge_index.numel() == 0:
        raise ValueError("graph has no non-self-loop edges")
    return edge_index


def label_vector(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    labels = np.asarray(value)
    if labels.ndim > 1:
        labels = labels.argmax(axis=-1)
    return labels.reshape(-1)


def resolve_num_clusters(data: Any, configured: int | None, num_nodes: int) -> int:
    if configured is not None:
        num_clusters = int(configured)
    else:
        true_labels = get_attr(data, "y")
        if true_labels is None:
            raise ValueError("Set --n_clusters when data.y is unavailable.")
        num_clusters = int(np.unique(label_vector(true_labels)).size)
    if num_clusters < 2 or num_clusters >= num_nodes:
        raise ValueError(
            f"n_clusters must satisfy 2 <= K < num_nodes ({num_nodes}); "
            f"received {num_clusters}."
        )
    return num_clusters


def resolve_device(requested: str) -> torch.device:
    normalized = "cuda" if requested == "gpu" else requested
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but it is not available.")
    return torch.device(normalized)


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data_pt", "--data-pt", dest="data_pt", required=True)
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "gpu"), default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_clusters", "--n-clusters", dest="n_clusters", type=int)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument(
        "--hidden_channels", "--hidden-channels", dest="hidden_channels", type=int, default=128
    )
    parser.add_argument(
        "--embedding_dim", "--embedding-dim", dest="embedding_dim", type=int, default=32
    )
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--weight_decay", "--weight-decay", dest="weight_decay", type=float, default=0.0
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--kmeans_n_init", "--kmeans-n-init", dest="kmeans_n_init", type=int, default=20
    )
    parser.add_argument(
        "--kmeans_max_iter",
        "--kmeans-max-iter",
        dest="kmeans_max_iter",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--log_interval", "--log-interval", dest="log_interval", type=int, default=10
    )
    return parser


def validate_common_args(args: argparse.Namespace) -> None:
    positive = {
        "epochs": args.epochs,
        "hidden_channels": args.hidden_channels,
        "embedding_dim": args.embedding_dim,
        "kmeans_n_init": args.kmeans_n_init,
        "kmeans_max_iter": args.kmeans_max_iter,
        "log_interval": args.log_interval,
    }
    invalid = [name for name, value in positive.items() if value < 1]
    if invalid:
        raise ValueError(f"These parameters must be positive: {', '.join(invalid)}")
    if args.lr <= 0:
        raise ValueError("lr must be positive")
    if args.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")


def prepare_inputs(args: argparse.Namespace) -> tuple[Any, torch.Tensor, torch.Tensor, int]:
    data = load_data(args.data_pt)
    features = as_features(get_attr(data, "x"))
    num_nodes = int(features.shape[0])
    edge_index = as_edge_index(get_attr(data, "edge_index"), num_nodes)
    num_clusters = resolve_num_clusters(data, args.n_clusters, num_nodes)
    return data, features, edge_index, num_clusters


def save_history(path: Path, history: list[Mapping[str, float]]) -> None:
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def cluster_and_save(
    *,
    model_name: str,
    embeddings: torch.Tensor | np.ndarray,
    num_clusters: int,
    args: argparse.Namespace,
    history: list[Mapping[str, float]],
    summary: Mapping[str, Any],
) -> None:
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    embeddings_array = np.asarray(embeddings, dtype=np.float32)
    if embeddings_array.ndim != 2:
        raise ValueError("learned embeddings must be two-dimensional")
    if not np.isfinite(embeddings_array).all():
        raise RuntimeError("learned embeddings contain NaN or infinite values")

    estimator = KMeans(
        n_clusters=num_clusters,
        init="k-means++",
        n_init=args.kmeans_n_init,
        max_iter=args.kmeans_max_iter,
        random_state=args.seed,
    )
    predicted_labels = estimator.fit_predict(embeddings_array).astype(np.int64, copy=False)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    labels_path = output_dir / f"{model_name}_{timestamp}_labels.npy"
    np.save(labels_path, predicted_labels)
    np.save(output_dir / "embeddings.npy", embeddings_array)
    np.save(output_dir / "cluster_centers.npy", estimator.cluster_centers_)
    save_history(output_dir / "training_history.csv", history)

    payload = {
        "model": model_name,
        "representation": "learned_graph_autoencoder_embedding",
        "requested_clusters": int(num_clusters),
        "clusters_found": int(np.unique(predicted_labels).size),
        "kmeans_inertia": float(estimator.inertia_),
        "kmeans_iterations": int(estimator.n_iter_),
        **dict(summary),
    }
    with (output_dir / "clustering_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)

    print(f"device={resolve_device(args.device)}")
    print(f"requested_clusters={num_clusters}")
    print(f"clusters_found={np.unique(predicted_labels).size}")
    print(f"kmeans_inertia={estimator.inertia_:.6f}")
    print(f"labels={labels_path}")
