#!/usr/bin/env python3
"""Cluster nodes with KMeans using only their input attributes."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.cluster import KMeans


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
        raise ValueError("data_pt must provide x and edge_index")
    return value


def as_numpy_features(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        if value.is_sparse:
            value = value.to_dense()
        value = value.detach().cpu().numpy()
    elif hasattr(value, "toarray"):
        value = value.toarray()
    features = np.asarray(value, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError("data.x must have shape [num_nodes, num_features]")
    if features.shape[0] < 2 or features.shape[1] < 1:
        raise ValueError("data.x must contain at least two nodes and one feature")
    if not np.isfinite(features).all():
        raise ValueError("data.x contains NaN or infinite values")
    return features


def label_vector(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    labels = np.asarray(value)
    if labels.ndim > 1:
        labels = labels.argmax(axis=-1)
    return labels.reshape(-1)


def resolve_num_clusters(data: Any, configured: int | None, num_nodes: int) -> int:
    if configured is not None:
        num_clusters = configured
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_pt", "--data-pt", dest="data_pt", required=True)
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_clusters", "--n-clusters", dest="n_clusters", type=int)
    parser.add_argument("--n_init", "--n-init", dest="n_init", type=int, default=20)
    parser.add_argument("--max_iter", "--max-iter", dest="max_iter", type=int, default=300)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--algorithm", choices=("lloyd", "elkan"), default="lloyd")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.n_init < 1 or args.max_iter < 1:
        raise ValueError("n_init and max_iter must be positive")
    if args.tol <= 0:
        raise ValueError("tol must be positive")

    data = load_data(args.data_pt)
    features = as_numpy_features(get_attr(data, "x"))
    num_clusters = resolve_num_clusters(data, args.n_clusters, features.shape[0])

    estimator = KMeans(
        n_clusters=num_clusters,
        init="k-means++",
        n_init=args.n_init,
        max_iter=args.max_iter,
        tol=args.tol,
        random_state=args.seed,
        algorithm=args.algorithm,
    )
    predicted_labels = estimator.fit_predict(features).astype(np.int64, copy=False)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    labels_path = output_dir / f"KMeansAttributes_{timestamp}_labels.npy"
    np.save(labels_path, predicted_labels)
    # The representation evaluated by this baseline is exactly data.x.
    np.save(output_dir / "embeddings.npy", features)
    np.save(output_dir / "cluster_centers.npy", estimator.cluster_centers_)
    with (output_dir / "clustering_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "inertia": float(estimator.inertia_),
                "iterations": int(estimator.n_iter_),
                "requested_clusters": num_clusters,
                "clusters_found": int(np.unique(predicted_labels).size),
                "representation": "original_node_attributes",
            },
            stream,
            indent=2,
        )

    print("representation=original_node_attributes")
    print(f"requested_clusters={num_clusters}")
    print(f"clusters_found={np.unique(predicted_labels).size}")
    print(f"inertia={estimator.inertia_:.6f}")
    print(f"iterations={estimator.n_iter_}")
    print(f"labels={labels_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
