#!/usr/bin/env python3
"""Shared clustering evaluation for the benchmark runner and manual use."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    f1_score,
    normalized_mutual_info_score,
    pairwise_distances,
    silhouette_score,
)


SUPPORTED_METRICS = (
    "runtime_seconds",
    "acc",
    "nmi",
    "ari",
    "f1_macro",
    "f1_micro",
    "modularity",
    "n_clusters",
    "silhouette",
    "calinski_harabasz",
    "davies_bouldin",
    "dbcv",
)
SUPERVISED_METRICS = {"acc", "nmi", "ari", "f1_macro", "f1_micro"}
EMBEDDING_METRICS = {"silhouette", "calinski_harabasz", "davies_bouldin", "dbcv"}
# Kept as an alias for callers that imported the previous public constant.
ATTRIBUTE_METRICS = EMBEDDING_METRICS


def torch_load(path: str | Path) -> Any:
    """Load old and new torch serialization formats on CPU."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def unwrap_data_pt(obj: Any) -> Any:
    if isinstance(obj, (tuple, list)):
        for item in obj:
            if get_attr(item, "edge_index") is not None or get_attr(item, "x") is not None:
                return item
        return obj[0]
    if isinstance(obj, dict) and "data" in obj:
        return obj["data"]
    return obj


def load_data(path: str | Path) -> Any:
    return unwrap_data_pt(torch_load(path))


def get_attr(data: Any, name: str) -> Any:
    if isinstance(data, dict):
        return data.get(name)
    return getattr(data, name, None)


def as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        if value.is_sparse:
            value = value.to_dense()
        value = value.detach().cpu().numpy()
    elif hasattr(value, "toarray"):
        value = value.toarray()
    return np.asarray(value)


def as_label_vector(value: Any) -> np.ndarray:
    labels = as_numpy(value)
    if labels.ndim > 1:
        labels = labels.argmax(axis=-1)
    return labels.reshape(-1).astype(np.int64)


def extract_labels_from_dict(value: dict[str, Any], path: str | Path) -> Any:
    for key in ("labels", "pred", "preds", "y_pred", "clusters"):
        if key in value:
            return value[key]
    raise ValueError(f"{path} contains a dictionary without a known labels key")


def load_pred_labels(path: str | Path) -> np.ndarray:
    labels = np.load(path, allow_pickle=True)
    if isinstance(labels, np.ndarray) and labels.shape == () and isinstance(labels.item(), dict):
        labels = extract_labels_from_dict(labels.item(), path)
    return as_label_vector(labels)


def infer_num_nodes(data: Any) -> int:
    num_nodes = get_attr(data, "num_nodes")
    if num_nodes is not None:
        return int(num_nodes)

    x = get_attr(data, "x")
    if x is not None:
        return int(as_numpy(x).shape[0])

    y = get_attr(data, "y")
    if y is not None:
        return int(as_label_vector(y).shape[0])

    edge_index = get_attr(data, "edge_index")
    if edge_index is not None:
        edges = normalize_edge_index(edge_index)
        return int(edges.max()) + 1 if edges.size else 0
    raise ValueError("Cannot infer node count: data has no num_nodes, x, y, or edge_index")


def normalize_edge_index(edge_index: Any) -> np.ndarray:
    edges = as_numpy(edge_index)
    if edges.ndim != 2:
        raise ValueError("edge_index must be a two-dimensional array")
    if edges.shape[0] != 2:
        edges = edges.T
    if edges.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, edges] or [edges, 2]")
    return edges.astype(np.int64, copy=False)


def clustering_match(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, np.ndarray]:
    """Match arbitrary cluster IDs to true IDs through optimal assignment."""
    true_values = np.unique(y_true)
    pred_values = np.unique(y_pred)
    true_index = {label: index for index, label in enumerate(true_values)}
    pred_index = {label: index for index, label in enumerate(pred_values)}

    contingency = np.zeros((len(true_values), len(pred_values)), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        contingency[true_index[true_label], pred_index[pred_label]] += 1

    row_indices, column_indices = linear_sum_assignment(-contingency)
    mapping = {
        pred_values[column]: true_values[row]
        for row, column in zip(row_indices, column_indices)
    }
    fallback = true_values[0]
    matched = np.asarray([mapping.get(label, fallback) for label in y_pred], dtype=np.int64)
    return float((matched == y_true).mean()), matched


def graph_modularity(data: Any, predicted_labels: np.ndarray) -> float:
    edge_index = get_attr(data, "edge_index")
    if edge_index is None:
        return math.nan

    num_nodes = len(predicted_labels)
    edges = normalize_edge_index(edge_index)
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    for source, target in edges.T:
        source = int(source)
        target = int(target)
        if source != target and 0 <= source < num_nodes and 0 <= target < num_nodes:
            graph.add_edge(source, target)

    if graph.number_of_edges() == 0:
        return math.nan

    communities: dict[int, set[int]] = {}
    for node, cluster in enumerate(predicted_labels):
        communities.setdefault(int(cluster), set()).add(node)
    return float(nx.algorithms.community.quality.modularity(graph, communities.values()))


def _stable_core_distances(distances: np.ndarray, dimension: int) -> np.ndarray:
    """All-points core distances without inverse powers overflowing in high D.

    Factor each row's smallest positive distance out of the generalized mean.
    Every remaining powered ratio is at most one and at least one is one.
    Zero distances are omitted, following hdbscan's duplicate-point convention.
    """
    positive = distances > 0
    nearest = np.where(positive, distances, np.inf).min(axis=1)
    ratios = np.zeros_like(distances)
    np.divide(nearest[:, None], distances, out=ratios, where=positive)
    np.power(ratios, dimension, out=ratios)
    means = ratios.sum(axis=1) / (len(distances) - 1)
    core = np.zeros(len(distances), dtype=np.float64)
    valid = means > 0
    core[valid] = nearest[valid] * np.exp(-np.log(means[valid]) / dimension)
    return core


def _stable_dbcv(features: np.ndarray, labels: np.ndarray, counts: np.ndarray) -> float:
    """Reference DBCV MST/separation logic with stable all-points core distances."""
    try:
        from hdbscan.validity import density_separation, internal_minimum_spanning_tree
    except ImportError as exc:
        raise ImportError(
            "DBCV requires hdbscan; install it with pip install -r requirements.txt"
        ) from exc

    cores = []
    internal_nodes = []
    sparseness = np.empty(len(counts), dtype=np.float64)
    for cluster in range(len(counts)):
        distances = pairwise_distances(features[labels == cluster], metric="euclidean")
        core = _stable_core_distances(distances, features.shape[1])
        # Reuse the distance matrix for mutual reachability to limit peak memory.
        np.maximum(distances, core[:, None], out=distances)
        np.maximum(distances, core[None, :], out=distances)
        nodes, edges = internal_minimum_spanning_tree(distances)
        cores.append(core)
        internal_nodes.append(nodes)
        sparseness[cluster] = edges[:, 2].max()
        del distances

    separation = np.full(len(counts), np.inf)
    for first in range(len(counts)):
        for second in range(first + 1, len(counts)):
            distance = density_separation(
                features, labels, first, second,
                internal_nodes[first], internal_nodes[second],
                cores[first], cores[second], metric="euclidean",
            )
            separation[first] = min(separation[first], distance)
            separation[second] = min(separation[second], distance)
    scores = (separation - sparseness) / np.maximum(separation, sparseness)
    return float(np.dot(counts / len(labels), scores))


def density_based_validity(features: np.ndarray, labels: np.ndarray) -> float:
    """Exact Euclidean DBCV on embeddings; -1 denotes noise, not a cluster.

    Noise stays in the total sample count used to weight cluster scores.
    Partitions with fewer than two non-noise clusters or singleton clusters
    have no usable density separation/MST and are reported as NaN.
    """
    clustered = labels != -1
    _, inverse, counts = np.unique(
        labels[clustered], return_inverse=True, return_counts=True
    )
    if len(counts) < 2 or np.any(counts < 2):
        return math.nan
    if features.shape[1] == 0 or not np.isfinite(features).all():
        return math.nan

    # Compact IDs avoid allocations proportional to the largest cluster label.
    normalized = np.full(len(labels), -1, dtype=np.int64)
    normalized[clustered] = inverse
    # The compiled MST implementation requires double precision.
    features = np.ascontiguousarray(features, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        score = _stable_dbcv(features, normalized, counts)
    return score if math.isfinite(score) else math.nan


def embedding_clustering_metrics(
    embeddings: Any,
    predicted_labels: np.ndarray,
    requested_metrics: Iterable[str],
) -> dict[str, float]:
    """Evaluate a partition in the model's learned embedding space."""
    requested = EMBEDDING_METRICS.intersection(requested_metrics)
    if not requested:
        return {}

    results = {metric: math.nan for metric in requested}
    if embeddings is None:
        raise ValueError(
            "Model embeddings are required for: " + ", ".join(sorted(requested))
        )

    features = as_numpy(embeddings)
    if features.ndim != 2:
        raise ValueError("Model embeddings must be a two-dimensional matrix")
    if features.shape[0] != len(predicted_labels):
        raise ValueError(
            f"Embedding row count ({features.shape[0]}) does not match graph node count "
            f"({len(predicted_labels)})"
        )

    num_clusters = int(np.unique(predicted_labels).size)
    if num_clusters < 2 or num_clusters >= len(predicted_labels):
        return results

    scorers = {
        "silhouette": silhouette_score,
        "calinski_harabasz": calinski_harabasz_score,
        "davies_bouldin": davies_bouldin_score,
        "dbcv": density_based_validity,
    }
    for metric in requested:
        try:
            results[metric] = float(scorers[metric](features, predicted_labels))
        except ValueError:
            results[metric] = math.nan
    return results


def validate_requested_metrics(requested_metrics: Iterable[str]) -> list[str]:
    metrics = list(dict.fromkeys(requested_metrics))
    unknown = sorted(set(metrics) - set(SUPPORTED_METRICS))
    if unknown:
        raise ValueError(f"Unsupported metrics: {', '.join(unknown)}")
    return metrics


def evaluate_clustering(
    data: Any,
    predicted_labels: Any,
    requested_metrics: Iterable[str],
    embeddings: Any = None,
) -> dict[str, float | int]:
    """Compute only requested metrics; supervised metrics become NaN without data.y."""
    requested = validate_requested_metrics(requested_metrics)
    labels = as_label_vector(predicted_labels)
    num_nodes = infer_num_nodes(data)
    if len(labels) != num_nodes:
        raise ValueError(
            f"Predicted label count ({len(labels)}) does not match graph node count ({num_nodes})"
        )

    results: dict[str, float | int] = {}
    if "n_clusters" in requested:
        results["n_clusters"] = int(np.unique(labels).size)
    if "modularity" in requested:
        results["modularity"] = graph_modularity(data, labels)
    results.update(embedding_clustering_metrics(embeddings, labels, requested))

    requested_supervised = SUPERVISED_METRICS.intersection(requested)
    if not requested_supervised:
        return results

    true_value = get_attr(data, "y")
    if true_value is None:
        for metric in requested_supervised:
            results[metric] = math.nan
        return results

    true_labels = as_label_vector(true_value)
    if len(true_labels) != num_nodes:
        raise ValueError(
            f"Ground-truth label count ({len(true_labels)}) does not match graph node count ({num_nodes})"
        )

    acc, matched = clustering_match(true_labels, labels)
    if "acc" in requested:
        results["acc"] = acc
    if "nmi" in requested:
        results["nmi"] = float(normalized_mutual_info_score(true_labels, labels))
    if "ari" in requested:
        results["ari"] = float(adjusted_rand_score(true_labels, labels))
    if "f1_macro" in requested:
        results["f1_macro"] = float(
            f1_score(true_labels, matched, average="macro", zero_division=0)
        )
    if "f1_micro" in requested:
        results["f1_micro"] = float(
            f1_score(true_labels, matched, average="micro", zero_division=0)
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one clustering labels file")
    parser.add_argument("--data-pt", "--data_pt", dest="data_pt", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output-json", default=None, help="Also save metrics as JSON")
    parser.add_argument(
        "--embeddings",
        default=None,
        help="Learned model embeddings (.npy), required by embedding-space internal metrics",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=[metric for metric in SUPPORTED_METRICS if metric != "runtime_seconds"],
        choices=SUPPORTED_METRICS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested = [metric for metric in args.metrics if metric != "runtime_seconds"]
    embeddings = np.load(args.embeddings, allow_pickle=False) if args.embeddings else None
    results = evaluate_clustering(
        load_data(args.data_pt),
        load_pred_labels(args.labels),
        requested,
        embeddings=embeddings,
    )
    serialized = json.dumps(results, indent=2, sort_keys=True, allow_nan=True)
    if args.output_json:
        Path(args.output_json).write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
