#!/usr/bin/env python3
"""Learn PyG Node2Vec embeddings and cluster them with KMeans."""

from __future__ import annotations

# Optional benchmark result selection; standalone/default training is unchanged.
import os as _benchmark_os
if _benchmark_os.environ.get("BENCHMARK_SELECT_BEST_NMI") == "1":
    from benchmark_selection import selection as _nmi_selection
else:
    _nmi_selection = None


import argparse
import csv
from datetime import datetime
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch_geometric.nn import Node2Vec
from torch_geometric.utils import to_undirected


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
    return not isinstance(value, type) and get_attr(value, "edge_index") is not None


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
        raise ValueError("data_pt must provide edge_index")
    return value


def as_edge_index(value: Any) -> torch.Tensor:
    edges = torch.as_tensor(value, dtype=torch.long)
    if edges.ndim != 2:
        raise ValueError("data.edge_index must be two-dimensional")
    if edges.shape[0] != 2:
        edges = edges.t().contiguous()
    if edges.shape[0] != 2:
        raise ValueError("data.edge_index must have shape [2, num_edges]")
    if edges.numel() == 0:
        raise ValueError("Node2Vec requires at least one edge")
    return edges


def label_vector(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    labels = np.asarray(value)
    if labels.ndim > 1:
        labels = labels.argmax(axis=-1)
    return labels.reshape(-1)


def infer_num_nodes(data: Any, edges: torch.Tensor) -> int:
    configured = get_attr(data, "num_nodes")
    if configured is not None:
        return int(configured)
    features = get_attr(data, "x")
    if features is not None:
        return int(features.shape[0])
    labels = get_attr(data, "y")
    if labels is not None:
        return int(label_vector(labels).size)
    return int(edges.max()) + 1


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


def boolean(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


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


def train_node2vec(
    model: Node2Vec,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    log_interval: int,
    device: torch.device,
    selection_callback=None,
) -> list[dict[str, float]]:
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for positive_walks, negative_walks in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(
                positive_walks.to(device),
                negative_walks.to(device),
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Node2Vec produced a non-finite loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1

        if batches == 0:
            raise RuntimeError("Node2Vec did not produce any training batches")
        if selection_callback is not None:
            selection_callback(epoch)
        average_loss = total_loss / batches
        history.append({"epoch": float(epoch), "loss": average_loss})
        if epoch == 1 or epoch % log_interval == 0 or epoch == epochs:
            print(
                f"epoch={epoch:04d}/{epochs:04d} loss={average_loss:.6f} "
                f"batches={batches}",
                flush=True,
            )
    return history


def save_history(path: Path, history: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("epoch", "loss"))
        writer.writeheader()
        writer.writerows(history)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_pt", "--data-pt", dest="data_pt", required=True)
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "gpu"), default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_clusters", "--n-clusters", dest="n_clusters", type=int)
    parser.add_argument("--embedding_dim", "--embedding-dim", dest="embedding_dim", type=int, default=128)
    parser.add_argument("--walk_length", "--walk-length", dest="walk_length", type=int, default=20)
    parser.add_argument("--context_size", "--context-size", dest="context_size", type=int, default=10)
    parser.add_argument("--walks_per_node", "--walks-per-node", dest="walks_per_node", type=int, default=10)
    parser.add_argument("--num_negative_samples", "--num-negative-samples", dest="num_negative_samples", type=int, default=1)
    parser.add_argument("--p", type=float, default=1.0)
    parser.add_argument("--q", type=float, default=1.0)
    parser.add_argument("--directed", type=boolean, default=False)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--num_workers", "--num-workers", dest="num_workers", type=int, default=0)
    parser.add_argument("--kmeans_n_init", "--kmeans-n-init", dest="kmeans_n_init", type=int, default=20)
    parser.add_argument("--kmeans_max_iter", "--kmeans-max-iter", dest="kmeans_max_iter", type=int, default=300)
    parser.add_argument("--log_interval", "--log-interval", dest="log_interval", type=int, default=1)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_integers = {
        "embedding_dim": args.embedding_dim,
        "walk_length": args.walk_length,
        "context_size": args.context_size,
        "walks_per_node": args.walks_per_node,
        "num_negative_samples": args.num_negative_samples,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "kmeans_n_init": args.kmeans_n_init,
        "kmeans_max_iter": args.kmeans_max_iter,
        "log_interval": args.log_interval,
    }
    invalid = [name for name, value in positive_integers.items() if value < 1]
    if invalid:
        raise ValueError(f"These parameters must be positive: {', '.join(invalid)}")
    if args.context_size > args.walk_length:
        raise ValueError("context_size must not exceed walk_length")
    if args.p <= 0 or args.q <= 0 or args.lr <= 0:
        raise ValueError("p, q, and lr must be positive")
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative")


def main() -> int:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    device = resolve_device(args.device)

    data = load_data(args.data_pt)
    edge_index = as_edge_index(get_attr(data, "edge_index"))
    num_nodes = infer_num_nodes(data, edge_index)
    if num_nodes < 2:
        raise ValueError("Node2Vec requires at least two nodes")
    if int(edge_index.min()) < 0 or int(edge_index.max()) >= num_nodes:
        raise ValueError("data.edge_index contains a node outside [0, num_nodes)")
    num_clusters = resolve_num_clusters(data, args.n_clusters, num_nodes)
    if not args.directed:
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)

    try:
        model = Node2Vec(
            edge_index=edge_index,
            embedding_dim=args.embedding_dim,
            walk_length=args.walk_length,
            context_size=args.context_size,
            walks_per_node=args.walks_per_node,
            p=args.p,
            q=args.q,
            num_negative_samples=args.num_negative_samples,
            num_nodes=num_nodes,
            sparse=True,
        ).to(device)
    except ImportError as exc:
        raise RuntimeError(
            "PyG Node2Vec requires a compatible torch_cluster installation"
        ) from exc

    loader = model.loader(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=args.lr)
    print(
        f"backend=torch_geometric.nn.Node2Vec nodes={num_nodes} "
        f"walk_length={args.walk_length} p={args.p} q={args.q}",
        flush=True,
    )
    def select_epoch(epoch):
        with _nmi_selection.inference(model):
            _nmi_selection.kmeans(
                model(torch.arange(num_nodes, device=device)), num_clusters, epoch,
                n_init=args.kmeans_n_init, max_iter=args.kmeans_max_iter,
            )

    history = train_node2vec(
        model,
        loader,
        optimizer,
        args.epochs,
        args.log_interval,
        device,
        selection_callback=select_epoch if _nmi_selection is not None else None,
    )

    model.eval()
    with torch.no_grad():
        node_ids = torch.arange(num_nodes, device=device)
        embeddings = model(node_ids).detach().cpu().numpy().astype(np.float32)
    estimator = KMeans(
        n_clusters=num_clusters,
        init="k-means++",
        n_init=args.kmeans_n_init,
        max_iter=args.kmeans_max_iter,
        random_state=args.seed,
    )
    predicted_labels = estimator.fit_predict(embeddings).astype(np.int64, copy=False)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    labels_path = output_dir / f"Node2VecKMeans_{timestamp}_labels.npy"
    np.save(labels_path, predicted_labels)
    np.save(output_dir / "embeddings.npy", embeddings)
    np.save(output_dir / "cluster_centers.npy", estimator.cluster_centers_)
    save_history(output_dir / "training_history.csv", history)

    print(f"device={device}")
    print(f"requested_clusters={num_clusters}")
    print(f"clusters_found={np.unique(predicted_labels).size}")
    print(f"kmeans_inertia={estimator.inertia_:.6f}")
    print(f"labels={labels_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
