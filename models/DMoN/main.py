#!/usr/bin/env python3
"""Unsupervised node clustering with PyG's DMoNPooling operator."""

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
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import DMoNPooling, GCNConv
from torch_geometric.utils import to_dense_adj, to_undirected


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
    return get_attr(value, "x") is not None or get_attr(value, "edge_index") is not None


def unwrap_data_pt(value: Any) -> Any:
    if isinstance(value, (tuple, list)):
        for item in value:
            if is_graph_data(item):
                return item
        if not value:
            raise ValueError("The data_pt file contains an empty sequence.")
        return value[0]
    if isinstance(value, dict) and "data" in value:
        return value["data"]
    return value


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

    if num_clusters < 2 or num_clusters > num_nodes:
        raise ValueError(
            f"n_clusters must be between 2 and the number of nodes ({num_nodes}); "
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


class DMoNNodeClustering(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        num_clusters: int,
        dropout: float,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.encoder = nn.ModuleList(
            GCNConv(input_channels if layer == 0 else hidden_channels, hidden_channels)
            for layer in range(num_layers)
        )
        self.pool = DMoNPooling(
            channels=[hidden_channels, hidden_channels],
            k=num_clusters,
            dropout=dropout,
        )

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        for convolution in self.encoder:
            x = F.selu(convolution(x, edge_index))
        return x

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        embeddings = self.encode(x, edge_index)
        assignments, _, _, spectral_loss, orthogonality_loss, cluster_loss = self.pool(
            embeddings,
            adjacency,
        )
        return assignments.squeeze(0), spectral_loss, orthogonality_loss, cluster_loss


def save_history(path: Path, history: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "epoch",
                "total_loss",
                "spectral_loss",
                "orthogonality_loss",
                "cluster_loss",
            ],
        )
        writer.writeheader()
        writer.writerows(history)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unsupervised DMoN node clustering")
    parser.add_argument("--data_pt", "--data-pt", dest="data_pt", required=True)
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "gpu"), default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n_clusters", "--n-clusters", dest="n_clusters", type=int, default=None
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--hidden_channels",
        "--hidden-channels",
        dest="hidden_channels",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--num_layers", "--num-layers", dest="num_layers", type=int, default=2
    )
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--weight_decay",
        "--weight-decay",
        dest="weight_decay",
        type=float,
        default=0.0,
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--log_interval", "--log-interval", dest="log_interval", type=int, default=1
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be at least 1")
    if args.hidden_channels < 1:
        raise ValueError("hidden_channels must be at least 1")
    if args.num_layers < 1:
        raise ValueError("num_layers must be at least 1")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    if args.log_interval < 1:
        raise ValueError("log_interval must be at least 1")

    set_seed(args.seed)
    device = resolve_device(args.device)
    data = unwrap_data_pt(torch_load(args.data_pt))
    raw_features = get_attr(data, "x")
    raw_edge_index = get_attr(data, "edge_index")
    if raw_features is None or raw_edge_index is None:
        raise ValueError("DMoN requires data.x and data.edge_index.")

    features = (
        raw_features.to_dense()
        if getattr(raw_features, "is_sparse", False)
        else raw_features
    )
    features = torch.as_tensor(features, dtype=torch.float32)
    if features.ndim != 2:
        raise ValueError("data.x must have shape [num_nodes, num_features]")
    edge_index = torch.as_tensor(raw_edge_index, dtype=torch.long)
    if edge_index.ndim != 2:
        raise ValueError("data.edge_index must be two-dimensional")
    if edge_index.shape[0] != 2:
        edge_index = edge_index.t().contiguous()
    if edge_index.shape[0] != 2:
        raise ValueError("data.edge_index must have shape [2, num_edges]")

    num_nodes = int(features.shape[0])
    if num_nodes < 2:
        raise ValueError("DMoN requires a graph with at least two nodes.")
    if edge_index.numel() == 0:
        raise ValueError("DMoN requires a graph with at least one edge.")
    num_clusters = resolve_num_clusters(data, args.n_clusters, num_nodes)

    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    adjacency = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)
    features = features.to(device)
    edge_index = edge_index.to(device)
    adjacency = adjacency.to(device)

    model = DMoNNodeClustering(
        input_channels=int(features.shape[1]),
        hidden_channels=args.hidden_channels,
        num_clusters=num_clusters,
        dropout=args.dropout,
        num_layers=args.num_layers,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        _, spectral_loss, orthogonality_loss, cluster_loss = model(
            features,
            edge_index,
            adjacency,
        )
        # Match PyG's official DMoN example: optimize spectral modularity and
        # collapse/cluster loss, while retaining orthogonality as a diagnostic.
        total_loss = spectral_loss + cluster_loss
        if not torch.isfinite(total_loss):
            raise RuntimeError(f"DMoN produced a non-finite loss at epoch {epoch}.")
        total_loss.backward()
        optimizer.step()
        if _nmi_selection is not None:
            with _nmi_selection.inference(model):
                candidate, _, _, _ = model(features, edge_index, adjacency)
                candidate_embeddings = model.encode(features, edge_index)
                _nmi_selection.consider(candidate.argmax(dim=-1), candidate_embeddings, epoch)

        record = {
            "epoch": float(epoch),
            "total_loss": float(total_loss.detach().cpu()),
            "spectral_loss": float(spectral_loss.detach().cpu()),
            "orthogonality_loss": float(orthogonality_loss.detach().cpu()),
            "cluster_loss": float(cluster_loss.detach().cpu()),
        }
        history.append(record)
        if epoch == 1 or epoch % args.log_interval == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:04d}/{args.epochs:04d} "
                f"loss={record['total_loss']:.6f} "
                f"spectral={record['spectral_loss']:.6f} "
                f"cluster={record['cluster_loss']:.6f} "
                f"orthogonality={record['orthogonality_loss']:.6f}",
                flush=True,
            )

    model.eval()
    with torch.no_grad():
        assignments, _, _, _ = model(features, edge_index, adjacency)
        embeddings = model.encode(features, edge_index)
    predicted_labels = assignments.argmax(dim=-1).detach().cpu().numpy().astype(np.int64)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    labels_path = output_dir / f"DMoN_{timestamp}_labels.npy"
    np.save(labels_path, predicted_labels)
    np.save(output_dir / "embeddings.npy", embeddings.detach().cpu().numpy())
    save_history(output_dir / "training_history.csv", history)
    print(f"device={device}")
    print(f"requested_clusters={num_clusters}")
    print(f"clusters_found={np.unique(predicted_labels).size}")
    print(f"labels={labels_path}")


if __name__ == "__main__":
    main()
