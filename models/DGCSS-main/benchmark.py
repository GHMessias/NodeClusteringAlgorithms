#!/usr/bin/env python3
"""Benchmark adapter for the KMeans-centroid DGCSS variant."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data

from runners.gae_runner import GaeRunner
from utils.b_matrix import BMatrix


def boolean(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def kmeans_n_init(value: str) -> str | int:
    if value.strip().lower() == "auto":
        return "auto"
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("kmeans_n_init must be 'auto' or a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("kmeans_n_init must be positive")
    return parsed


def load_data_pt(path: Path) -> Any:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "data" in payload:
        payload = payload["data"]
    if isinstance(payload, (list, tuple)):
        candidates = [
            value
            for value in payload
            if (
                value.get("x") is not None
                and value.get("edge_index") is not None
                if isinstance(value, dict)
                else getattr(value, "x", None) is not None
                and getattr(value, "edge_index", None) is not None
                and not isinstance(value, type)
            )
        ]
        if len(candidates) != 1:
            raise ValueError("data_pt must contain exactly one graph")
        payload = candidates[0]
    if isinstance(payload, dict):
        payload = Data(**payload)
    if getattr(payload, "x", None) is None or getattr(payload, "edge_index", None) is None:
        raise ValueError("data_pt must provide x and edge_index")
    return payload


def label_vector(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    labels = np.asarray(value)
    if labels.ndim > 1:
        labels = labels.argmax(axis=-1)
    return labels.reshape(-1)


def resolve_num_clusters(data: Any, configured: int | None) -> int:
    if configured is not None:
        num_clusters = configured
    else:
        labels = getattr(data, "y", None)
        if labels is None:
            raise ValueError("--n_clusters is required when data.y is unavailable")
        num_clusters = int(np.unique(label_vector(labels)).size)
    if num_clusters < 2 or num_clusters >= int(data.num_nodes):
        raise ValueError("n_clusters must satisfy 2 <= K < num_nodes")
    return num_clusters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_pt", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "gpu"), default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_clusters", type=int, default=None)
    parser.add_argument("--find_centroids_alg", choices=("KMeans",), default="KMeans")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--c_loss_gama", type=float, default=20.0)
    parser.add_argument("--p_interval", type=int, default=10)
    parser.add_argument("--hidden_layer", type=int, default=64)
    parser.add_argument("--output_layer", type=int, default=16)
    parser.add_argument("--b_matrix_degree", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=0.00005)
    parser.add_argument("--lr_change_gamma", type=float, default=0.95)
    parser.add_argument("--lr_change_epochs", type=int, default=50)
    parser.add_argument("--update_clusters_step_size", type=float, default=0.1)
    parser.add_argument("--rechose_centroids", type=boolean, default=False)
    parser.add_argument("--not_improving_limit", type=int, default=100)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--attention_heads", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--concat", type=boolean, default=True)
    parser.add_argument("--kmeans_random_state", type=int, default=0)
    parser.add_argument("--kmeans_n_init", type=kmeans_n_init, default="auto")
    parser.add_argument("--save_native_plots", type=boolean, default=False)
    parser.add_argument("--debug", type=boolean, default=False)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.DEBUG if args.debug else logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if args.epochs < 1 or args.p_interval < 1 or args.b_matrix_degree < 1:
        raise ValueError("epochs, p_interval, and b_matrix_degree must be positive")

    data_path = Path(args.data_pt).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "native_plots"
    if args.save_native_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)

    data = load_data_pt(data_path)
    num_clusters = resolve_num_clusters(data, args.n_clusters)
    logging.info(
        "DGCSS KMeans input: nodes=%d features=%d clusters=%d",
        int(data.num_nodes),
        int(data.x.shape[1]),
        num_clusters,
    )

    b_matrix = BMatrix(data)
    b_matrix.calc_t_order_neighbors(data, t=args.b_matrix_degree)
    b_matrix.create_edge_index()

    runner = GaeRunner(
        epochs=args.epochs,
        data=data,
        b_edge_index=b_matrix.edge_index,
        n_clusters=num_clusters,
        find_centroids_alg=args.find_centroids_alg,
        c_loss_gama=args.c_loss_gama,
        p_interval=args.p_interval,
        centroids_plot_file=str(plots_dir / "centroids.png"),
        clustering_plot_file=str(plots_dir / "clustering.png"),
        loss_log_file=str(output_dir / "loss_history.csv"),
        metrics_log_file=str(output_dir / "training_metrics.csv"),
        hidden_layer=args.hidden_layer,
        output_layer=args.output_layer,
        device=args.device,
        seed=args.seed,
        learning_rate=args.learning_rate,
        lr_change_gamma=args.lr_change_gamma,
        lr_change_epochs=args.lr_change_epochs,
        update_clusters_step_size=args.update_clusters_step_size,
        rechose_centroids=args.rechose_centroids,
        not_improving_limit=args.not_improving_limit,
        num_layers=args.num_layers,
        attention_heads=args.attention_heads,
        dropout=args.dropout,
        concat=args.concat,
        kmeans_random_state=args.kmeans_random_state,
        kmeans_n_init=args.kmeans_n_init,
        evaluate_during_training=False,
        save_plots=args.save_native_plots,
    )
    runner.run_training()
    if runner.final_labels is None or runner.final_embeddings is None:
        raise RuntimeError("DGCSS did not produce final assignments and embeddings")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    labels_path = output_dir / f"DGCSS_{timestamp}_labels.npy"
    np.save(labels_path, runner.final_labels.astype(np.int64, copy=False))
    np.save(output_dir / "embeddings.npy", runner.final_embeddings)
    logging.info("DGCSS labels saved in %s", labels_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
