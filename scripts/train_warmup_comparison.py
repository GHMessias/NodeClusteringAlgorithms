#!/usr/bin/env python3
"""Retrain DensGNN from run metadata and save warm-up/final embeddings."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "models" / "DensGNN" / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
import pandas as pd
import sklearn
import torch
import yaml

from generate_figures import _densgnn_support_types
from graph_benchmark.clustering.densgnn import run_densgnn
from graph_benchmark.datasets.pyg_pt import load_pyg_pt_from_config
from graph_benchmark.utils.densgnn_loss import canonical_point_selection
from graph_benchmark.utils.seed import set_random_seed


def repo_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def export_snapshot(
    output_dir: Path,
    stage: str,
    embeddings: np.ndarray,
    params: dict[str, Any],
    ground_truth: np.ndarray | None,
) -> Path:
    """Classify points on original embeddings, using the figure's definition."""
    point_types, _ = _densgnn_support_types(embeddings, params)
    frame = pd.DataFrame(embeddings, columns=[f"embedding_{i}" for i in range(embeddings.shape[1])])
    frame.insert(0, "point_type", np.where(point_types == "ruído", "noise", point_types))
    frame.insert(0, "node_id", np.arange(len(embeddings)))
    if ground_truth is not None:
        frame.insert(1, "ground_truth", ground_truth)
    path = output_dir / f"{stage}_core_border.csv"
    frame.to_csv(path, index=False)
    return path


def train_comparison(
    source_dir: str | Path,
    *,
    e_warm: int | None = None,
    epochs: int | None = None,
    point_selection: str | None = None,
    point_probability_threshold: float | None = None,
    device: str | None = None,
    seed: int | None = None,
    data_pt: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    """Run one continuous training, using existing metadata as configuration.

    ``epochs`` is the total, including ``e_warm``. Outputs go into a fresh
    directory; an existing directory is rejected to preserve prior results.
    ``point_selection`` overrides the source run's core/border/adaptive mode.
    """
    source_dir = repo_path(source_dir)
    source = yaml.safe_load((source_dir / "metadata.yaml").read_text())
    if source["model"] != "densgnn":
        raise ValueError("This comparison script supports the densgnn model.")
    params = dict(source["model_params"])
    if e_warm is not None:
        params["warmup_epochs"] = e_warm
    if epochs is not None:
        params["epochs"] = epochs
    if point_selection is not None:
        params["point_selection"] = canonical_point_selection(point_selection)
    if point_probability_threshold is not None:
        params["point_probability_threshold"] = float(point_probability_threshold)
    if not 0.0 <= float(params["point_probability_threshold"]) <= 1.0:
        raise ValueError("point_probability_threshold must be between 0 and 1.")
    if not 0 <= int(params["warmup_epochs"]) <= int(params["epochs"]):
        raise ValueError("Require 0 <= e_warm <= epochs (total epochs).")
    if int(params["epochs"]) < 1:
        raise ValueError("Require at least one total training epoch.")
    selected_seed = int(source["seed"] if seed is None else seed)
    selected_device = str(source["device"] if device is None else device)
    params["random_state"] = selected_seed
    params["device"] = selected_device
    graph_path = repo_path(source["data_pt"] if data_pt is None else data_pt)
    bundle = load_pyg_pt_from_config({"path": str(graph_path), "name": source["dataset"]})
    data = bundle["data"]
    ground_truth = None
    if getattr(data, "y", None) is not None:
        ground_truth = data.y.detach().cpu().numpy()
        if ground_truth.ndim == 2 and ground_truth.shape[1] == 1:
            ground_truth = ground_truth.reshape(-1)
        elif ground_truth.ndim > 1:
            ground_truth = ground_truth.argmax(axis=-1)
        if ground_truth.shape != (data.num_nodes,):
            raise ValueError("Ground-truth label count differs from node count.")

    started = datetime.now(timezone.utc)
    if output_dir is None:
        selection = canonical_point_selection(str(params["point_selection"]))
        run_id = started.strftime("%Y%m%d_%H%M%S_%f") + f"_seed{selected_seed}_{selection}_warmup_comparison"
        output_dir = REPO_ROOT / "results" / str(source["dataset"]) / "runs" / run_id / "densgnn"
    output_dir = repo_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, capture_output=True,
    ).stdout.strip() or None
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True, capture_output=True,
    ).stdout.strip()
    metadata = {
        "dataset": source["dataset"], "data_pt": str(graph_path), "model": "densgnn",
        "source_run": str(source_dir), "seed": selected_seed, "device": selected_device,
        "model_params": params, "start_time": started.isoformat(), "status": "running",
        "git_commit": commit, "git_worktree_dirty": bool(dirty),
        "torch_version": str(torch.__version__), "sklearn_version": sklearn.__version__,
        "warmup_snapshot": "after warmup_epochs optimizer steps, eval mode, before clustering",
        "final_snapshot": "after all optimizer steps, eval mode",
        "node_order": "original data.x row order; zero-based node_id",
        "determinism_note": "Same seed does not guarantee identity to historical results across environments.",
    }
    metadata_path = output_dir / "metadata.yaml"

    def write_metadata() -> None:
        metadata_path.write_text(yaml.safe_dump(metadata, sort_keys=False))

    def save_warmup(embeddings: np.ndarray, completed_epochs: int) -> None:
        if embeddings.shape[0] != data.num_nodes:
            raise ValueError("Warm-up embedding count differs from node count.")
        np.save(output_dir / "embeddings_warmup.npy", embeddings)
        metadata["warmup_completed_epochs"] = completed_epochs
        write_metadata()
        print(f"Warm-up snapshot: epoch {completed_epochs}, {embeddings.shape}", flush=True)

    write_metadata()
    clock_start = time.perf_counter()
    try:
        set_random_seed(selected_seed)
        result = run_densgnn(
            bundle, {"params": params}, warmup_callback=save_warmup,
        )
        metadata["training_runtime_seconds"] = time.perf_counter() - clock_start
        final = np.asarray(result["features"])
        warmup = np.load(output_dir / "embeddings_warmup.npy", allow_pickle=False)
        if final.shape != warmup.shape or final.shape[0] != data.num_nodes:
            raise ValueError("Warm-up and final embedding shapes differ.")
        np.save(output_dir / "embeddings.npy", final)
        np.save(output_dir / "labels.npy", result["assignments"])
        np.save(output_dir / "node_ids.npy", np.arange(data.num_nodes))
        pd.DataFrame(result["metadata"]["loss_history"]).to_csv(output_dir / "loss_history.csv", index=False)
        for stage, embeddings in (("warmup", warmup), ("final", final)):
            csv_path = export_snapshot(output_dir, stage, embeddings, params, ground_truth)
            counts = pd.read_csv(csv_path, usecols=["point_type"])["point_type"].value_counts().to_dict()
            metadata[f"{stage}_point_counts"] = {key: int(value) for key, value in counts.items()}
        metadata["actual_device"] = result["metadata"]["device"]
        metadata["cuda_used"] = str(metadata["actual_device"]).startswith("cuda")
        metadata["status"] = "success"
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        metadata["end_time"] = datetime.now(timezone.utc).isoformat()
        metadata["runtime_seconds"] = time.perf_counter() - clock_start
        write_metadata()
    print(f"Saved comparison: {output_dir}", flush=True)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, help="Existing densgnn model directory with metadata.yaml")
    parser.add_argument("--e-warm", type=int, help="Warm-up epochs; defaults to source metadata")
    parser.add_argument("--epochs", type=int, help="Total epochs including warm-up; defaults to source metadata")
    parser.add_argument(
        "--point-selection", choices=("core", "border", "adaptive"),
        help="Support selection for clustering; defaults to source metadata",
    )
    parser.add_argument(
        "--point-probability-threshold", type=float,
        help="Core/border membership probability cutoff in [0, 1]; defaults to source metadata",
    )
    parser.add_argument("--device", help="Defaults to source metadata; use cpu if CUDA is unavailable")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--data-pt")
    parser.add_argument("--output-dir", help="New directory only; relative paths resolve from repository root")
    args = parser.parse_args()
    train_comparison(
        args.source_dir, e_warm=args.e_warm, epochs=args.epochs, device=args.device,
        point_selection=args.point_selection,
        point_probability_threshold=args.point_probability_threshold,
        seed=args.seed, data_pt=args.data_pt, output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
