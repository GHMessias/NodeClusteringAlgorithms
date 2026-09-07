from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from tqdm.auto import tqdm


ROOT_DIR = Path(__file__).resolve().parent
UCODE_DIR = ROOT_DIR / "models" / "UCODE"
if str(UCODE_DIR) not in sys.path:
    sys.path.insert(0, str(UCODE_DIR))

from UCODEncoder import GCN  # noqa: E402
from Lossfunction import loss_modularity_trace  # noqa: E402
from main import load_data_pt  # noqa: E402


sns.set_theme(style="whitegrid", context="talk")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate UCoDe NMI across training epochs."
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to one .pt dataset or a directory containing .pt datasets.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/ucode_eval",
        help="Directory where CSVs and plots will be saved.",
    )
    parser.add_argument("--epochs", type=int, default=500, help="Training epochs.")
    parser.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="Evaluate NMI every N epochs.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Training device.",
    )
    parser.add_argument(
        "--kmeans-clusters",
        type=int,
        default=None,
        help="Optional number of clusters for KMeans. Defaults to the number of true labels.",
    )
    parser.add_argument(
        "--hid-units",
        type=int,
        default=16,
        help="Output embedding dimension used by UCoDe.",
    )
    parser.add_argument(
        "--hid-dimension",
        type=int,
        default=512,
        help="Hidden layer size inside the UCoDe encoder.",
    )
    parser.add_argument(
        "--layer1-neurons",
        type=int,
        default=None,
        help="Explicit neuron count for the first linear layer. Overrides --hid-dimension.",
    )
    parser.add_argument(
        "--layer2-neurons",
        type=int,
        default=None,
        help="Explicit neuron count for the second linear layer. Overrides --hid-units.",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="How many independent runs to average.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate used by the Adam optimizer.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.1,
        help="Weight decay used by the Adam optimizer.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed. Each run uses seed + run_index.",
    )
    args = parser.parse_args()
    if args.layer1_neurons is not None:
        args.hid_dimension = args.layer1_neurons
    if args.layer2_neurons is not None:
        args.hid_units = args.layer2_neurons
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_datasets(path_str: str) -> list[Path]:
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    if path.is_file():
        if path.suffix != ".pt":
            raise ValueError(f"Expected a .pt file, got: {path}")
        return [path]

    dataset_paths = sorted(path.rglob("*.pt"))
    if not dataset_paths:
        raise FileNotFoundError(f"No .pt datasets found under: {path}")
    return dataset_paths


def dataset_slug(dataset_path: Path) -> str:
    return dataset_path.stem.replace(" ", "_")


def evaluate_epoch(
    model: GCN,
    features_device: torch.Tensor,
    adj_device: torch.Tensor,
    true_labels: np.ndarray,
    nb_nodes: int,
    hid_units: int,
    n_communities: int,
) -> tuple[float, float]:
    sigmoidlogit = nn.Sigmoid()
    model.eval()
    with torch.no_grad():
        logits = model(features_device, adj_device)
        logits = sigmoidlogit(logits)

    argmax_preds = torch.argmax(logits[0], dim=1).detach().cpu().numpy()
    argmax_nmi = normalized_mutual_info_score(true_labels, argmax_preds)

    cpu_logits = logits.view(nb_nodes, hid_units).detach().cpu().numpy()
    kmeans_preds = KMeans(
        init="k-means++",
        n_clusters=n_communities,
        random_state=0,
    ).fit(cpu_logits).labels_
    kmeans_nmi = normalized_mutual_info_score(true_labels, kmeans_preds)
    return argmax_nmi, kmeans_nmi


def run_single_dataset(
    dataset_path: Path,
    args: argparse.Namespace,
    output_dir: Path,
) -> pd.DataFrame:
    preprocessed_data = load_data_pt(str(dataset_path))
    features, adj, B, true_labels, _, nb_nodes, ft_size, _, m = preprocessed_data
    n_communities = args.kmeans_clusters
    if n_communities is None:
        n_communities = len(np.unique(true_labels))

    features_device = features.to(args.device)
    adj_device = adj.to(args.device)
    B_device = B.to(args.device)

    rows: list[dict[str, float | int | str]] = []

    for run_idx in range(args.num_runs):
        run_seed = args.seed + run_idx
        set_seed(run_seed)

        model = GCN(ft_size, args.hid_units, nb_nodes, args.hid_dimension).to(args.device)
        optimiser = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        progress_desc = f"{dataset_path.name} | run {run_idx + 1}/{args.num_runs}"
        for epoch in tqdm(range(1, args.epochs + 1), desc=progress_desc):
            model.train()
            optimiser.zero_grad()
            logits = model(features_device, adj_device)
            loss = loss_modularity_trace(
                logits,
                B_device,
                n_communities,
                args.hid_units,
                m,
                "non-overlap",
            )
            loss.backward()
            optimiser.step()

            if epoch % args.eval_every != 0 and epoch != args.epochs:
                continue

            argmax_nmi, kmeans_nmi = evaluate_epoch(
                model=model,
                features_device=features_device,
                adj_device=adj_device,
                true_labels=true_labels,
                nb_nodes=nb_nodes,
                hid_units=args.hid_units,
                n_communities=n_communities,
            )
            rows.append(
                {
                    "dataset": dataset_path.stem,
                    "dataset_path": str(dataset_path),
                    "run": run_idx + 1,
                    "seed": run_seed,
                    "epoch": epoch,
                    "loss": float(loss.detach().cpu().item()),
                    "argmax_nmi": float(argmax_nmi),
                    "kmeans_nmi": float(kmeans_nmi),
                    "layer1_neurons": args.hid_dimension,
                    "layer2_neurons": args.hid_units,
                    "hid_units": args.hid_units,
                    "hid_dimension": args.hid_dimension,
                    "kmeans_clusters": int(n_communities),
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                }
            )

    dataset_df = pd.DataFrame(rows)
    csv_path = output_dir / f"{dataset_slug(dataset_path)}_nmi_by_epoch.csv"
    dataset_df.to_csv(csv_path, index=False)
    print(f"saved: {csv_path}")
    return dataset_df


def plot_dataset_results(dataset_df: pd.DataFrame, output_dir: Path) -> Path:
    summary = (
        dataset_df.groupby("epoch", as_index=False)
        .agg(
            argmax_nmi_mean=("argmax_nmi", "mean"),
            argmax_nmi_std=("argmax_nmi", "std"),
            kmeans_nmi_mean=("kmeans_nmi", "mean"),
            kmeans_nmi_std=("kmeans_nmi", "std"),
        )
        .fillna(0.0)
    )

    dataset_name = dataset_df["dataset"].iloc[0]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        summary["epoch"],
        summary["argmax_nmi_mean"],
        label="Argmax NMI",
        linewidth=2.2,
        marker="o",
        markersize=3,
    )
    ax.plot(
        summary["epoch"],
        summary["kmeans_nmi_mean"],
        label="KMeans NMI",
        linewidth=2.2,
        marker="o",
        markersize=3,
    )

    ax.fill_between(
        summary["epoch"],
        summary["argmax_nmi_mean"] - summary["argmax_nmi_std"],
        summary["argmax_nmi_mean"] + summary["argmax_nmi_std"],
        alpha=0.15,
    )
    ax.fill_between(
        summary["epoch"],
        summary["kmeans_nmi_mean"] - summary["kmeans_nmi_std"],
        summary["kmeans_nmi_mean"] + summary["kmeans_nmi_std"],
        alpha=0.15,
    )

    ax.set_title(f"UCoDe NMI by epoch | {dataset_name}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NMI")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()

    plot_path = output_dir / f"{dataset_name}_nmi_by_epoch.png"
    fig.savefig(plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {plot_path}")
    return plot_path


def plot_combined_kmeans(all_results: pd.DataFrame, output_dir: Path) -> Path | None:
    datasets = all_results["dataset"].drop_duplicates().tolist()
    if len(datasets) <= 1:
        return None

    summary = (
        all_results.groupby(["dataset", "epoch"], as_index=False)
        .agg(kmeans_nmi_mean=("kmeans_nmi", "mean"))
    )

    fig, ax = plt.subplots(figsize=(11, 7))
    for dataset_name in datasets:
        dataset_df = summary[summary["dataset"] == dataset_name]
        ax.plot(
            dataset_df["epoch"],
            dataset_df["kmeans_nmi_mean"],
            linewidth=2.2,
            marker="o",
            markersize=3,
            label=dataset_name,
        )

    ax.set_title("UCoDe KMeans NMI by epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NMI")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()

    plot_path = output_dir / "combined_kmeans_nmi_by_epoch.png"
    fig.savefig(plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {plot_path}")
    return plot_path


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested with --device cuda, but it is not available.")
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_paths = resolve_datasets(args.data)
    all_results: list[pd.DataFrame] = []

    for dataset_path in dataset_paths:
        dataset_df = run_single_dataset(dataset_path, args, output_dir)
        if dataset_df.empty:
            print(f"warning: no evaluation rows generated for {dataset_path}")
            continue
        plot_dataset_results(dataset_df, output_dir)
        all_results.append(dataset_df)

    if not all_results:
        raise RuntimeError("No results were produced.")

    combined_df = pd.concat(all_results, ignore_index=True)
    combined_csv_path = output_dir / "all_datasets_nmi_by_epoch.csv"
    combined_df.to_csv(combined_csv_path, index=False)
    print(f"saved: {combined_csv_path}")
    plot_combined_kmeans(combined_df, output_dir)


if __name__ == "__main__":
    main()
