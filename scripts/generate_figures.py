#!/usr/bin/env python3
"""Generate reproducible figures from one completed benchmark run."""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.cluster import HDBSCAN
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from evaluate import as_label_vector, get_attr, load_data, load_pred_labels


IDENTIFICATION_COLUMNS = {
    "dataset",
    "run_id",
    "model",
    "seed",
    "status",
    "error",
}
LOWER_IS_BETTER = {"runtime_seconds", "davies_bouldin"}
METRIC_LABELS = {
    "runtime_seconds": "Runtime (s)",
    "acc": "ACC",
    "nmi": "NMI",
    "ari": "ARI",
    "f1_macro": "F1 macro",
    "f1_micro": "F1 micro",
    "modularity": "Modularidade",
    "n_clusters": "Número de clusters",
    "silhouette": "Silhouette",
    "calinski_harabasz": "Calinski-Harabasz",
    "davies_bouldin": "Davies-Bouldin",
}


@dataclass(frozen=True)
class FigureConfig:
    dpi: int = 180
    max_points: int | None = 5000
    perplexity: float = 30.0
    tsne_iterations: int = 1000


class ProjectionUnavailable(ValueError):
    """Raised when an embedding has no meaningful two-dimensional projection."""


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    return payload if isinstance(payload, dict) else {}


def _save_figure(fig: plt.Figure, path: Path, config: FigureConfig) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _sample_indices(num_nodes: int, max_points: int | None, seed: int) -> np.ndarray:
    indices = np.arange(num_nodes)
    if max_points is None or num_nodes <= max_points:
        return indices
    generator = np.random.default_rng(seed)
    return np.sort(generator.choice(num_nodes, size=max_points, replace=False))


def _embedding_projection(
    embeddings: np.ndarray,
    *,
    seed: int,
    config: FigureConfig,
) -> np.ndarray:
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError(f"Invalid embedding shape for projection: {embeddings.shape}")
    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain NaN or infinite values")

    raw_features = embeddings.astype(np.float64, copy=False)
    # Preserve the learned geometry when the embedding is already a plane.
    if raw_features.shape[1] == 2:
        return raw_features
    if raw_features.shape[0] < 3:
        raise ValueError(f"Invalid embedding shape for projection: {embeddings.shape}")
    centered = raw_features - raw_features.mean(axis=0, keepdims=True)
    centered_energy = float(np.sum(centered * centered))
    reference_energy = max(float(np.sum(raw_features * raw_features)), 1.0)
    numerical_floor = np.finfo(np.float64).eps * reference_energy
    if not np.isfinite(centered_energy) or centered_energy <= numerical_floor:
        raise ProjectionUnavailable(
            "Embedding projection skipped: all nodes have identical or "
            "numerically indistinguishable representations"
        )

    features = StandardScaler().fit_transform(raw_features)
    if not np.isfinite(features).all() or float(np.sum(features * features)) <= np.finfo(np.float64).eps:
        raise ProjectionUnavailable(
            "Embedding projection skipped: standardization produced no usable variance"
        )
    perplexity = min(float(config.perplexity), float(features.shape[0] - 1))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        max_iter=config.tsne_iterations,
        random_state=seed,
    )
    # PCA initialization in scikit-learn divides by the total variance.  Treat
    # that warning as a normal, model-local projection failure rather than
    # letting invalid coordinates reach native t-SNE routines.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message="invalid value encountered in divide",
            category=RuntimeWarning,
        )
        try:
            projection = tsne.fit_transform(features)
        except RuntimeWarning as exc:
            raise ProjectionUnavailable(
                f"Embedding projection skipped because PCA/t-SNE was degenerate: {exc}"
            ) from exc
    if not np.isfinite(projection).all():
        raise ProjectionUnavailable(
            "Embedding projection skipped because t-SNE returned non-finite coordinates"
        )
    return projection


def _plot_partition(
    axis: plt.Axes,
    projection: np.ndarray,
    labels: np.ndarray,
    title: str,
    legend_title: str,
) -> None:
    unique_labels = np.unique(labels)
    colors = plt.get_cmap("tab20", max(len(unique_labels), 1))
    for position, label in enumerate(unique_labels):
        mask = labels == label
        axis.scatter(
            projection[mask, 0],
            projection[mask, 1],
            s=10,
            alpha=0.72,
            color=colors(position),
            label=str(label),
            rasterized=True,
        )
    axis.set_title(title)
    if len(unique_labels) <= 20:
        axis.legend(title=legend_title, fontsize=7, markerscale=1.4)
    else:
        axis.text(
            0.02,
            0.98,
            f"{len(unique_labels)} grupos",
            transform=axis.transAxes,
            ha="left",
            va="top",
        )
    axis.grid(alpha=0.15)


def _densgnn_support_types(
    embeddings: np.ndarray,
    model_params: dict[str, Any],
) -> tuple[np.ndarray, float]:
    required = {
        "hdbscan_min_cluster_size",
        "hdbscan_min_samples",
        "hdbscan_cluster_selection_method",
        "point_probability_threshold",
    }
    missing = sorted(required - set(model_params))
    if missing:
        raise ValueError(f"DensGNN metadata is missing: {', '.join(missing)}")

    clusterer = HDBSCAN(
        min_cluster_size=int(model_params["hdbscan_min_cluster_size"]),
        min_samples=int(model_params["hdbscan_min_samples"]),
        cluster_selection_method=str(model_params["hdbscan_cluster_selection_method"]),
        copy=True,
    ).fit(embeddings)
    threshold = float(model_params["point_probability_threshold"])
    cluster_labels = clusterer.labels_.astype(np.int64)
    probabilities = clusterer.probabilities_
    support_types = np.full(len(embeddings), "ruído", dtype=object)
    non_noise = cluster_labels >= 0
    support_types[non_noise & (probabilities >= threshold)] = "core"
    support_types[non_noise & (probabilities < threshold)] = "border"
    return support_types, threshold


def _plot_support_types(
    axis: plt.Axes,
    projection: np.ndarray,
    support_types: np.ndarray,
    title: str,
) -> None:
    styles = {
        "ruído": {"color": "lightgray", "marker": "x", "s": 15, "alpha": 0.5},
        "border": {"color": "tab:orange", "marker": "^", "s": 17, "alpha": 0.72},
        "core": {"color": "tab:blue", "marker": "o", "s": 10, "alpha": 0.72},
    }
    for support_type in ("ruído", "border", "core"):
        mask = support_types == support_type
        axis.scatter(
            projection[mask, 0],
            projection[mask, 1],
            label=f"{support_type} (n={int(mask.sum())})",
            rasterized=True,
            **styles[support_type],
        )
    axis.set_title(title)
    axis.legend(title="Tipo do ponto", fontsize=8, markerscale=1.4)
    axis.grid(alpha=0.15)


def plot_model_embedding_overview(
    model_dir: Path,
    ground_truth: np.ndarray | None,
    config: FigureConfig,
) -> Path:
    metadata = _load_yaml(model_dir / "metadata.yaml")
    model_name = str(metadata.get("model", model_dir.name))
    seed = int(metadata.get("seed", 0))
    embeddings = np.load(model_dir / "embeddings.npy", allow_pickle=False)
    predicted = load_pred_labels(model_dir / "labels.npy")
    if embeddings.shape[0] != predicted.shape[0]:
        raise ValueError("Embedding and predicted-label counts differ")
    if ground_truth is not None and ground_truth.shape[0] != predicted.shape[0]:
        raise ValueError("Ground-truth and predicted-label counts differ")

    full_embeddings = embeddings
    full_support_types: np.ndarray | None = None
    threshold: float | None = None
    if model_name in {"densgnn", "deep_densgnn"}:
        full_support_types, threshold = _densgnn_support_types(
            full_embeddings,
            dict(metadata.get("model_params", {})),
        )

    indices = _sample_indices(len(predicted), config.max_points, seed)
    embeddings = embeddings[indices]
    predicted = predicted[indices]
    sampled_truth = ground_truth[indices] if ground_truth is not None else None
    support_types = full_support_types[indices] if full_support_types is not None else None
    projection = _embedding_projection(
        embeddings,
        seed=seed,
        config=config,
    )
    direct_embedding = embeddings.shape[1] == 2
    projection_name = "Embedding 2D" if direct_embedding else "t-SNE"
    axis_name = "Dimensão latente" if direct_embedding else "t-SNE"

    columns = 2 + int(support_types is not None)
    fig, axes = plt.subplots(
        1,
        columns,
        figsize=(7 * columns, 6),
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    row = axes[0]
    if sampled_truth is None:
        row[0].text(
            0.5,
            0.5,
            "Ground truth indisponível",
            transform=row[0].transAxes,
            ha="center",
            va="center",
        )
        row[0].set_title(f"{projection_name} — Ground truth")
    else:
        _plot_partition(
            row[0],
            projection,
            sampled_truth,
            f"{projection_name} — Ground truth",
            "Classe",
        )
    _plot_partition(
        row[1],
        projection,
        predicted,
        f"{projection_name} — Clusterização",
        "Cluster",
    )
    if support_types is not None and threshold is not None:
        _plot_support_types(
            row[2],
            projection,
            support_types,
            f"{projection_name} — Core/border (threshold={threshold:g})",
        )
    for axis in row:
        axis.set_xlabel(f"{axis_name} 1")
        if direct_embedding:
            axis.set_aspect("equal", adjustable="box")
    row[0].set_ylabel(f"{axis_name} 2")

    sample_note = (
        f"amostra de {len(indices)}/{len(full_embeddings)} nós"
        if len(indices) < len(full_embeddings)
        else f"{len(indices)} nós"
    )
    projection_description = (
        "embeddings 2D (coordenadas originais)"
        if direct_embedding
        else "projeção t-SNE dos embeddings"
    )
    fig.suptitle(
        f"{model_name} — {projection_description} ({sample_note})",
        fontsize=14,
    )
    return _save_figure(fig, model_dir / "figures" / "embedding_projections.png", config)


def _load_densgnn_algorithm(model_dir: Path) -> dict[str, Any]:
    metrics_path = model_dir / "metrics.json"
    if not metrics_path.is_file():
        return {}
    with metrics_path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    algorithm = payload.get("algorithm", {})
    return algorithm if isinstance(algorithm, dict) else {}


def plot_densgnn_epoch_metrics(model_dir: Path, config: FigureConfig) -> Path | None:
    history = _load_densgnn_algorithm(model_dir).get("evaluation_history", [])
    if not history:
        return None
    frame = pd.DataFrame(history).sort_values("epoch")
    panels = [
        ("nmi", None, "NMI"),
        ("silhouette_embedding", "silhouette_features", "Silhouette"),
        ("calinski_harabasz_embedding", "calinski_harabasz_features", "Calinski-Harabasz"),
        ("davies_bouldin_embedding", "davies_bouldin_features", "Davies-Bouldin"),
        ("num_clusters", None, "Número de clusters"),
        ("noise_ratio", None, "Proporção de ruído"),
    ]
    available = [panel for panel in panels if panel[0] in frame]
    if not available:
        return None

    fig, axes = plt.subplots(3, 2, figsize=(15, 14), squeeze=False)
    for axis, (primary, secondary, title) in zip(axes.ravel(), available):
        axis.plot(frame["epoch"], frame[primary], linewidth=2, label="embedding" if secondary else title)
        if secondary and secondary in frame:
            axis.plot(frame["epoch"], frame[secondary], linewidth=2, label="features originais")
        axis.set(title=title, xlabel="Época", ylabel=title)
        axis.grid(alpha=0.2)
        if secondary:
            axis.legend()
    for axis in axes.ravel()[len(available):]:
        axis.set_visible(False)
    fig.suptitle("DensGNN — avaliação por época", fontsize=15)
    fig.tight_layout()
    return _save_figure(fig, model_dir / "figures" / "epoch_metrics.png", config)


def plot_densgnn_losses(model_dir: Path, config: FigureConfig) -> Path | None:
    history = _load_densgnn_algorithm(model_dir).get("loss_history", [])
    if not history:
        return None
    frame = pd.DataFrame(history).sort_values("epoch")
    losses = [
        ("reconstruction_loss", "Perda de reconstrução"),
        ("clustering_loss", "Perda de clustering"),
        ("total_loss", "Perda total"),
    ]
    available = [item for item in losses if item[0] in frame]
    if not available:
        return None
    fig, axes = plt.subplots(len(available), 1, figsize=(13, 4 * len(available)), squeeze=False, sharex=True)
    for axis, (column, title) in zip(axes.ravel(), available):
        axis.plot(frame["epoch"], frame[column], linewidth=2)
        axis.set(title=title, ylabel=column)
        axis.grid(alpha=0.2)
    axes.ravel()[-1].set_xlabel("Época")
    fig.suptitle("DensGNN — perdas de treinamento", fontsize=15)
    fig.tight_layout()
    return _save_figure(fig, model_dir / "figures" / "loss_history.png", config)


def plot_densgnn_supports(model_dir: Path, config: FigureConfig) -> Path | None:
    statistics = _load_densgnn_algorithm(model_dir).get("cluster_support_stats", [])
    if not statistics:
        return None
    frame = pd.DataFrame(statistics).sort_values("cluster_id")
    positions = np.arange(len(frame))
    core = pd.to_numeric(frame["core_support_points"], errors="coerce").fillna(0).to_numpy()
    border = pd.to_numeric(frame["border_support_points"], errors="coerce").fillna(0).to_numpy()
    cluster_sizes = pd.to_numeric(frame["cluster_size"], errors="coerce").fillna(0).to_numpy()

    fig, axis = plt.subplots(figsize=(max(12, len(frame) * 0.65), 6.5))
    axis.bar(
        positions,
        cluster_sizes,
        width=0.9,
        facecolor="none",
        edgecolor="#333333",
        linewidth=1.4,
        linestyle="--",
        label="Total do cluster",
    )
    axis.bar(positions, core, color="#4C78A8", label="Core")
    axis.bar(positions, border, bottom=core, color="#F58518", label="Border")
    axis.set(
        title="DensGNN — pontos de suporte por cluster",
        xlabel="Cluster HDBSCAN",
        ylabel="Quantidade de pontos",
    )
    axis.set_xticks(positions, frame["cluster_id"].astype(int))
    axis.margins(y=0.18)
    axis.grid(axis="y", alpha=0.2)

    for position, (_, row) in enumerate(frame.iterrows()):
        height = max(float(row["cluster_size"]), float(row["core_support_points"] + row["border_support_points"]))
        axis.text(
            position,
            height,
            f"u={int(row['unique_support_points'])} / total={int(row['cluster_size'])}",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=30 if len(frame) > 12 else 0,
        )

    handles, labels = axis.get_legend_handles_labels()
    alphas = (
        pd.to_numeric(frame["alpha_j"], errors="coerce")
        if "alpha_j" in frame
        else pd.Series(np.nan, index=frame.index)
    )
    if bool(alphas.notna().any()):
        alpha_axis = axis.twinx()
        alpha_line, = alpha_axis.plot(
            positions, alphas, color="#54A24B", marker="o", linewidth=2, label=r"$\alpha_j$"
        )
        alpha_axis.set_ylabel(r"$\alpha_j$")
        alpha_axis.set_ylim(0, 1)
        handles.append(alpha_line)
        labels.append(r"$\alpha_j$")
    axis.legend(handles, labels, loc="upper left")
    fig.tight_layout()
    return _save_figure(fig, model_dir / "figures" / "support_points.png", config)


def _successful_metrics(run_dir: Path) -> pd.DataFrame:
    csv_path = run_dir / "metrics.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Run metrics not found: {csv_path}")
    frame = pd.read_csv(csv_path)
    required = {"model", "status"}
    if not required.issubset(frame):
        raise ValueError(f"Invalid run metrics; missing {sorted(required - set(frame))}")
    frame = frame[frame["status"] == "success"].copy()
    for column in frame.columns:
        if column not in IDENTIFICATION_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _plot_metric_group(
    frame: pd.DataFrame,
    metrics: Iterable[str],
    title: str,
    path: Path,
    config: FigureConfig,
) -> Path | None:
    available = [metric for metric in metrics if metric in frame and frame[metric].notna().any()]
    if not available:
        return None
    columns = min(3, len(available))
    rows = math.ceil(len(available) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(6 * columns, 4.4 * rows), squeeze=False)
    for axis, metric in zip(axes.ravel(), available):
        values = frame[["model", metric]].dropna().sort_values(
            metric, ascending=metric in LOWER_IS_BETTER
        )
        colors = ["#E45756" if model == "densgnn" else "#4C78A8" for model in values["model"]]
        axis.barh(values["model"], values[metric], color=colors, alpha=0.9)
        axis.invert_yaxis()
        axis.set(title=METRIC_LABELS.get(metric, metric), xlabel="Valor na run")
        axis.grid(axis="x", alpha=0.2)
    for axis in axes.ravel()[len(available):]:
        axis.set_visible(False)
    fig.suptitle(title, fontsize=15)
    fig.tight_layout()
    return _save_figure(fig, path, config)


def plot_benchmark_ranking(run_dir: Path, frame: pd.DataFrame, config: FigureConfig) -> Path | None:
    metrics = [
        metric
        for metric in (
            "acc",
            "nmi",
            "ari",
            "f1_macro",
            "f1_micro",
            "modularity",
            "silhouette",
            "calinski_harabasz",
            "davies_bouldin",
        )
        if metric in frame and frame[metric].notna().any()
    ]
    if len(frame) < 2 or not metrics:
        return None
    values = frame.set_index("model")[metrics]
    ranks = pd.DataFrame(index=values.index)
    for metric in metrics:
        ranks[metric] = values[metric].rank(
            method="average", ascending=metric in LOWER_IS_BETTER
        )
    ranks["ranking_medio"] = ranks.mean(axis=1, skipna=True)
    ranks = ranks.sort_values("ranking_medio")
    fig, axis = plt.subplots(figsize=(max(12, len(metrics) * 1.25), max(5, len(ranks) * 0.55)))
    sns.heatmap(ranks, annot=True, fmt=".1f", cmap="YlGnBu_r", ax=axis)
    axis.set(title="Ranking por métrica (1 = melhor)", xlabel="Métrica", ylabel="Modelo")
    fig.tight_layout()
    return _save_figure(fig, run_dir / "figures" / "benchmark_ranking.png", config)


def generate_run_figures(
    run_dir: str | Path,
    data_pt: str | Path,
    *,
    config: FigureConfig | None = None,
) -> dict[str, Any]:
    """Generate all applicable per-model and run-level figures."""

    config = config or FigureConfig()
    run_dir = Path(run_dir).resolve()
    data_pt = Path(data_pt).resolve()
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    # Remove an artifact produced by older generator versions so it cannot be
    # mistaken for part of the current embedding analysis.
    (figures_dir / "embedding_comparison_pca.png").unlink(missing_ok=True)
    report: dict[str, Any] = {
        "run_dir": str(run_dir),
        "data_pt": str(data_pt),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated": [],
        "skipped": [],
        "errors": {},
    }

    data = load_data(data_pt)
    raw_truth = get_attr(data, "y")
    ground_truth = as_label_vector(raw_truth) if raw_truth is not None else None

    model_dirs = sorted(
        path
        for path in run_dir.iterdir()
        if path.is_dir() and path.name != "figures" and (path / "metadata.yaml").is_file()
    )
    for model_dir in model_dirs:
        metadata = _load_yaml(model_dir / "metadata.yaml")
        if metadata.get("status") != "success":
            report["skipped"].append(f"{model_dir.name}: failed execution")
            continue
        try:
            path = plot_model_embedding_overview(model_dir, ground_truth, config)
            report["generated"].append(str(path.relative_to(run_dir)))
        except ProjectionUnavailable as exc:
            report["skipped"].append(
                f"{model_dir.name}.embedding_projections: {exc}"
            )
        except Exception as exc:
            report["errors"][f"{model_dir.name}.embedding_projections"] = f"{type(exc).__name__}: {exc}"

        if model_dir.name in {"densgnn", "deep_densgnn"}:
            densgnn_plots: tuple[tuple[str, Callable[[Path, FigureConfig], Path | None]], ...] = (
                ("epoch_metrics", plot_densgnn_epoch_metrics),
                ("loss_history", plot_densgnn_losses),
                ("support_points", plot_densgnn_supports),
            )
            for name, plotter in densgnn_plots:
                try:
                    path = plotter(model_dir, config)
                    if path is None:
                        report["skipped"].append(f"{model_dir.name}.{name}: data unavailable")
                    else:
                        report["generated"].append(str(path.relative_to(run_dir)))
                except Exception as exc:
                    report["errors"][f"{model_dir.name}.{name}"] = f"{type(exc).__name__}: {exc}"

    try:
        frame = _successful_metrics(run_dir)
        benchmark_plots = (
            (
                "benchmark_quality",
                ("acc", "nmi", "ari", "f1_macro", "f1_micro", "modularity"),
                "Qualidade externa e estrutural",
            ),
            (
                "benchmark_internal",
                ("silhouette", "calinski_harabasz", "davies_bouldin"),
                "Métricas internas nos embeddings",
            ),
            (
                "benchmark_runtime_clusters",
                ("runtime_seconds", "n_clusters"),
                "Custo e granularidade do particionamento",
            ),
        )
        for name, metrics, title in benchmark_plots:
            path = _plot_metric_group(
                frame,
                metrics,
                title,
                figures_dir / f"{name}.png",
                config,
            )
            if path is None:
                report["skipped"].append(f"{name}: metrics unavailable")
            else:
                report["generated"].append(str(path.relative_to(run_dir)))
        ranking_path = plot_benchmark_ranking(run_dir, frame, config)
        if ranking_path is None:
            report["skipped"].append("benchmark_ranking: requires at least two models")
        else:
            report["generated"].append(str(ranking_path.relative_to(run_dir)))
    except Exception as exc:
        report["errors"]["benchmark"] = f"{type(exc).__name__}: {exc}"

    manifest_path = figures_dir / "manifest.yaml"
    with manifest_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(report, stream, sort_keys=False, allow_unicode=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Completed results/<dataset>/runs/<run_id> directory")
    parser.add_argument("--data-pt", required=True, help="Graph data.pt used by the run")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--max-points", type=int, default=5000)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--tsne-iterations", type=int, default=1000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = generate_run_figures(
        args.run_dir,
        args.data_pt,
        config=FigureConfig(
            dpi=args.dpi,
            max_points=args.max_points,
            perplexity=args.perplexity,
            tsne_iterations=args.tsne_iterations,
        ),
    )
    print(f"Generated {len(report['generated'])} figure(s)")
    for path in report["generated"]:
        print(f"  {path}")
    if report["errors"]:
        print("Figure generation completed with errors:")
        for name, error in report["errors"].items():
            print(f"  {name}: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
