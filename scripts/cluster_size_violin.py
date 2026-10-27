"""Cluster-size distributions for exactly the methods listed in a LaTeX table."""

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import yaml

from benchmark_comparison import DATASET_NAMES, MODEL_NAMES


@dataclass
class ClusterDistributions:
    clusters: pd.DataFrame
    runs: pd.DataFrame
    coverage: pd.DataFrame
    model_order: list[str]
    dataset_order: list[str]
    latex_path: Path
    latex_sha256: str


def _inside_results(path, root):
    path = Path(path).expanduser()
    path = (path if path.is_absolute() else root / path).resolve()
    if not path.is_relative_to((root / "results").resolve()):
        raise ValueError(f"Only paths under results/ are allowed: {path}")
    return path


def read_latex_cohort(latex_path):
    """Use the actual model blocks and dataset header, not the larger auto-selection."""
    reverse = {name: key for key, name in MODEL_NAMES.items()}
    models, datasets = [], []
    in_table = False
    for line in latex_path.read_text(encoding="utf-8").splitlines():
        if line.startswith(r"\begin{tabular}"):
            in_table = True
            continue
        if line.startswith(r"\end{tabular}"):
            break
        if not in_table or " & " not in line:
            continue
        cells = [part.strip() for part in line.strip().removesuffix(r"\\").split("&")]
        if cells[0] == "Modelo":
            datasets = cells[2:]
        elif cells[0]:
            if cells[0] not in reverse:
                raise ValueError(f"Unknown model label in LaTeX: {cells[0]}")
            if reverse[cells[0]] not in models:
                models.append(reverse[cells[0]])
    if not models or not datasets:
        raise ValueError("No model blocks/dataset header found in the LaTeX table")
    return models, datasets


def cluster_sizes(labels):
    """Noise (-1) is excluded as a cluster but retained in the node denominator."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or not labels.size or not np.issubdtype(labels.dtype, np.number):
        raise ValueError("Labels must be a nonempty numeric vector")
    if not np.isfinite(labels).all() or not np.equal(labels, np.floor(labels)).all():
        raise ValueError("Labels must be finite integers")
    if (labels < -1).any():
        raise ValueError("Only -1 is supported as a negative/noise label")
    ids, counts = np.unique(labels[labels != -1], return_counts=True)
    return ids, counts, len(labels), int(np.count_nonzero(labels == -1))


def load_cluster_distributions(latex_path, root):
    root = Path(root).resolve()
    latex_path = _inside_results(latex_path, root)
    models, dataset_labels = read_latex_cohort(latex_path)
    folder = latex_path.parent
    attempts = pd.read_csv(folder / "selected_attempts.csv")
    audit = pd.read_csv(folder / "run_audit.csv")
    source_datasets = list(audit.dataset.drop_duplicates())
    dataset_map = {DATASET_NAMES.get(key.removesuffix("_benchmark_adaptive"), key): key for key in source_datasets}
    if set(dataset_labels) - set(dataset_map):
        raise ValueError("The LaTeX datasets differ from the saved run audit")
    datasets = [dataset_map[label] for label in dataset_labels]
    attempts = attempts.loc[attempts.model.isin(models) & attempts.dataset.isin(datasets)].copy()
    if attempts.duplicated(["dataset", "model", "seed"]).any():
        raise ValueError("Duplicate seeds in selected_attempts.csv; refresh the comparison notebook")
    rows, run_rows, expected_nodes = [], [], {}
    for row in attempts.itertuples(index=False):
        entry = {"dataset": row.dataset, "model": row.model, "seed": int(row.seed), "run_id": row.run_id,
                 "status": "not_successful", "detail": "", "n_clusters": 0, "n_nodes": np.nan,
                 "noise_nodes": np.nan, "noise_percent": np.nan, "labels_path": "", "labels_sha256": ""}
        if str(row.status).strip().lower() != "success":
            entry["detail"] = str(row.error)
            run_rows.append(entry)
            continue
        try:
            metrics_path = _inside_results(row.metrics_file, root)
            raw_model = getattr(row, "source_model", row.model)
            model_dir = _inside_results(metrics_path.parent / raw_model, root)
            labels_path = _inside_results(model_dir / "labels.npy", root)
            entry["labels_path"] = str(labels_path)
            ids, counts, n_nodes, noise = cluster_sizes(np.load(labels_path, allow_pickle=False))
            metadata_path = _inside_results(model_dir / "metadata.yaml", root)
            metadata = yaml.safe_load(metadata_path.read_text()) if metadata_path.is_file() else {}
            if not isinstance(metadata, dict):
                raise ValueError("Metadata must be a mapping")
            if metadata and metadata.get("seed") != int(row.seed):
                raise ValueError("Metadata seed differs from the table")
            variant = {"densgnn": "adaptive", "densgnn_core": "core", "densgnn_border": "border"}.get(row.model)
            if variant and metadata.get("model_params", {}).get("point_selection") not in ({"adaptive", "adaptative"} if variant == "adaptive" else {variant}):
                raise ValueError(f"Expected DensGNN {variant} metadata")
            shape = metadata.get("embeddings_shape")
            if shape and int(shape[0]) != n_nodes:
                raise ValueError("Label count differs from metadata embedding shape")
            if row.dataset in expected_nodes and expected_nodes[row.dataset] != n_nodes:
                raise ValueError("Different node counts within the same dataset")
            expected_nodes[row.dataset] = n_nodes
            entry.update(status="loaded" if len(counts) else "noise_only", n_clusters=len(counts), n_nodes=n_nodes,
                         noise_nodes=noise, noise_percent=100 * noise / n_nodes,
                         labels_sha256=hashlib.sha256(labels_path.read_bytes()).hexdigest())
            entry["largest_cluster_percent"] = 100 * counts.max() / n_nodes if len(counts) else np.nan
            for cluster_id, count in zip(ids, counts):
                rows.append({"dataset": row.dataset, "model": row.model, "seed": int(row.seed), "run_id": row.run_id,
                             "cluster_id": int(cluster_id), "cluster_size": int(count), "n_nodes": n_nodes,
                             "cluster_percent": 100 * count / n_nodes, "within_run_weight": 1 / len(counts)})
        except (OSError, ValueError, TypeError, KeyError) as exc:
            entry.update(status="unavailable", detail=str(exc))
        run_rows.append(entry)
    clusters = pd.DataFrame(rows, columns=["dataset", "model", "seed", "run_id", "cluster_id", "cluster_size", "n_nodes", "cluster_percent", "within_run_weight"])
    runs = pd.DataFrame(run_rows, columns=["dataset", "model", "seed", "run_id", "status", "detail", "n_clusters", "n_nodes", "noise_nodes", "noise_percent", "labels_path", "labels_sha256", "largest_cluster_percent"])
    coverage_rows = []
    for dataset in datasets:
        for model in models:
            spec = audit.loc[audit.dataset.eq(dataset) & audit.model.eq(model)]
            selected = runs.loc[runs.dataset.eq(dataset) & runs.model.eq(model)]
            observed = clusters.loc[clusters.dataset.eq(dataset) & clusters.model.eq(model)]
            coverage_rows.append({"dataset": dataset, "model": model,
                                  "expected_runs": int(spec.expected_runs.iloc[0]) if len(spec) else 0,
                                  "plotted_runs": int(selected.status.eq("loaded").sum()),
                                  "noise_only_runs": int(selected.status.eq("noise_only").sum()),
                                  "unavailable_runs": int(selected.status.eq("unavailable").sum()),
                                  "n_cluster_observations": len(observed)})
    return ClusterDistributions(clusters, runs, pd.DataFrame(coverage_rows), models, datasets,
                                latex_path, hashlib.sha256(latex_path.read_bytes()).hexdigest())


def equal_run_weights(frame):
    """Each run has total mass one, then all runs are normalized together."""
    if frame.empty:
        return np.array([], dtype=float)
    counts = frame.groupby("run_id")["cluster_percent"].transform("size").to_numpy(float)
    weights = 1 / counts
    return weights / weights.sum()


def weighted_quantiles(values, weights, probabilities=(.25, .5, .75)):
    order = np.argsort(values)
    values, weights = np.asarray(values)[order], np.asarray(weights)[order]
    positions = (np.cumsum(weights) - .5 * weights) / weights.sum()
    return np.interp(probabilities, positions, values)


def plot_cluster_size_violins(data, *, scale="log", bw_adjust=.7, figsize=(7.6, 6.4)):
    """KDEs describe cluster shares; they are not clustering-quality scores."""
    if scale not in {"log", "linear"} or bw_adjust <= 0:
        raise ValueError("Use scale='log' or 'linear' and bw_adjust > 0")
    transform = np.log10 if scale == "log" else np.asarray
    palette = {model: "#009E73" if model == "densgnn" else "#7E9AB5" for model in data.model_order}
    style = {"font.family": "DejaVu Sans", "font.size": 8.5, "pdf.fonttype": 42,
             "ps.fonttype": 42, "svg.fonttype": "none", "figure.facecolor": "white",
             "axes.facecolor": "white", "text.color": "#18212B", "axes.labelcolor": "#18212B",
             "xtick.color": "#374151", "ytick.color": "#18212B"}
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, len(data.dataset_order), figsize=figsize, sharex=True, sharey=True, squeeze=False)
        axes = axes.flat
        minimum = data.clusters.cluster_percent.min() if not data.clusters.empty else 0.1
        low = min(.01, minimum * .8) if scale == "log" else 0
        x_min, x_max = (np.log10(low), np.log10(115)) if scale == "log" else (0, 105)
        coverage = data.coverage.set_index(["dataset", "model"])
        for panel, (ax, dataset) in enumerate(zip(axes, data.dataset_order)):
            for position, model in enumerate(data.model_order):
                if model == "densgnn":
                    ax.axhspan(position - .46, position + .46, color="#009E73", alpha=.075, zorder=0)
                elif position % 2 == 0:
                    ax.axhspan(position - .46, position + .46, color="#64748B", alpha=.035, zorder=0)
                frame = data.clusters.loc[data.clusters.dataset.eq(dataset) & data.clusters.model.eq(model)]
                info = coverage.loc[(dataset, model)]
                ax.text(1.025, position, f"{int(info.plotted_runs)}/{int(info.expected_runs)}", transform=ax.get_yaxis_transform(),
                        ha="left", va="center", fontsize=7, color="#64748B", clip_on=False)
                if frame.empty:
                    ax.text(.5, position, "No valid clusters" if info.noise_only_runs else "Pending / unavailable",
                            transform=ax.get_yaxis_transform(), ha="center", va="center", fontsize=7, color="#94A3B8")
                    continue
                values = frame.cluster_percent.to_numpy(float)
                weights = equal_run_weights(frame)
                coordinates = transform(values)
                color = palette[model]
                # Never fabricate a density for a singleton or constant sample.
                if len(values) > 1 and np.ptp(coordinates) > 1e-10:
                    try:
                        kde = gaussian_kde(coordinates, weights=weights,
                                           bw_method=lambda estimator: estimator.scotts_factor() * bw_adjust)
                        grid = np.linspace(coordinates.min(), coordinates.max(), 256)
                        density = kde(grid)
                        width = .34 * density / density.max()
                        ax.fill_between(grid, position - width, position + width,
                                        facecolor=color, edgecolor=color, alpha=.65, linewidth=.75, zorder=2)
                    except np.linalg.LinAlgError:
                        ax.plot(coordinates, np.full(len(values), position), "|", color=color)
                else:
                    ax.plot([coordinates[0], coordinates[0]], [position - .23, position + .23], color=color, lw=2.3, zorder=3)
                q1, median, q3 = transform(weighted_quantiles(values, weights))
                ax.plot([q1, q3], [position, position], color="#243342", lw=2.2, solid_capstyle="round", zorder=4)
                ax.scatter([median], [position], s=17, c="white", edgecolors="#243342", linewidths=.7, zorder=5)
            short = dataset.removesuffix("_benchmark_adaptive")
            ax.set_title(f"({chr(97 + panel)})  {DATASET_NAMES.get(short, short)}", loc="left", fontsize=10, fontweight="semibold", pad=16)
            ax.text(1.025, 1.012, "Runs", transform=ax.transAxes, fontsize=7, color="#64748B", clip_on=False)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(len(data.model_order) - .6, -.6)
            ax.grid(axis="x", color="#CBD5E1", linewidth=.5, alpha=.7)
            ax.set_axisbelow(True)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(axis="both", length=0, pad=5)
            if scale == "log":
                ticks = 10. ** np.arange(np.ceil(np.log10(low)), 3)
                ax.set_xticks(np.log10(ticks), [f"{value:g}" for value in ticks])
            else:
                ax.set_xticks([0, 20, 40, 60, 80, 100])
        first = fig.axes[0]
        first.set_yticks(range(len(data.model_order)), [MODEL_NAMES.get(model, model) for model in data.model_order])
        for text, model in zip(first.get_yticklabels(), data.model_order):
            if model == "densgnn":
                text.set_color("#00795B")
                text.set_fontweight("semibold")
        handles = [Patch(facecolor="#7E9AB5", alpha=.65, label="Other selected methods"),
                   Patch(facecolor="#009E73", alpha=.65, label="DensGNN (adaptive)"),
                   Line2D([], [], color="#243342", lw=2.2, label="Interquartile range"),
                   Line2D([], [], marker="o", color="#243342", markerfacecolor="white", markersize=4, linestyle="", label="Median")]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.57, .99), ncol=4, frameon=False, fontsize=7.5)
        fig.supxlabel("Cluster size (% of nodes)" + (" · logarithmic scale" if scale == "log" else ""), x=.57, y=.055, fontsize=9)
        fig.text(.57, .023, "Equal weight per run · Noise (−1) excluded as a cluster · Density trimmed to observed sizes",
                 ha="center", fontsize=7.2, color="#64748B")
        fig.subplots_adjust(left=.19, right=.94, top=.89, bottom=.115, wspace=.38)
    return fig


def export_cluster_distributions(data, figure, output_dir, *, scale, bw_adjust):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"}):
        for extension in ("png", "pdf", "svg"):
            figure.savefig(output_dir / f"cluster_size_violins.{extension}", dpi=400,
                           bbox_inches="tight", facecolor="white")
    data.clusters.to_csv(output_dir / "cluster_sizes.csv", index=False)
    data.runs.to_csv(output_dir / "run_summary.csv", index=False)
    data.coverage.to_csv(output_dir / "coverage.csv", index=False)
    caption = (
        "Cluster-size distributions for the methods included in the selected benchmark tables. "
        "Each observation is the percentage of all graph nodes assigned to one non-noise cluster. "
        "Successful runs with valid label artifacts contribute equal total weight; each cluster within a run receives equal weight. "
        f"Densities are estimated in {'log10 cluster-percentage' if scale == 'log' else 'cluster-percentage'} space "
        "and trimmed to the observed range, with width normalized separately for each violin. "
        "White circles and dark segments indicate weighted medians and interquartile ranges. "
        "Constant distributions are shown as lines. Counts indicate plotted/expected runs. "
        "Noise (-1) is not treated as a cluster and remains in the percentage denominator. "
        "The plot is descriptive; cluster size alone is not a measure of clustering quality."
    )
    (output_dir / "figure_caption.txt").write_text(caption + "\n", encoding="utf-8")
    manifest = {"source_latex": str(data.latex_path), "source_latex_sha256": data.latex_sha256,
                "models_in_latex_order": data.model_order, "datasets": data.dataset_order,
                "scale": scale, "bw_adjust": bw_adjust, "weighting": "equal total mass per run",
                "noise_label": -1, "observations": "cluster cardinalities / total nodes * 100",
                "runs": data.runs.astype(object).where(pd.notna(data.runs), None).to_dict(orient="records")}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
