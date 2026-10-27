"""Reusable analysis of DeepDensGNN multiscale density logits.

The public entry point is :func:`analyze_logits_root`.  It receives either an
experiment directory or its ``runs`` directory and discovers every successful
DeepDensGNN execution below it.  No ground-truth labels are required.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import adjusted_rand_score


LOWER_IS_BETTER = {
    "davies_bouldin",
    "runtime_seconds",
    "reconstruction_loss",
    "clustering_loss",
    "total_loss",
}

METRIC_PRIORITY = (
    "silhouette",
    "modularity",
    "calinski_harabasz",
    "nmi",
    "ari",
    "acc",
    "davies_bouldin",
)


@dataclass
class LogitsAnalysis:
    runs_root: Path
    export_dir: Path
    model: str
    selection_metric: str
    run_summary: pd.DataFrame
    cluster_diagnostics: pd.DataFrame
    scale_profiles: pd.DataFrame
    loss_history: pd.DataFrame
    audit: pd.DataFrame
    figures: list[Path]


def _repo_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "scripts" / "main.py").is_file():
            return candidate
    raise FileNotFoundError(
        "Repository root not found. Run the notebook inside the repository "
        "or pass an absolute results path."
    )


def resolve_runs_root(path: str | Path) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = _repo_root() / value
    value = value.resolve()
    if value.name != "runs" and (value / "runs").is_dir():
        value = value / "runs"
    if not value.is_dir():
        raise FileNotFoundError(f"Runs directory not found: {value}")
    return value


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / p.sum()
    q = q / q.sum()
    middle = 0.5 * (p + q)

    def kl(left: np.ndarray, right: np.ndarray) -> float:
        mask = left > 0
        return float(np.sum(left[mask] * np.log(left[mask] / right[mask])))

    return 0.5 * kl(p, middle) + 0.5 * kl(q, middle)


def _centered_logits(pi: np.ndarray, temperature: float) -> np.ndarray:
    # Softmax logits are identifiable only up to an additive constant.  The
    # row-centred log-probabilities are therefore the canonical representation.
    logits = temperature * np.log(np.clip(pi, np.finfo(float).tiny, 1.0))
    return logits - logits.mean(axis=1, keepdims=True)


def _encoder_name(algorithm: dict[str, Any], metadata: dict[str, Any]) -> str | None:
    return algorithm.get("encoder_type") or metadata.get("model_params", {}).get(
        "encoder_type"
    )


def _metric_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"dataset", "run_id", "model", "seed", "status", "error"}
    return [column for column in frame.columns if column not in excluded]


def load_logits_runs(
    runs_root: str | Path,
    model: str = "deep_densgnn",
    encoder_filter: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all usable runs and return summary, cluster, profile, loss and audit tables."""

    runs_root = resolve_runs_root(runs_root)
    summaries: list[dict[str, Any]] = []
    clusters: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    losses: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []

    run_dirs = sorted(path for path in runs_root.iterdir() if path.is_dir())
    for run_dir in run_dirs:
        metrics_csv = run_dir / "metrics.csv"
        model_dir = run_dir / model
        metrics_json = model_dir / "metrics.json"
        metadata_yaml = model_dir / "metadata.yaml"
        labels_path = model_dir / "labels.npy"

        base_audit = {"run_id": run_dir.name, "model": model}
        if not metrics_csv.is_file():
            audit.append({**base_audit, "usable": False, "reason": "missing metrics.csv"})
            continue

        try:
            metrics_frame = pd.read_csv(metrics_csv, skipinitialspace=True)
        except Exception as exc:  # malformed results should not hide valid runs
            audit.append(
                {**base_audit, "usable": False, "reason": f"metrics.csv: {exc}"}
            )
            continue

        selected = metrics_frame.loc[metrics_frame["model"].astype(str).str.strip() == model]
        if selected.empty:
            audit.append({**base_audit, "usable": False, "reason": "model row absent"})
            continue
        row = selected.iloc[-1]
        if str(row.get("status", "")).strip().lower() != "success":
            audit.append(
                {
                    **base_audit,
                    "usable": False,
                    "reason": f"status={row.get('status')}: {row.get('error', '')}",
                }
            )
            continue
        if not metrics_json.is_file():
            audit.append(
                {**base_audit, "usable": False, "reason": "missing model metrics.json"}
            )
            continue

        try:
            payload = json.loads(metrics_json.read_text(encoding="utf-8"))
        except Exception as exc:
            audit.append(
                {**base_audit, "usable": False, "reason": f"metrics.json: {exc}"}
            )
            continue

        # Metadata is optional for this analysis; the algorithm payload is not.
        metadata: dict[str, Any] = {}
        if metadata_yaml.is_file():
            try:
                import yaml

                metadata = yaml.safe_load(metadata_yaml.read_text(encoding="utf-8")) or {}
            except Exception:
                metadata = {}

        algorithm = payload.get("algorithm", {})
        encoder = _encoder_name(algorithm, metadata)
        if encoder_filter is not None and encoder != encoder_filter:
            audit.append(
                {
                    **base_audit,
                    "usable": False,
                    "reason": f"encoder={encoder!r}, expected {encoder_filter!r}",
                }
            )
            continue

        scale_info = algorithm.get("density_scale_info") or {}
        if not scale_info.get("active", False):
            audit.append(
                {**base_audit, "usable": False, "reason": "multiscale density inactive"}
            )
            continue

        try:
            candidate_k = np.asarray(scale_info["candidate_k"], dtype=float)
            pi = np.asarray(scale_info["pi"], dtype=float)
            cluster_labels = np.asarray(scale_info["cluster_labels"])
            if pi.ndim != 2 or pi.shape != (len(cluster_labels), len(candidate_k)):
                raise ValueError(
                    f"pi shape {pi.shape} differs from "
                    f"({len(cluster_labels)}, {len(candidate_k)})"
                )
            if np.any(pi < 0) or not np.allclose(pi.sum(axis=1), 1.0, atol=1e-5):
                raise ValueError("pi rows are not valid probability distributions")
        except Exception as exc:
            audit.append({**base_audit, "usable": False, "reason": f"logits: {exc}"})
            continue

        temperature = _number(algorithm.get("density_temperature", 1.0))
        if not np.isfinite(temperature) or temperature <= 0:
            temperature = 1.0
        logits = _centered_logits(pi, temperature)
        k_eff = np.sum(pi * candidate_k[None, :], axis=1)
        entropy = -np.sum(pi * np.log(np.clip(pi, np.finfo(float).tiny, 1.0)), axis=1)
        normalized_entropy = entropy / np.log(len(candidate_k)) if len(candidate_k) > 1 else np.zeros_like(entropy)
        uniform = np.full(len(candidate_k), 1.0 / len(candidate_k))
        js_uniform = np.asarray([_js_divergence(values, uniform) for values in pi])

        summary: dict[str, Any] = {
            "dataset": str(row.get("dataset", payload.get("dataset", runs_root.parent.name))).strip(),
            "run_id": str(row.get("run_id", run_dir.name)).strip(),
            "seed": row.get("seed"),
            "encoder": encoder,
            "temperature": temperature,
            "candidate_k": ",".join(f"{value:g}" for value in candidate_k),
            "n_density_clusters": len(cluster_labels),
            "mean_k_eff": float(k_eff.mean()),
            "std_k_eff": float(k_eff.std(ddof=0)),
            "mean_entropy": float(normalized_entropy.mean()),
            "mean_js_uniform": float(js_uniform.mean()),
            "labels_path": str(labels_path) if labels_path.is_file() else None,
        }
        for column in _metric_columns(metrics_frame):
            summary[column] = _number(row.get(column))
        summaries.append(summary)

        for cluster_index, cluster_label in enumerate(cluster_labels):
            cluster_row = {
                "dataset": summary["dataset"],
                "run_id": summary["run_id"],
                "seed": summary["seed"],
                "cluster_label": cluster_label,
                "k_eff": float(k_eff[cluster_index]),
                "normalized_entropy": float(normalized_entropy[cluster_index]),
                "js_uniform": float(js_uniform[cluster_index]),
            }
            for scale_index, scale in enumerate(candidate_k):
                cluster_row[f"pi_k{scale:g}"] = float(pi[cluster_index, scale_index])
                cluster_row[f"logit_k{scale:g}"] = float(logits[cluster_index, scale_index])
                profiles.append(
                    {
                        "dataset": summary["dataset"],
                        "run_id": summary["run_id"],
                        "seed": summary["seed"],
                        "cluster_label": cluster_label,
                        "k": float(scale),
                        "pi": float(pi[cluster_index, scale_index]),
                        "centered_logit": float(logits[cluster_index, scale_index]),
                    }
                )
            clusters.append(cluster_row)

        for loss in algorithm.get("loss_history", []):
            losses.append(
                {
                    "dataset": summary["dataset"],
                    "run_id": summary["run_id"],
                    "seed": summary["seed"],
                    **{key: _number(value) for key, value in loss.items()},
                }
            )

        audit.append({**base_audit, "usable": True, "reason": "ok"})

    summary_frame = pd.DataFrame(summaries)
    cluster_frame = pd.DataFrame(clusters)
    profile_frame = pd.DataFrame(profiles)
    loss_frame = pd.DataFrame(losses)
    audit_frame = pd.DataFrame(audit)
    if summary_frame.empty:
        reasons = audit_frame.to_string(index=False) if not audit_frame.empty else "no run directories"
        raise RuntimeError(f"No usable {model} run found under {runs_root}.\n{reasons}")
    return summary_frame, cluster_frame, profile_frame, loss_frame, audit_frame


def choose_selection_metric(summary: pd.DataFrame, requested: str | None) -> str:
    candidates: Iterable[str] = (requested,) if requested else METRIC_PRIORITY
    for metric in candidates:
        if metric and metric in summary and pd.to_numeric(summary[metric], errors="coerce").notna().any():
            return metric
    if requested:
        for metric in METRIC_PRIORITY:
            if metric in summary and pd.to_numeric(summary[metric], errors="coerce").notna().any():
                print(
                    f"Metric {requested!r} is unavailable; using {metric!r} to rank runs."
                )
                return metric
    raise ValueError("No usable selection metric was found in metrics.csv.")


def _save(fig: plt.Figure, export_dir: Path, filename: str, figures: list[Path]) -> None:
    path = export_dir / filename
    fig.savefig(path, dpi=180, bbox_inches="tight")
    figures.append(path)
    plt.show()
    plt.close(fig)


def _extreme_runs(summary: pd.DataFrame, metric: str) -> list[tuple[str, str]]:
    ordered = summary.dropna(subset=[metric]).sort_values(
        metric, ascending=metric in LOWER_IS_BETTER
    )
    runs = [(ordered.iloc[0]["run_id"], "best")]
    if len(ordered) > 1 and ordered.iloc[-1]["run_id"] != runs[0][0]:
        runs.append((ordered.iloc[-1]["run_id"], "worst"))
    return runs


def _plot_extremes(
    summary: pd.DataFrame,
    clusters: pd.DataFrame,
    metric: str,
    export_dir: Path,
    figures: list[Path],
) -> None:
    runs = _extreme_runs(summary, metric)

    fig, axes = plt.subplots(2, len(runs), figsize=(6.2 * len(runs), 8), squeeze=False)
    for column, (run_id, rank) in enumerate(runs):
        selected = clusters.loc[clusters["run_id"] == run_id]
        pi_columns = [name for name in selected if name.startswith("pi_k")]
        logit_columns = [name for name in selected if name.startswith("logit_k")]
        value = float(summary.loc[summary["run_id"] == run_id, metric].iloc[0])
        sns.heatmap(selected[pi_columns], cmap="viridis", ax=axes[0, column], cbar=True)
        axes[0, column].set_title(f"{rank}: {metric}={value:.4g}\nmixture weights $\\pi$")
        axes[0, column].set_xlabel("density scale")
        axes[0, column].set_ylabel("HDBSCAN cluster")
        axes[0, column].set_xticklabels([name.replace("pi_k", "k=") for name in pi_columns], rotation=45)
        limit = np.nanmax(np.abs(selected[logit_columns].to_numpy()))
        limit = max(float(limit), 1e-9)
        sns.heatmap(
            selected[logit_columns],
            cmap="coolwarm",
            center=0,
            vmin=-limit,
            vmax=limit,
            ax=axes[1, column],
            cbar=True,
        )
        axes[1, column].set_title(f"centered logits\n{run_id}")
        axes[1, column].set_xlabel("density scale")
        axes[1, column].set_ylabel("HDBSCAN cluster")
        axes[1, column].set_xticklabels([name.replace("logit_k", "k=") for name in logit_columns], rotation=45)
    fig.suptitle("DeepDensGNN multiscale density: best/worst run", y=1.01, fontsize=14)
    fig.tight_layout()
    _save(fig, export_dir, "01_logits_extremes.png", figures)


def _plot_cluster_logit_curves(
    summary: pd.DataFrame,
    profiles: pd.DataFrame,
    metric: str,
    export_dir: Path,
    figures: list[Path],
) -> None:
    """Plot the same final centered logits as the heatmap, one line per cluster."""

    runs = _extreme_runs(summary, metric)
    fig, axes = plt.subplots(
        1, len(runs), figsize=(7.2 * len(runs), 5.8),
        squeeze=False, sharey=True,
    )
    markers = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "*")
    styles = ("-", "--", "-.", ":")
    for ax, (run_id, rank) in zip(axes.flat, runs):
        selected = profiles.loc[profiles["run_id"] == run_id]
        groups = list(selected.groupby("cluster_label", sort=True))
        colors = sns.color_palette("husl", n_colors=len(groups))
        for index, ((cluster_label, values), color) in enumerate(zip(groups, colors)):
            values = values.sort_values("k")
            ax.plot(
                values["k"], values["centered_logit"],
                color=color, marker=markers[index % len(markers)],
                linestyle=styles[index % len(styles)],
                linewidth=1.6, markersize=5, alpha=0.85,
                label=f"Cluster {cluster_label}",
            )
        value = float(summary.loc[summary["run_id"] == run_id, metric].iloc[0])
        seed = summary.loc[summary["run_id"] == run_id, "seed"].iloc[0]
        ax.axhline(0, color="black", linestyle="--", linewidth=0.9, alpha=0.55)
        ax.set_xticks(sorted(selected["k"].unique()))
        ax.set(
            xlabel="Density scale k", ylabel="Centered logit",
            title=f"{rank}: {metric}={value:.4g} | seed {seed}",
        )
        ax.grid(alpha=0.25)
        ax.legend(
            title="HDBSCAN cluster", fontsize=8, ncol=3,
            loc="upper center", bbox_to_anchor=(0.5, -0.15),
        )
    fig.suptitle("DeepDensGNN: final logit curves by cluster", fontsize=14)
    fig.text(
        0.5, 0.01,
        "Lines connect final values across k, not training epochs. "
        "Cluster IDs are local to each run.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    _save(fig, export_dir, "07_cluster_logit_curves.png", figures)


def summarize_scale_profile(profiles: pd.DataFrame) -> pd.DataFrame:
    """Return the curve and shaded band used by the mean scale-weight plot."""
    # First average clusters within a run; then give every run the same weight.
    per_run = profiles.groupby(["run_id", "k"], as_index=False)["pi"].mean()
    aggregate = per_run.groupby("k")["pi"].agg(["mean", "std", "count"]).reset_index()
    aggregate["std"] = aggregate["std"].fillna(0.0)
    aggregate = aggregate.rename(columns={
        "k": "candidate_k", "mean": "mean_mixture_weight",
        "std": "std_mixture_weight", "count": "n_runs",
    })
    aggregate["lower_bound"] = (
        aggregate["mean_mixture_weight"] - aggregate["std_mixture_weight"]
    ).clip(lower=0.0)
    aggregate["upper_bound"] = (
        aggregate["mean_mixture_weight"] + aggregate["std_mixture_weight"]
    )
    aggregate["uniform_weight"] = 1.0 / len(aggregate)
    return aggregate


def _plot_profile(
    profiles: pd.DataFrame, export_dir: Path, figures: list[Path]
) -> pd.DataFrame:
    aggregate = summarize_scale_profile(profiles)
    x = aggregate["candidate_k"].to_numpy(float)
    mean = aggregate["mean_mixture_weight"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(x, mean, marker="o", linewidth=2, label="mean scale weight")
    ax.fill_between(
        x, aggregate["lower_bound"].to_numpy(float),
        aggregate["upper_bound"].to_numpy(float),
        alpha=0.22, label="±1 SD across runs",
    )
    ax.axhline(1 / len(x), color="black", linestyle="--", linewidth=1, label="uniform")
    ax.set(xlabel="candidate k", ylabel="mean mixture weight", title="Learned multiscale density profile")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, export_dir, "02_scale_profile.png", figures)
    return aggregate


def _plot_diagnostics(
    summary: pd.DataFrame,
    clusters: pd.DataFrame,
    metric: str,
    export_dir: Path,
    figures: list[Path],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    diagnostics = [
        ("k_eff", "Effective neighbourhood size"),
        ("normalized_entropy", "Normalized scale entropy"),
        ("js_uniform", "Jensen–Shannon vs. uniform"),
    ]
    for ax, (column, title) in zip(axes, diagnostics):
        sns.boxplot(y=clusters[column], ax=ax, color="#8ecae6", width=0.35)
        sns.stripplot(y=clusters[column], ax=ax, color="#023047", alpha=0.45, size=3)
        ax.set_title(title)
        ax.set_xlabel(f"{len(summary)} usable run(s)")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(f"Density-scale diagnostics (runs ranked by {metric})", y=1.02)
    fig.tight_layout()
    _save(fig, export_dir, "03_scale_diagnostics.png", figures)


def _plot_metric_relationships(
    summary: pd.DataFrame,
    metric: str,
    export_dir: Path,
    figures: list[Path],
) -> None:
    diagnostic_columns = ["mean_k_eff", "mean_entropy", "mean_js_uniform"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, column in zip(axes, diagnostic_columns):
        valid = summary[[column, metric]].dropna()
        ax.scatter(valid[column], valid[metric], s=55, alpha=0.8, color="#31572c")
        if len(valid) >= 3 and valid[column].nunique() > 1 and valid[metric].nunique() > 1:
            rho = valid.corr(method="spearman").iloc[0, 1]
            ax.set_title(f"Spearman ρ={rho:.3f}")
        else:
            ax.set_title("insufficient runs for correlation")
        ax.set(xlabel=column, ylabel=metric)
        ax.grid(alpha=0.25)
    fig.suptitle("Internal metric versus learned scale behaviour", y=1.02)
    fig.tight_layout()
    _save(fig, export_dir, "04_metric_relationships.png", figures)


def _plot_stability(
    summary: pd.DataFrame,
    profiles: pd.DataFrame,
    metric: str,
    export_dir: Path,
    figures: list[Path],
) -> None:
    ordered = summary.sort_values(metric, ascending=metric in LOWER_IS_BETTER)
    run_ids = ordered["run_id"].tolist()
    short = [f"seed {seed}" for seed in ordered["seed"]]
    n_runs = len(run_ids)
    if n_runs < 2:
        return

    ari = np.full((n_runs, n_runs), np.nan)
    js = np.full((n_runs, n_runs), np.nan)
    label_arrays: dict[str, np.ndarray] = {}
    mean_profiles: dict[str, pd.Series] = {}
    for _, row in ordered.iterrows():
        labels_path = row.get("labels_path")
        if isinstance(labels_path, str) and Path(labels_path).is_file():
            label_arrays[row["run_id"]] = np.load(labels_path)
        mean_profiles[row["run_id"]] = (
            profiles.loc[profiles["run_id"] == row["run_id"]]
            .groupby("k")["pi"]
            .mean()
        )

    for i, left in enumerate(run_ids):
        for j, right in enumerate(run_ids):
            if left in label_arrays and right in label_arrays and len(label_arrays[left]) == len(label_arrays[right]):
                ari[i, j] = adjusted_rand_score(label_arrays[left], label_arrays[right])
            common = mean_profiles[left].index.intersection(mean_profiles[right].index)
            if len(common):
                js[i, j] = _js_divergence(
                    mean_profiles[left].loc[common].to_numpy(),
                    mean_profiles[right].loc[common].to_numpy(),
                )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(ari, annot=n_runs <= 12, fmt=".2f", vmin=0, vmax=1, cmap="viridis", xticklabels=short, yticklabels=short, ax=axes[0])
    axes[0].set_title("Partition stability (ARI)")
    sns.heatmap(js, annot=n_runs <= 12, fmt=".3f", vmin=0, cmap="magma_r", xticklabels=short, yticklabels=short, ax=axes[1])
    axes[1].set_title("Mean scale-profile divergence (JS)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    _save(fig, export_dir, "05_run_stability.png", figures)


def _plot_training(
    losses: pd.DataFrame, export_dir: Path, figures: list[Path]
) -> None:
    if losses.empty or "epoch" not in losses:
        return
    fields = [
        field
        for field in ("reconstruction_loss", "clustering_loss", "total_loss", "selected_clusters")
        if field in losses and losses[field].notna().any()
    ]
    if not fields:
        return
    ncols = 2
    nrows = math.ceil(len(fields) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.2 * nrows), squeeze=False)
    for ax, field in zip(axes.flat, fields):
        pivot = losses.pivot_table(index="epoch", columns="run_id", values=field, aggfunc="last")
        median = pivot.median(axis=1)
        q1 = pivot.quantile(0.25, axis=1)
        q3 = pivot.quantile(0.75, axis=1)
        ax.plot(median.index, median, linewidth=2, label="median")
        ax.fill_between(median.index, q1, q3, alpha=0.22, label="IQR")
        ax.set(xlabel="epoch", ylabel=field, title=field.replace("_", " ").title())
        ax.grid(alpha=0.25)
        ax.legend()
    for ax in axes.flat[len(fields):]:
        ax.set_visible(False)
    fig.suptitle("DeepDensGNN training history", y=1.01)
    fig.tight_layout()
    _save(fig, export_dir, "06_training_curves.png", figures)


def analyze_logits_root(
    runs_root: str | Path,
    *,
    model: str = "deep_densgnn",
    selection_metric: str | None = None,
    encoder_filter: str | None = None,
    export_dir: str | Path | None = None,
) -> LogitsAnalysis:
    """Load, validate, plot and export a complete logits analysis."""

    sns.set_theme(style="whitegrid", context="notebook")
    resolved_runs = resolve_runs_root(runs_root)
    if export_dir is None:
        resolved_export = resolved_runs.parent / "logits_analysis"
    else:
        resolved_export = Path(export_dir).expanduser()
        if not resolved_export.is_absolute():
            resolved_export = _repo_root() / resolved_export
        resolved_export = resolved_export.resolve()
    resolved_export.mkdir(parents=True, exist_ok=True)

    summary, clusters, profiles, losses, audit = load_logits_runs(
        resolved_runs, model=model, encoder_filter=encoder_filter
    )
    metric = choose_selection_metric(summary, selection_metric)
    figures: list[Path] = []

    _plot_extremes(summary, clusters, metric, resolved_export, figures)
    _plot_cluster_logit_curves(summary, profiles, metric, resolved_export, figures)
    mean_profile = _plot_profile(profiles, resolved_export, figures)
    _plot_diagnostics(summary, clusters, metric, resolved_export, figures)
    _plot_metric_relationships(summary, metric, resolved_export, figures)
    _plot_stability(summary, profiles, metric, resolved_export, figures)
    _plot_training(losses, resolved_export, figures)

    tables = {
        "run_summary.csv": summary,
        "cluster_scale_diagnostics.csv": clusters,
        "scale_profiles.csv": profiles,
        "mean_scale_profile.csv": mean_profile,
        "loss_history.csv": losses,
        "audit.csv": audit,
    }
    for filename, frame in tables.items():
        frame.to_csv(resolved_export / filename, index=False)

    manifest = {
        "runs_root": str(resolved_runs),
        "model": model,
        "encoder_filter": encoder_filter,
        "selection_metric": metric,
        "usable_runs": int(len(summary)),
        "ignored_runs": int((~audit["usable"]).sum()) if not audit.empty else 0,
        "figures": [path.name for path in figures],
        "tables": list(tables),
        "notes": [
            "Centered logits are reconstructed from pi because softmax logits are identifiable only up to an additive constant.",
            "Cluster logit curves connect final logits across density scales, not epochs; cluster IDs are local to each run.",
            "No ground-truth labels are required; supervised metrics are used only when already present in metrics.csv.",
        ],
    }
    (resolved_export / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Usable runs: {len(summary)} / discovered: {len(audit)}")
    print(f"Selection metric: {metric}")
    print(f"Outputs: {resolved_export}")
    return LogitsAnalysis(
        runs_root=resolved_runs,
        export_dir=resolved_export,
        model=model,
        selection_metric=metric,
        run_summary=summary,
        cluster_diagnostics=clusters,
        scale_profiles=profiles,
        loss_history=losses,
        audit=audit,
        figures=figures,
    )
