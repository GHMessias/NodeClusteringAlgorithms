#!/usr/bin/env python3
"""Aggregate existing benchmark runs by dataset noise level; never rerun training.

Example: python scripts/plot_alpha_noise.py
Paths supplied on the command line are relative to the repository root.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "acc": "ACC ↑", "nmi": "NMI ↑", "ari": "ARI ↑",
    "f1_macro": "F1 macro ↑", "f1_micro": "F1 micro ↑",
    "modularity": "Modularidade ↑", "n_clusters": "Número de clusters",
    "runtime_seconds": "Tempo de execução (s) ↓", "silhouette": "Silhouette ↑",
    "calinski_harabasz": "Calinski–Harabasz ↑", "davies_bouldin": "Davies–Bouldin ↓",
}
NAMES = {
    "dgc_efr": "DGC-EFR", "dgcss": "DGCSS", "scgc": "SCGC",
    "densgnn": "DensGNN", "deep_densgnn": "DeepDensGNN",
}


def resolve(path: str) -> Path:
    result = Path(path).expanduser()
    return result if result.is_absolute() else ROOT / result


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def flatten_params(params: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in params.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_params(value, name))
        else:
            flat[name] = json.dumps(value, sort_keys=True)
    return flat


def collect(results: Path, configs: Path, noise_parameter: str = "alpha") -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    sources = {}
    metrics = []
    expected = []
    for path in sorted(configs.glob("*.yaml")):
        config = load_yaml(path)
        experiment = config["experiment"]
        data_name = Path(experiment["data_pt"]).name
        match = re.fullmatch(rf"G_([0-9]+(?:\.[0-9]+)?)_{re.escape(noise_parameter)}_val\.pt", data_name)
        if not match:
            raise ValueError(f"Dataset filename does not identify {noise_parameter}: {path}")
        key = (experiment["dataset"], data_name)
        if key in sources:
            raise ValueError(f"Ambiguous dataset/config mapping: {key}")
        alpha = float(match[1])
        sources[key] = (alpha, str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path))
        metrics.extend(m for m in config["metrics"]["include"] if m not in metrics)
        for model, settings in config["models"].items():
            if settings.get("enabled", False):
                expected.append({noise_parameter: alpha, "model": model, "expected_runs": experiment["runs"]})

    if not sources:
        raise ValueError(f"No YAML configurations found: {configs}")
    rows = []
    for csv_path in sorted((results / "runs").glob("*/metrics.csv")):
        frame = pd.read_csv(csv_path)
        if frame.duplicated(["run_id", "model"]).any():
            raise ValueError(f"Duplicate model executions: {csv_path}")
        for row in frame.to_dict("records"):
            metadata_path = csv_path.parent / row["model"] / "metadata.yaml"
            metadata = load_yaml(metadata_path)
            if (metadata["dataset"], metadata["model"], metadata["seed"], metadata["status"]) != (
                row["dataset"], row["model"], row["seed"], row["status"]
            ):
                raise ValueError(f"CSV/metadata mismatch: {metadata_path}")
            if row["run_id"] != csv_path.parent.name:
                raise ValueError(f"Run ID/path mismatch: {csv_path}")
            key = (metadata["dataset"], Path(metadata["data_pt"]).name)
            if key not in sources:
                continue
            alpha, config_path = sources[key]
            row.update(
                config=config_path, data_pt=metadata["data_pt"],
                metadata_path=str(metadata_path),
                model_params_json=json.dumps(metadata["model_params"], sort_keys=True),
                selected_k=(metadata.get("cluster_count") or {}).get("selected_k"),
                cluster_count_applied=metadata.get("cluster_count_applied", False),
            )
            row[noise_parameter] = alpha
            rows.append(row)
    if not rows:
        raise ValueError(f"No matching runs found: {results}")
    runs = pd.DataFrame(rows).sort_values(["model", noise_parameter, "seed", "run_id"])
    if runs.duplicated([noise_parameter, "model", "seed"]).any():
        raise ValueError(f"Repeated {noise_parameter}/model/seed: select an unambiguous set of experiments")
    for metric in metrics:
        runs[metric] = pd.to_numeric(runs[metric], errors="raise").replace([np.inf, -np.inf], np.nan)
    return runs, metrics, pd.DataFrame(expected)


def aggregate(runs: pd.DataFrame, metrics: list[str], expected: pd.DataFrame, noise_parameter: str = "alpha"):
    good = runs[runs.status == "success"]
    summary = good.groupby([noise_parameter, "model"])[metrics].agg(["count", "mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index().sort_values(["model", noise_parameter])
    counts = runs.assign(
        successful_runs=(runs.status == "success").astype(int),
        failed_runs=(runs.status != "success").astype(int),
    ).groupby([noise_parameter, "model"]).agg(
        observed_runs=("run_id", "size"),
        successful_runs=("successful_runs", "sum"), failed_runs=("failed_runs", "sum"),
    ).reset_index()
    counts = expected.merge(counts, on=[noise_parameter, "model"], how="outer").fillna(0)
    counts["missing_runs"] = (counts.expected_runs - counts.observed_runs).clip(lower=0)
    return summary, counts.sort_values(["model", noise_parameter])


def experiment_notes(runs: pd.DataFrame, noise_parameter: str = "alpha") -> list[str]:
    notes = []
    symbol = "α" if noise_parameter == "alpha" else "ρ"
    for model, group in runs.groupby("model", sort=False):
        flat = [(getattr(row, noise_parameter), flatten_params(json.loads(row.model_params_json)))
                for row in group.itertuples()]
        keys = set().union(*(params.keys() for _, params in flat))
        for key in sorted(keys):
            # Inferred K is reported separately from chosen hyperparameters.
            if key.split(".")[-1] == "n_clusters":
                continue
            values = {params.get(key, "ausente") for _, params in flat}
            if len(values) > 1:
                details = []
                for value in sorted(values):
                    alphas = sorted({alpha for alpha, params in flat if params.get(key, "ausente") == value})
                    details.append(f"{value} em {symbol}={', '.join(f'{a:g}' for a in alphas)}")
                notes.append(f"{NAMES.get(model, model)}: {key} = {'; '.join(details)}.")
    applied = runs[runs.cluster_count_applied]
    if not applied.empty and applied.selected_k.dropna().nunique() > 1:
        details = []
        for alpha, group in applied.groupby(noise_parameter):
            values = sorted(group.selected_k.dropna().astype(int).unique())
            details.append(f"{symbol}={alpha:g}: K={','.join(map(str, values))}")
        models = ", ".join(NAMES.get(m, m) for m in applied.model.unique())
        notes.append(f"K estimado pelo Newman–Riolo ({models}): {'; '.join(details)}.")
    return notes


def plot_metrics(summary, metrics, models, alphas, notes, stem: Path, columns: int,
                 noise_parameter: str = "alpha"):
    rows = int(np.ceil(len(metrics) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(6.1 * columns, 3.8 * rows), squeeze=False)
    colors = plt.get_cmap("tab10")
    markers = ["o", "s", "^", "D", "P", "v", "X"]
    for ax, metric in zip(axes.flat, metrics):
        for i, model in enumerate(models):
            part = summary[summary.model == model].set_index(noise_parameter).reindex(alphas)
            mean = part[f"{metric}_mean"].to_numpy(dtype=float)
            # ddof=1 standard deviation is undefined for n=1; omit its band.
            std = part[f"{metric}_std"].to_numpy(dtype=float)
            ax.plot(alphas, mean, color=colors(i % 10), marker=markers[i % len(markers)],
                    linewidth=2, markersize=5, label=NAMES.get(model, model))
            ax.fill_between(alphas, mean - std, mean + std, color=colors(i % 10), alpha=0.12)
        ax.set_title(LABELS.get(metric, metric), fontweight="bold")
        ax.set_xlabel("α — aumento do ruído" if noise_parameter == "alpha" else "ρ — ruído estrutural (valor no arquivo)")
        ax.set_xticks(alphas)
        if len(alphas) > 6:
            ax.tick_params(axis="x", labelsize=8, rotation=30)
        ax.grid(alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
        if metric in {"calinski_harabasz", "runtime_seconds"}:
            ax.set_yscale("log")
            ax.set_ylabel("Escala logarítmica")
    for ax in list(axes.flat)[len(metrics):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle("Comportamento dos modelos conforme o ruído aumenta", fontsize=16, y=0.98)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.945),
               ncol=min(len(models), 5), frameon=False)
    caption = "Linhas: média das runs bem-sucedidas; faixas: ±1 desvio padrão amostral (não é intervalo de confiança)."
    # Short footnote on every figure; full audit details are in README.md.
    if notes:
        caption += f"\nHá alterações de hiperparâmetros/K entre valores de {noise_parameter}: consulte README.md desta análise."
    fig.text(0.5, 0.015, caption, ha="center", va="bottom", fontsize=9 if columns > 1 else 7)
    fig.tight_layout(rect=(0, 0.095 if rows == 1 else 0.065, 1, 0.87 if rows == 1 else 0.92))
    for extension in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{extension}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(default_noise_parameter: str = "alpha"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noise-parameter", choices=["alpha", "rho"], default=default_noise_parameter)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--configs-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    noise_parameter = args.noise_parameter
    results = resolve(args.results_dir or ("results/custom_alpha_feat" if noise_parameter == "alpha" else "results/custom_rho_val"))
    configs = resolve(args.configs_dir or ("configs/configs_synco" if noise_parameter == "alpha" else "configs/configs_synco_rho"))
    output = resolve(args.output_dir) if args.output_dir else results / f"analysis_{noise_parameter}"
    runs, metrics, expected = collect(results, configs, noise_parameter)
    summary, counts = aggregate(runs, metrics, expected, noise_parameter)
    notes = experiment_notes(runs, noise_parameter)
    output.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output / "runs_consolidated.csv", index=False)
    summary.to_csv(output / f"metrics_by_model_{noise_parameter}.csv", index=False)
    counts.to_csv(output / "run_counts.csv", index=False)
    models = list(runs.model.unique())
    models.sort(key=lambda m: list(NAMES).index(m) if m in NAMES else len(NAMES))
    alphas = sorted(runs[noise_parameter].unique())
    primary = [m for m in ["acc", "nmi", "ari", "f1_macro", "modularity", "n_clusters"] if m in metrics]
    plot_metrics(summary, primary or metrics, models, alphas, notes, output / "overview", 3, noise_parameter)
    if "silhouette" in metrics:
        plot_metrics(summary, [*primary, "silhouette"], models, alphas, notes,
                     output / "overview_with_silhouette", 3, noise_parameter)
    internal = [m for m in ["silhouette", "modularity", "davies_bouldin", "calinski_harabasz", "n_clusters", "runtime_seconds"] if m in metrics]
    if internal:
        plot_metrics(summary, internal, models, alphas, notes, output / "internal_metrics_overview", 3, noise_parameter)
    secondary = [m for m in metrics if m not in primary]
    if secondary:
        plot_metrics(summary, secondary, models, alphas, notes, output / "additional_metrics", 3, noise_parameter)
    for metric in metrics:
        plot_metrics(summary, [metric], models, alphas, notes, output / f"{metric}_vs_{noise_parameter}", 1, noise_parameter)
    command = f"venv/bin/python scripts/plot_{noise_parameter}_noise.py"
    report = [
        f"# Análise de ruído por {noise_parameter}", "",
        f"{len(runs)} execuções; {len(models)} modelos; {noise_parameter}: {', '.join(f'{a:g}' for a in alphas)}.", "",
        f"Cada {noise_parameter} vem do arquivo G_<{noise_parameter}>_{noise_parameter}_val.pt registrado em metadata.yaml, associado à configuração pelo dataset e nome do arquivo.",
        "O eixo usa os valores literais do nome do arquivo; nenhuma conversão para fração ou porcentagem foi presumida.",
        "As métricas são as já salvas em cada runs/<run_id>/metrics.csv; nenhum treino ou avaliação foi reexecutado.", "",
        "As curvas mostram a média e ±1 desvio padrão amostral (ddof=1) das runs bem-sucedidas. Não são intervalos de confiança.",
        "Execuções com falha permanecem no CSV consolidado e na contagem, mas não entram nas médias.",
        "Valores ausentes não entram na média; <metric>_count registra a quantidade válida de cada métrica.",
        "Tempo e Calinski–Harabasz usam escala logarítmica. As métricas internas dos embeddings descrevem espaços aprendidos por modelos diferentes.", "",
        "## Condições dos experimentos", "",
        *([f"- {note}" for note in notes] if notes else ["- Hiperparâmetros registrados e K estimado não variam entre os níveis de ruído."]), "",
        f"Essas curvas descrevem os experimentos executados. Se parâmetros/K variarem, diferenças não podem ser atribuídas exclusivamente ao {noise_parameter}.", "",
        "## Arquivos", "",
        "- overview.png/pdf: painel de ACC, NMI, ARI, F1 macro, modularidade e número de clusters.",
        "- overview_with_silhouette.png/pdf: o mesmo painel, incluindo silhouette.",
        "- internal_metrics_overview.png/pdf: silhouette, modularidade, Davies–Bouldin, Calinski–Harabasz, número de clusters e tempo, sem uso de rótulos reais.",
        "- additional_metrics.png/pdf: demais métricas configuradas.",
        f"- <metric>_vs_{noise_parameter}.png/pdf: gráfico individual por métrica.",
        f"- runs_consolidated.csv: uma linha por execução, agrupada por modelo/{noise_parameter}/seed, com configuração e metadados de origem.",
        f"- metrics_by_model_{noise_parameter}.csv: count, mean e std de cada métrica por modelo e {noise_parameter}.",
        "- run_counts.csv: runs esperadas, observadas, bem-sucedidas, falhas e ausentes.", "",
        "## Reproduzir", "", "```bash", command, "```", "",
        "Argumentos opcionais: --results-dir, --configs-dir e --output-dir; caminhos relativos usam a raiz do repositório.",
    ]
    (output / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(counts.to_string(index=False))
    for note in notes:
        print(note)
    print(f"Saved analysis to {output}")


if __name__ == "__main__":
    main()
