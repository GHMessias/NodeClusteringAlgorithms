#!/usr/bin/env python3
"""Compare saved predictions across seeds and alphas, without ground-truth labels.

Uses the existing analysis_alpha/runs_consolidated.csv snapshot by default.
No training is launched and no benchmark metrics are overwritten.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from plot_alpha_noise import NAMES, experiment_notes, resolve


def verify_graphs(runs: pd.DataFrame) -> pd.DataFrame:
    # Loading requires PyTorch/PyG; neither data.y nor the external scores are used.
    from evaluate import as_numpy, get_attr, load_data

    reference_edges = None
    reference_shape = None
    records = []
    for alpha, group in runs.groupby("alpha"):
        paths = group.data_pt.unique()
        if len(paths) != 1:
            raise ValueError(f"More than one dataset at alpha={alpha}: {paths}")
        data = load_data(paths[0])
        features = as_numpy(get_attr(data, "x"))
        edges = as_numpy(get_attr(data, "edge_index"))
        if reference_edges is None:
            reference_edges, reference_shape = edges, features.shape
        same_edges = np.array_equal(reference_edges, edges)
        if not same_edges or features.shape != reference_shape:
            raise ValueError("Cross-alpha node correspondence requires identical indexed topology and feature shape")
        records.append(dict(alpha=alpha, data_pt=paths[0], n_nodes=features.shape[0],
                            n_features=features.shape[1], same_indexed_edges=same_edges))
    return pd.DataFrame(records)


def compare(left: np.ndarray, right: np.ndarray) -> dict:
    return {"ari": adjusted_rand_score(left, right),
            "ami": adjusted_mutual_info_score(left, right, average_method="arithmetic")}


def summarize(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    grouped = frame.groupby(keys)[["ari", "ami"]].agg(["count", "mean", "std", "min", "max"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    return grouped.reset_index()


def plot_summary(within, between, models, alphas, baseline, output):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), squeeze=False)
    colors = plt.get_cmap("tab10")
    for row, metric in enumerate(["ari", "ami"]):
        for column, frame in enumerate([within, between[between.alpha_from == baseline]]):
            x_column = "alpha" if column == 0 else "alpha_to"
            for i, model in enumerate(models):
                group = frame[frame.model == model].sort_values(x_column)
                x = group[x_column].to_numpy(dtype=float)
                mean = group[f"{metric}_mean"].to_numpy(dtype=float)
                std = group[f"{metric}_std"].to_numpy(dtype=float)
                axes[row, column].plot(x, mean, marker="o", color=colors(i),
                                       label=NAMES.get(model, model), linewidth=2)
                axes[row, column].fill_between(x, mean-std, mean+std, color=colors(i), alpha=.12)
            title = "Entre seeds, no mesmo α" if column == 0 else f"Entre alphas, mesma seed — referência α={baseline:g}"
            axes[row, column].set_title(title, fontsize=11)
            axes[row, column].set_xlabel("α — aumento do ruído")
            axes[row, column].set_ylabel(f"{metric.upper()} entre previsões")
            axes[row, column].set_xticks(alphas if column == 0 else [a for a in alphas if a > baseline])
            axes[row, column].axhline(0, color="gray", linewidth=.8, linestyle="--")
            axes[row, column].set_ylim(-.1, 1.05)
            axes[row, column].grid(alpha=.2)
            axes[row, column].spines[["top", "right"]].set_visible(False)
    fig.suptitle("Estabilidade das partições — sem rótulos reais", fontsize=16)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .955), ncol=len(models), frameon=False)
    fig.text(.5, .015,
             "Média ± desvio padrão descritivo; pares entre seeds não são independentes.\n"
             "Concordância não mede qualidade. Há mudanças de parâmetros/K entre alphas; consulte README.md.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .075, 1, .915))
    for extension in ["png", "pdf"]:
        fig.savefig(output / f"stability_overview.{extension}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-csv", default="results/custom_alpha_feat/analysis_alpha/runs_consolidated.csv")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    source = resolve(args.runs_csv)
    output = resolve(args.output_dir) if args.output_dir else source.parent / "stability"
    runs = pd.read_csv(source)
    runs = runs[runs.status == "success"].copy()
    if runs.empty or runs.duplicated(["model", "alpha", "seed"]).any():
        raise ValueError("An unambiguous set of successful model/alpha/seed executions is required")
    graph_audit = verify_graphs(runs)
    labels = {}
    diagnostics = []
    for row in runs.itertuples():
        path = Path(row.metadata_path).parent / "labels.npy"
        partition = np.load(path, allow_pickle=False)
        expected_nodes = int(graph_audit.loc[graph_audit.alpha == row.alpha, "n_nodes"].item())
        if partition.ndim != 1 or len(partition) != expected_nodes:
            raise ValueError(f"Invalid partition shape: {path}")
        if not np.isfinite(partition).all():
            raise ValueError(f"Non-finite labels: {path}")
        labels[row.model, row.alpha, row.seed] = partition
        values, counts = np.unique(partition, return_counts=True)
        probabilities = counts / len(partition)
        diagnostics.append(dict(model=row.model, alpha=row.alpha, seed=row.seed, run_id=row.run_id,
                                n_nodes=len(partition), n_clusters=len(values),
                                largest_cluster_fraction=probabilities.max(),
                                effective_clusters=np.exp(-np.sum(probabilities * np.log(probabilities))),
                                noise_fraction=float(np.mean(partition == -1)), labels_path=str(path)))
    within_pairs = []
    for (model, alpha), group in runs.groupby(["model", "alpha"]):
        seeds = sorted(group.seed.unique())
        for left, right in combinations(seeds, 2):
            within_pairs.append(dict(model=model, alpha=alpha, seed_left=left, seed_right=right,
                                     **compare(labels[model, alpha, left], labels[model, alpha, right])))
    between_pairs = []
    for model, group in runs.groupby("model"):
        for left, right in combinations(sorted(group.alpha.unique()), 2):
            common_seeds = sorted(set(group[group.alpha == left].seed) & set(group[group.alpha == right].seed))
            for seed in common_seeds:
                between_pairs.append(dict(model=model, alpha_from=left, alpha_to=right, seed=seed,
                                          **compare(labels[model, left, seed], labels[model, right, seed])))
    if not within_pairs or not between_pairs:
        raise ValueError("Requires at least two seeds and two alphas to produce both analyses")
    within_pairs = pd.DataFrame(within_pairs)
    between_pairs = pd.DataFrame(between_pairs)
    within = summarize(within_pairs, ["model", "alpha"])
    between = summarize(between_pairs, ["model", "alpha_from", "alpha_to"])
    diagnostics = pd.DataFrame(diagnostics)
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in [
        ("within_alpha_pairs", within_pairs), ("within_alpha_summary", within),
        ("between_alpha_same_seed_pairs", between_pairs), ("between_alpha_same_seed_summary", between),
        ("partition_diagnostics", diagnostics), ("graph_alignment_audit", graph_audit),
    ]:
        frame.to_csv(output / f"{name}.csv", index=False)
    models = sorted(runs.model.unique(), key=lambda m: list(NAMES).index(m) if m in NAMES else len(NAMES))
    alphas = sorted(runs.alpha.unique())
    plot_summary(within, between, models, alphas, alphas[0], output)
    notes = experiment_notes(runs)
    report = [
        "# Estabilidade das partições, sem ground truth", "",
        f"Origem: {source}. São {len(runs)} partições salvas; nenhum treino foi reexecutado.", "",
        "ARI e AMI são calculados entre duas previsões. data.y e as métricas externas salvas não são consultados.",
        "Os scores ignoram a numeração arbitrária dos clusters: 1 indica partições idênticas até renomeação; valores próximos de 0 indicam concordância próxima do acaso.",
        "Referências: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html e https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_mutual_info_score.html", "",
        "## Comparações", "",
        f"- Mesmo alpha: {len(within_pairs)} pares de seeds distintas. Com 10 runs são 45 pares por modelo/alpha.",
        f"- Entre alphas: {len(between_pairs)} comparações de previsões com a mesma seed, para todos os pares de alphas.",
        f"- O gráfico da direita usa α={alphas[0]:g} como referência, não como verdade ou dataset sem ruído.",
        "- Faixas mostram ±1 desvio padrão amostral, sem clipping. São descritivas, não intervalos de confiança; os 45 pares compartilham runs.", "",
        "## Limites de interpretação", "",
        "Todos os datasets possuem a mesma topologia indexada, contagem de nós e formato de atributos. A análise usa a posição do nó como identidade.",
        "Não há identificadores persistentes explícitos nos arquivos; a correspondência depende de o gerador ter preservado a ordem dos nós. A topologia indexada idêntica é evidência dessa correspondência.",
        "Existe apenas um arquivo/draw de atributos por alpha. As 10 seeds variam o treinamento, não são 10 realizações independentes do ruído.",
        "Estabilidade entre alphas inclui o efeito de todas as diferenças entre os experimentos. A mesma seed numérica não garante a mesma trajetória aleatória.",
        "Concordância alta não demonstra qualidade: partições degeneradas também podem ser estáveis. partition_diagnostics.csv registra número de clusters, fração do maior cluster e número efetivo de clusters.",
        "Se existir label -1, ele é tratado como um grupo nesta comparação; noise_fraction informa sua presença, sem exclusão seletiva de nós.", "",
        *[f"- {note}" for note in notes], "",
        "Para robustez sobre realizações independentes de ruído, é preciso gerar vários datasets por alpha e treinar novamente, com parâmetros fixos e identidade dos nós preservada.", "",
        "## Reproduzir", "", "```bash", "venv/bin/python scripts/evaluate_alpha_stability.py", "```", "",
        "CSVs: within_alpha_pairs/summary, between_alpha_same_seed_pairs/summary, partition_diagnostics, graph_alignment_audit.",
        "Figura: stability_overview.png/pdf.",
    ]
    (output / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("Within-alpha ARI means:")
    print(within.pivot(index="alpha", columns="model", values="ari_mean").round(3).to_string())
    print(f"Same-seed ARI means vs alpha={alphas[0]:g}:")
    print(between[between.alpha_from == alphas[0]].pivot(index="alpha_to", columns="model", values="ari_mean").round(3).to_string())
    print(f"Saved {len(within_pairs)} within-alpha pairs and {len(between_pairs)} cross-alpha pairs to {output}")


if __name__ == "__main__":
    main()
