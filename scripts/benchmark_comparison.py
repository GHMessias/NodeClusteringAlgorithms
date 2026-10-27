"""Build partial benchmark tables and compare means with DensGNN adaptive.

Configuration files define the expected models, seeds, datasets and metrics.
Only completed successful executions contribute to means and sample SDs.
Missing CSVs are pending results, so this module also works before training.
"""

from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

REFERENCE_MODEL = "densgnn"
DEFAULT_TABLE_METRICS = ("nmi", "ari", "acc", "f1_macro")
INTERNAL_TABLE_METRICS = ("silhouette", "calinski_harabasz", "davies_bouldin", "modularity")
METRIC_DIRECTIONS = {
    "nmi": "higher", "ari": "higher", "acc": "higher",
    "f1_macro": "higher", "f1_micro": "higher", "modularity": "higher",
    "silhouette": "higher", "calinski_harabasz": "higher", "dbcv": "higher",
    "runtime_seconds": "lower", "davies_bouldin": "lower", "n_clusters": None,
}
PERCENT_METRICS = {"nmi", "ari", "acc", "f1_macro", "f1_micro"}
METRIC_NAMES = {
    "nmi": "NMI", "ari": "ARI", "acc": "ACC", "f1_macro": "F1-macro",
    "f1_micro": "F1-micro", "runtime_seconds": "Tempo (s)",
    "modularity": "Modularidade", "silhouette": "Silhouette",
    "calinski_harabasz": "CH", "davies_bouldin": "DB", "dbcv": "DBCV",
    "n_clusters": "Clusters",
}
DATASET_NAMES = {
    "cora": "Cora", "citeseer": "CiteSeer", "coraml": "Cora-ML",
    "polblogs": "PolBlogs", "blogcatalog": "BlogCatalog",
    "facebook": "Facebook", "wiki": "Wiki",
}
MODEL_NAMES = {
    "kmeans_attributes": "KMeans (attributes)", "node2vec_kmeans": "Node2Vec + KMeans",
    "gae_kmeans": "GAE + KMeans", "vgae_kmeans": "VGAE + KMeans",
    "arga_kmeans": "ARGA + KMeans", "coane": "CoANE", "coane_topk": "CoANE top-k",
    "dgc_efr": "DGC-EFR", "soli": "SOLI", "ucode": "UCoDe", "ns4gc": "NS4GC",
    "dgcluster": "DGCluster", "dgcss": "DGCSS", "sanece": "SANECE",
    "egc_vgae": "EGC-VGAE", "dsagc": "DSAGC", "scgc": "SCGC", "dmon": "DMoN",
    "densgnn_core": "DensGNN (core)", "densgnn_border": "DensGNN (border)",
    REFERENCE_MODEL: "DensGNN (adaptive)",
}
RELATION_COLORS = {
    "better": "#b91c1c", "worse": "#1d4ed8",
    "missing": "#6b7280", "not_applicable": "#6b7280",
}


def _repo_path(value, root):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def load_catalog(config_dir, root):
    """Group core/border source experiments with their benchmark dataset.

    Both variants save model=densgnn in separate result directories. Keep their
    source identifiers for reading CSVs and give them distinct analysis IDs.
    adaptive.yaml is excluded because densgnn.yaml already defines the reference.
    """
    rows = []
    for path in sorted(Path(config_dir).glob("core_border_adaptive_*/*.yaml")):
        if path.name == "adaptive.yaml":
            continue
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        experiment = config["experiment"]
        variant = path.stem if path.stem in {"core", "border"} else None
        dataset = str(experiment["dataset"])
        if variant:
            reference_path = path.with_name("densgnn.yaml")
            dataset = (
                str(yaml.safe_load(reference_path.read_text(encoding="utf-8"))["experiment"]["dataset"])
                if reference_path.is_file()
                else path.parent.name.removeprefix("core_border_adaptive_") + "_benchmark_adaptive"
            )
        for model, settings in config["models"].items():
            if model == "deep_densgnn" or not settings.get("enabled", False):
                continue
            if variant and (model != "densgnn" or settings.get("params", {}).get("point_selection") != variant):
                raise ValueError(f"A variante {variant} deve habilitar DensGNN com point_selection={variant}: {path}")
            if not variant and model == REFERENCE_MODEL and settings.get("params", {}).get("point_selection") not in {"adaptive", "adaptative"}:
                raise ValueError(f"A referência deve usar point_selection=adaptive: {path}")
            expected = int(experiment["runs"])
            if expected < 1:
                raise ValueError(f"Número de runs inválido: {path}")
            result_path = (_repo_path(experiment["output_root"], root) / experiment["dataset"]).resolve()
            if not result_path.is_relative_to((root / "results").resolve()):
                raise ValueError(f"A análise consulta somente results/: {path}")
            rows.append({
                "dataset": dataset, "model": f"densgnn_{variant}" if variant else model,
                "source_dataset": str(experiment["dataset"]), "source_model": model,
                "expected_runs": expected, "initial_seed": int(experiment["seed"]),
                "result_path": str(result_path),
                "csv_filename": config["output"].get("csv_filename", "metrics.csv"),
                "requested_metrics": tuple(config["metrics"]["include"]), "config_file": str(path),
            })
    if not rows:
        raise ValueError(f"Nenhum YAML de benchmark encontrado em {config_dir}")
    catalog = pd.DataFrame(rows)
    if catalog.duplicated(["dataset", "model"]).any():
        raise ValueError("Há mais de uma configuração para o mesmo dataset/modelo.")
    if catalog.duplicated(["source_dataset", "source_model"]).any():
        raise ValueError("Duas configurações usam o mesmo dataset/modelo de origem; não é possível separar as runs.")
    return catalog


def load_attempts(catalog):
    """Take a snapshot of saved rows, without treating absent results as failures."""
    parts, issues = [], []
    metrics = list(METRIC_DIRECTIONS)
    expected = catalog.set_index(["source_dataset", "source_model"])
    sources = catalog[["source_dataset", "result_path", "csv_filename"]].drop_duplicates()
    for dataset, result_path, filename in sources.itertuples(index=False, name=None):
        for path in sorted((Path(result_path) / "runs").glob(f"*/{filename}")):
            try:
                frame = pd.read_csv(path, skipinitialspace=True)
                if frame.empty:
                    continue
                if not {"model", "seed", "status"}.issubset(frame.columns):
                    raise ValueError("CSV sem model, seed ou status; pode estar sendo escrito")
                for position, values in frame.iterrows():
                    model = str(values["model"]).strip()
                    status = str(values["status"]).strip().lower()
                    seed = pd.to_numeric(values["seed"], errors="coerce")
                    if (dataset, model) not in expected.index:
                        continue
                    spec = expected.loc[(dataset, model)]
                    if not pd.notna(seed) or not np.isfinite(seed) or float(seed) != int(seed):
                        issues.append({"path": str(path), "detail": f"Linha {position}: seed incompleta/inválida"})
                        continue
                    if int(seed) not in range(spec.initial_seed, spec.initial_seed + spec.expected_runs):
                        issues.append({"path": str(path), "detail": f"Seed {seed} fora das runs configuradas"})
                        continue
                    if status not in {"success", "failed", "oot", "timeout"}:
                        issues.append({"path": str(path), "detail": f"Linha {position}: status incompleto/inválido"})
                        continue
                    saved_dataset = values.get("dataset", dataset)
                    if pd.notna(saved_dataset) and str(saved_dataset).strip() != dataset:
                        issues.append({"path": str(path), "detail": "Dataset do CSV difere da configuração"})
                        continue
                    row = {
                        "dataset": spec.dataset, "model": spec.model, "seed": int(seed), "status": status,
                        "source_dataset": dataset, "source_model": model,
                        "run_id": str(values.get("run_id", path.parent.name)),
                        "error": str(values.get("error", "")), "metrics_file": str(path),
                    }
                    for metric in metrics:
                        value = pd.to_numeric(values.get(metric, np.nan), errors="coerce")
                        row[metric] = value if pd.notna(value) and np.isfinite(value) else np.nan
                    parts.append(row)
            except (OSError, ValueError, pd.errors.ParserError) as exc:
                issues.append({"path": str(path), "detail": str(exc)})
    columns = ["dataset", "model", "source_dataset", "source_model", "seed", "status", "run_id", "error", "metrics_file", *metrics]
    attempts = pd.DataFrame(parts, columns=columns)
    attempts["_success"] = attempts["status"].eq("success").astype(int)
    # Prefer the latest successful attempt for each configured seed, even when
    # a later retry failed. Failed and superseded attempts remain in the audit.
    selected = attempts.sort_values(["_success", "run_id", "metrics_file"], kind="stable").drop_duplicates(
        ["dataset", "model", "seed"], keep="last"
    )
    return attempts, selected, pd.DataFrame(issues, columns=["path", "detail"])


def compare_mean(value, reference, metric, is_reference=False):
    if not np.isfinite(value):
        return "missing"
    if is_reference:
        return "reference"
    if METRIC_DIRECTIONS[metric] is None:
        return "no_direction"
    if not np.isfinite(reference):
        return "no_reference"
    if np.isclose(value, reference, rtol=1e-9, atol=1e-12):
        return "tie"
    better = value > reference if METRIC_DIRECTIONS[metric] == "higher" else value < reference
    return "better" if better else "worse"


def _format_cell(row, metric, percent):
    if not row["requested"]:
        return "N/A"
    count, expected = row["n_valid"], row["expected_runs"]
    mean, std = row["mean"], row["std"]
    if not np.isfinite(mean):
        if row["n_completed"] == 0:
            return f"Pendente (0/{expected})"
        if row["n_oot"] == expected:
            return f"OOT (0/{expected})"
        if row["n_success"] == 0:
            return f"Sem sucesso (0/{expected})"
        return f"Indefinido (0/{expected})"
    scale = 100 if percent and metric in PERCENT_METRICS else 1
    decimals = 2 if scale == 100 or metric in {"runtime_seconds", "n_clusters", "calinski_harabasz"} else 4
    text = f"{mean * scale:.{decimals}f}"
    if np.isfinite(std):
        text += f" ± {std * scale:.{decimals}f}"
    # Complete cells match the compact paper layout; partial cells disclose n.
    if count < expected or count == 1:
        text += f" (n={count}/{expected})"
    return text


@dataclass
class ComparisonReport:
    table: pd.DataFrame
    relations: pd.DataFrame
    summary: pd.DataFrame
    audit: pd.DataFrame
    attempts: pd.DataFrame
    selected: pd.DataFrame
    issues: pd.DataFrame
    snapshot: str
    percent: bool
    selection_note: str | None = None

    def for_metrics(self, metrics, percent=None):
        """Make another table from this same snapshot, without rereading runs."""
        metrics = list(dict.fromkeys(metrics))
        if not metrics or set(metrics) - set(self.summary.metric):
            raise ValueError("Inclua as métricas desejadas ao construir o retrato com build_report.")
        percent = self.percent if percent is None else percent
        summary = self.summary.loc[self.summary.metric.isin(metrics)].copy()
        table, relations = format_report_tables(summary, metrics, percent)
        return replace(self, table=table, relations=relations, summary=summary, percent=percent)

    def without_models(self, models):
        """Remove display models without changing means or rerunning selection."""
        excluded = set(models)

        def filtered(frame):
            return frame.loc[~frame.model.isin(excluded)].copy()

        summary = filtered(self.summary)
        if summary.empty:
            raise ValueError("A remoção deixaria a tabela sem modelos.")
        metrics = list(dict.fromkeys(self.summary.metric))
        table, relations = format_report_tables(summary, metrics, self.percent)
        return replace(self, table=table, relations=relations, summary=summary,
                       audit=filtered(self.audit), attempts=filtered(self.attempts),
                       selected=filtered(self.selected))

    def styled(self):
        styles = self.relations.map(lambda relation: f"color: {RELATION_COLORS.get(relation, 'inherit')}")
        reference_rows = [position for position, key in enumerate(self.table.index)
                          if key[0] == MODEL_NAMES[REFERENCE_MODEL]]
        styles.iloc[reference_rows, :] = "color: #ffffff; background-color: #1f2937"
        block_starts = [position + 1 for position, key in enumerate(self.table.index)
                        if position == 0 or key[0] != self.table.index[position - 1][0]]
        style = (
            self.table.style.format(escape="html")
            .apply(lambda _: styles, axis=None)
            .set_properties(**{"text-align": "center", "white-space": "nowrap"})
            .set_table_styles([
                {"selector": "th", "props": [("text-align", "center")]},
                {"selector": "td", "props": [("padding", "4px 10px")]},
                *[{"selector": f"tbody tr:nth-child({position})", "props": [("border-top", "1px solid #333")]}
                  for position in block_starts],
                *[{"selector": f"tbody tr:nth-child({position + 1}) th",
                   "props": [("color", "#ffffff"), ("background-color", "#1f2937")]}
                  for position in reference_rows],
            ])
        )
        return style.set_caption(self.selection_note) if self.selection_note else style


def build_report(config_dir, root, table_metrics=DEFAULT_TABLE_METRICS, percent=True):
    metrics = list(dict.fromkeys(table_metrics))
    if not metrics or set(metrics) - set(METRIC_DIRECTIONS):
        raise ValueError("Selecione métricas suportadas em METRIC_DIRECTIONS.")
    catalog = load_catalog(config_dir, Path(root))
    attempts, selected, issues = load_attempts(catalog)
    audit_rows, summary_rows = [], []
    for spec in catalog.itertuples(index=False):
        chosen = selected.loc[(selected.dataset == spec.dataset) & (selected.model == spec.model)]
        success = chosen.loc[chosen.status == "success"]
        oot = chosen.status.isin(["oot", "timeout"]) | chosen.error.str.contains(r"\bOOT\b|out of time", case=False, regex=True)
        completed = len(chosen)
        state = "Concluído" if completed == spec.expected_runs else "Pendente/em execução"
        if completed == spec.expected_runs and len(success) < completed:
            state = "Concluído com falhas"
        audit = {
            "dataset": spec.dataset, "model": spec.model, "expected_runs": spec.expected_runs,
            "source_dataset": spec.source_dataset, "source_model": spec.source_model,
            "result_path": spec.result_path,
            "n_completed": completed, "n_success": len(success), "n_failed": completed - len(success),
            "n_oot": int(oot.sum()), "n_pending": max(spec.expected_runs - completed, 0), "state": state,
        }
        audit_rows.append(audit)
        for metric in metrics:
            requested = metric in spec.requested_metrics
            values = success[metric].dropna() if requested else pd.Series(dtype=float)
            summary_rows.append({
                **audit, "metric": metric, "requested": requested, "n_valid": len(values),
                "mean": values.mean() if len(values) else np.nan,
                "std": values.std(ddof=1) if len(values) > 1 else np.nan,
            })
    summary = pd.DataFrame(summary_rows)
    audit = pd.DataFrame(audit_rows)
    table, relations = format_report_tables(summary, metrics, percent)
    return ComparisonReport(table, relations, summary, audit, attempts, selected, issues,
                            datetime.now(ZoneInfo("America/Sao_Paulo")).isoformat(timespec="seconds"), percent)


def format_report_tables(summary, metrics, percent):
    variants = ["densgnn_core", "densgnn_border", REFERENCE_MODEL]
    models = [model for model in summary.model.drop_duplicates() if model not in variants]
    models += [model for model in variants if model in set(summary.model)]
    dataset_priority = {name: i for i, name in enumerate(DATASET_NAMES)}
    datasets = sorted(summary.dataset.unique(), key=lambda name: dataset_priority.get(name.removesuffix("_benchmark_adaptive"), 100))
    dataset_labels = [DATASET_NAMES.get(name.removesuffix("_benchmark_adaptive"), name) for name in datasets]
    if len(set(dataset_labels)) != len(dataset_labels):
        dataset_labels = datasets
    metric_labels = {}
    for metric in metrics:
        direction = METRIC_DIRECTIONS[metric]
        arrow = " ↑" if direction == "higher" else (" ↓" if direction == "lower" else "")
        metric_labels[metric] = METRIC_NAMES[metric] + arrow
    index = pd.MultiIndex.from_tuples(
        [(MODEL_NAMES.get(model, model), metric_labels[metric]) for model in models for metric in metrics],
        names=["Modelo", "Métrica"],
    )
    table = pd.DataFrame(index=index, columns=dataset_labels, dtype=object)
    relations = pd.DataFrame("missing", index=index, columns=dataset_labels)
    lookup = summary.set_index(["dataset", "model", "metric"])
    for dataset, label in zip(datasets, dataset_labels):
        for model in models:
            for metric in metrics:
                cell = (MODEL_NAMES.get(model, model), metric_labels[metric])
                key = (dataset, model, metric)
                if key not in lookup.index:
                    table.loc[cell, label] = "N/A"
                    relations.loc[cell, label] = "not_applicable"
                    continue
                row = lookup.loc[key]
                table.loc[cell, label] = _format_cell(row, metric, percent)
                ref_key = (dataset, REFERENCE_MODEL, metric)
                reference = lookup.loc[ref_key, "mean"] if ref_key in lookup.index else np.nan
                relations.loc[cell, label] = (
                    compare_mean(row["mean"], reference, metric, model == REFERENCE_MODEL)
                    if row["requested"] else "not_applicable"
                )
    return table, relations


@dataclass
class TopRankSelection:
    report: ComparisonReport | None
    selected_models: list[str]
    exclusions: pd.DataFrame
    ranks: pd.DataFrame
    pending_reference: pd.DataFrame
    coverage: pd.DataFrame | None = None


def _largest_feasible_subset(better, max_better, condition_groups=None, min_metrics=1):
    """Exact maximum-cardinality subset with monotone rank restrictions.

    Include-first traversal breaks equal-size ties by the input model order.
    The upper bound skips branches that cannot improve the current solution.
    Optional groups require the capacity in only min_metrics conditions each.
    """
    best = []

    def search(position, chosen, counts):
        nonlocal best
        if len(chosen) > len(best):
            best = chosen.copy()
        if len(chosen) + len(better) - position <= len(best):
            return
        next_counts = counts + better[position]
        passes = next_counts <= max_better
        feasible = (np.all(passes) if condition_groups is None else
                    all(np.count_nonzero(passes[group]) >= min_metrics for group in condition_groups))
        if feasible:
            search(position + 1, chosen + [position], next_counts)
        search(position + 1, chosen, counts)

    search(0, [], np.zeros(better.shape[1], dtype=int))
    return best


def select_top_rank_subset(report, datasets, metrics, max_rank=3, *,
                           metric_groups=None, min_metrics_per_group=1):
    """Select the largest common model set keeping adaptive within max_rank.

    By default every dataset/metric must satisfy the limit. metric_groups can
    instead require at least min_metrics_per_group within each group/dataset.
    Means and runs are unchanged. Missing competitor means exclude that competitor;
    missing reference means prevent selection instead of dropping conditions.
    This is a result-conditioned subset, not a ranking of the full benchmark.
    """
    if isinstance(max_rank, bool) or not isinstance(max_rank, int) or max_rank < 1:
        raise ValueError("max_rank deve ser um inteiro positivo.")
    metrics = list(dict.fromkeys(metrics))
    if not metrics or set(metrics) - set(report.summary.metric):
        raise ValueError("Inclua as métricas de seleção ao construir o retrato.")
    if any(METRIC_DIRECTIONS[metric] is None for metric in metrics):
        raise ValueError("A seleção exige métricas com direção de melhora definida.")
    if metric_groups is not None:
        grouped_metrics = [metric for group in metric_groups.values() for metric in group]
        if set(grouped_metrics) != set(metrics) or len(grouped_metrics) != len(metrics):
            raise ValueError("Os grupos devem dividir as métricas selecionadas sem repetições.")
        if (isinstance(min_metrics_per_group, bool) or not isinstance(min_metrics_per_group, int)
                or min_metrics_per_group < 1
                or any(len(group) < min_metrics_per_group for group in metric_groups.values())):
            raise ValueError("min_metrics_per_group deve caber em todos os grupos e ser positivo.")
    dataset_keys = []
    for name in dict.fromkeys(datasets):
        matches = [key for key in report.summary.dataset.unique()
                   if name in {key, key.removesuffix("_benchmark_adaptive")}]
        if len(matches) != 1:
            raise ValueError(f"Dataset não encontrado ou ambíguo: {name}")
        dataset_keys.append(matches[0])
    if not dataset_keys:
        raise ValueError("Selecione pelo menos um dataset.")
    conditions = [(dataset, metric) for dataset in dataset_keys for metric in metrics]
    condition_groups = None if metric_groups is None else [
        [index for index, (key, metric) in enumerate(conditions) if key == dataset and metric in group]
        for dataset in dataset_keys for group in metric_groups.values()
    ]
    relevant = report.summary.loc[report.summary.dataset.isin(dataset_keys) & report.summary.metric.isin(metrics)]
    lookup = relevant.set_index(["dataset", "model", "metric"])
    models = sorted(set(relevant.model) - {REFERENCE_MODEL})

    def mean(dataset, model, metric):
        key = (dataset, model, metric)
        return lookup.loc[key, "mean"] if key in lookup.index else np.nan

    pending = [
        {"dataset": dataset, "metric": metric, "reason": "Adaptive sem média válida; aguarde resultados."}
        for dataset, metric in conditions if not np.isfinite(mean(dataset, REFERENCE_MODEL, metric))
    ]
    if pending:
        return TopRankSelection(None, [], pd.DataFrame(), pd.DataFrame(), pd.DataFrame(pending))

    eligible, excluded = [], []
    for model in models:
        missing = [f"{dataset}/{metric}" for dataset, metric in conditions
                   if not np.isfinite(mean(dataset, model, metric))]
        if missing:
            excluded.append({"model": model, "reason": "Dados insuficientes", "details": "; ".join(missing)})
        else:
            eligible.append(model)
    better = np.array([
        [compare_mean(mean(dataset, model, metric), mean(dataset, REFERENCE_MODEL, metric), metric) == "better"
         for dataset, metric in conditions] for model in eligible
    ], dtype=int).reshape(len(eligible), len(conditions))
    chosen = [eligible[position] for position in _largest_feasible_subset(
        better, max_rank - 1, condition_groups, min_metrics_per_group
    )]
    for model in eligible:
        if model not in chosen:
            excluded.append({"model": model, "reason": "Limite de posição no subconjunto máximo", "details": ""})
    selected_models = chosen + [REFERENCE_MODEL]
    selected_summary = relevant.loc[relevant.model.isin(selected_models)].copy()
    table, relations = format_report_tables(selected_summary, metrics, report.percent)

    def selected_rows(frame):
        return frame.loc[frame.dataset.isin(dataset_keys) & frame.model.isin(selected_models)].copy()

    criterion = ("em todas as condições selecionadas" if metric_groups is None else
                 f"em pelo menos {min_metrics_per_group} métrica(s) de cada grupo por dataset")
    subset = replace(report, table=table, relations=relations, summary=selected_summary,
                     audit=selected_rows(report.audit), attempts=selected_rows(report.attempts),
                     selected=selected_rows(report.selected),
                     selection_note=f"Subconjunto condicionado aos resultados: adaptive até a {max_rank}ª posição "
                                    f"{criterion}; não é o benchmark completo.")
    rank_rows = []
    for dataset, metric in conditions:
        reference = mean(dataset, REFERENCE_MODEL, metric)
        better_all = [model for model in models if compare_mean(mean(dataset, model, metric), reference, metric) == "better"]
        better_selected = [model for model in chosen if compare_mean(mean(dataset, model, metric), reference, metric) == "better"]
        ref = lookup.loc[(dataset, REFERENCE_MODEL, metric)]
        rank_rows.append({
            "dataset": dataset, "metric": metric, "rank_selected": 1 + len(better_selected),
            "rank_all_available": 1 + len(better_all), "n_models_selected": len(selected_models),
            "n_models_available": 1 + sum(np.isfinite(mean(dataset, model, metric)) for model in models),
            "adaptive_n_valid": ref.n_valid, "expected_runs": ref.expected_runs,
            "better_selected": ", ".join(better_selected),
            "within_limit": 1 + len(better_selected) <= max_rank,
        })
    ranks = pd.DataFrame(rank_rows)
    groups = metric_groups if metric_groups is not None else {"all_metrics": metrics}
    coverage_rows = []
    for dataset in dataset_keys:
        for name, group in groups.items():
            values = ranks.loc[ranks.dataset.eq(dataset) & ranks.metric.isin(group)]
            required = min_metrics_per_group if metric_groups is not None else len(group)
            coverage_rows.append({
                "dataset": dataset, "metric_group": name, "n_metrics": len(values),
                "n_within_limit": int(values.within_limit.sum()), "required": required,
                "best_rank": int(values.rank_selected.min()),
                "qualifying_metrics": ", ".join(values.loc[values.within_limit, "metric"]),
            })
    coverage = pd.DataFrame(coverage_rows)
    assert coverage.n_within_limit.ge(coverage.required).all()
    return TopRankSelection(subset, selected_models,
                            pd.DataFrame(excluded, columns=["model", "reason", "details"]),
                            ranks, pd.DataFrame(columns=["dataset", "metric", "reason"]), coverage)


def _latex_escape(value):
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
        "±": r"$\pm$", "↑": r"$\uparrow$", "↓": r"$\downarrow$",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def export_report(report, export_dir, *, table_label="tab:benchmark-densgnn-adaptive", latex_style="colored"):
    """Export tables; ranked LaTeX highlights the two largest distinct means."""
    if latex_style not in {"colored", "ranked"}:
        raise ValueError("latex_style deve ser colored ou ranked.")
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in {
        "comparison_formatted": report.table, "comparison_relations": report.relations,
        "summary_numeric": report.summary, "run_audit": report.audit,
        "all_attempts": report.attempts, "selected_attempts": report.selected, "load_issues": report.issues,
    }.items():
        frame.to_csv(export_dir / f"{name}.csv", index=name.startswith("comparison_"))
    legend = (
        "Vermelho: média melhor que DensGNN (adaptive). Azul: média pior. "
        "Empates, referência e comparações sem referência ficam sem destaque. "
        "↑ maior é melhor; ↓ menor é melhor. Cores não representam significância estatística. "
        "n=k/N indica média parcial ou menos valores válidos; N/A é métrica ou variante não configurada. "
        "Pendente inclui execuções ainda não iniciadas ou em andamento. "
        "Falhas e OOT não entram nas médias. Atualize a análise para incluir novas runs."
    )
    if report.percent:
        legend += " NMI, ARI, ACC e F1 são apresentados em porcentagem (média e desvio multiplicados por 100)."
    html = (
        '<!doctype html><html lang="pt-BR"><meta charset="utf-8">'
        '<title>Benchmark versus DensGNN adaptive</title>'
        '<style>body{font-family:serif;margin:24px}table{border-collapse:collapse}'
        'th,td{border-bottom:1px solid #ddd}p{max-width:1100px}</style>'
        f'<h1>Benchmark versus DensGNN adaptive</h1><p>Retrato: {report.snapshot}</p><p>{legend}</p>'
        + report.styled().to_html() + '</html>'
    )
    (export_dir / "comparison_vs_densgnn.html").write_text(html, encoding="utf-8")
    latex_frame = report.table.map(_latex_escape)
    if latex_style == "colored":
        for row in range(len(latex_frame)):
            for column in range(len(latex_frame.columns)):
                relation = report.relations.iat[row, column]
                if relation in {"better", "worse"}:
                    color = RELATION_COLORS[relation].lstrip("#").upper()
                    latex_frame.iat[row, column] = rf"\textcolor[HTML]{{{color}}}{{{latex_frame.iat[row, column]}}}"
    else:
        # Numeric means drive the emphasis, never rounded/display strings.
        # Each comparison contains one dataset and one metric across models.
        metrics = list(dict.fromkeys(report.summary.metric))
        datasets = sorted(report.summary.dataset.unique(), key=lambda name: {
            key: index for index, key in enumerate(DATASET_NAMES)
        }.get(name.removesuffix("_benchmark_adaptive"), 100))
        labels = [DATASET_NAMES.get(name.removesuffix("_benchmark_adaptive"), name) for name in datasets]
        if len(set(labels)) != len(labels):
            labels = datasets
        for dataset, label in zip(datasets, labels):
            for metric in metrics:
                group = report.summary.loc[report.summary.dataset.eq(dataset) & report.summary.metric.eq(metric)]
                finite = group.loc[group.requested & np.isfinite(group["mean"])]
                values = sorted(finite["mean"].tolist(), reverse=True)
                if not values:
                    continue
                largest = values[0]
                second = next((value for value in values if not np.isclose(value, largest, rtol=1e-9, atol=1e-12)), np.nan)
                direction = METRIC_DIRECTIONS[metric]
                arrow = " ↑" if direction == "higher" else (" ↓" if direction == "lower" else "")
                for item in finite.itertuples(index=False):
                    cell = (MODEL_NAMES.get(item.model, item.model), METRIC_NAMES[metric] + arrow)
                    value = latex_frame.loc[cell, label]
                    if np.isclose(item.mean, largest, rtol=1e-9, atol=1e-12):
                        latex_frame.loc[cell, label] = r"\textbf{" + value + "}"
                    elif np.isfinite(second) and np.isclose(item.mean, second, rtol=1e-9, atol=1e-12):
                        latex_frame.loc[cell, label] = r"\underline{" + value + "}"
    latex_frame.index = pd.MultiIndex.from_tuples(
        [tuple(_latex_escape(part) for part in key) for key in latex_frame.index], names=["Modelo", "Métrica"]
    )
    latex_frame.columns = [_latex_escape(column) for column in latex_frame.columns]
    lines = [
        r"\begin{tabular}{ll" + "c" * len(latex_frame.columns) + "}",
        r"\toprule", "Modelo & Métrica & " + " & ".join(latex_frame.columns) + r" \\", r"\midrule",
    ]
    previous_model = None
    for (model, metric), values in latex_frame.iterrows():
        new_block = model != previous_model
        if new_block and previous_model is not None:
            lines.append(r"\midrule")
        lines.append(" & ".join([model if new_block else "", metric, *values]) + r" \\")
        previous_model = model
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    tabular = "\n".join(lines) + "\n"
    emphasis_note = (
        "Vermelho: melhor; azul: pior. " if latex_style == "colored" else
        "Maior média em negrito e segunda maior média distinta sublinhada; empates recebem o mesmo destaque. "
        "O destaque segue o valor numérico, independentemente da direção da métrica. "
    )
    document = (
        "% Required packages: \\usepackage{booktabs}, \\usepackage{graphicx}"
        + (", \\usepackage{xcolor}" if latex_style == "colored" else "") + "\n"
        "\\begin{table*}[t]\n\\centering\n\\scriptsize\n"
        "\\caption{Resultados comparados com DensGNN (adaptive). " + emphasis_note
        + "Médias parciais indicam n=k/N; falhas e OOT são excluídos. "
        + ("NMI, ARI, ACC e F1 em porcentagem. " if report.percent else "")
        + (_latex_escape(report.selection_note) + " " if report.selection_note else "")
        + "Destaques comparam médias disponíveis, sem teste de significância.}\n"
        f"\\label{{{table_label}}}\n" + "\\resizebox{\\textwidth}{!}{%\n"
        + tabular + "}%\n\\end{table*}\n"
    )
    (export_dir / "comparison_vs_densgnn.tex").write_text(document, encoding="utf-8")
