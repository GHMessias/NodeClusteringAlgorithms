import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

from evaluate_outputs import evaluate_file, infer_model_name, load_true_labels, select_latest_per_model


SCENARIO_SPECS = (
    {
        "name": "heterophily",
        "regex": re.compile(r"^(?P<base>.+)_hetero_(?P<value>\d+(?:\.\d+)?)$"),
        "datasets_subdir": "hetero_increase",
        "value_parser": float,
        "label_prefix": "hetero",
    },
    {
        "name": "edges",
        "regex": re.compile(r"^(?P<base>.+)_(?P<value>\d+)_edges$"),
        "datasets_subdir": "node_increase",
        "value_parser": int,
        "label_prefix": "edges",
    },
)

SCENARIO_ORDER = {spec["name"]: idx for idx, spec in enumerate(SCENARIO_SPECS)}
METRICS = ("acc", "nmi", "ari", "f1_macro", "f1_micro")
LONG_FIELDS = [
    "dataset",
    "dataset_base",
    "scenario",
    "scenario_value",
    "scenario_label",
    "status",
    "file",
    "n_samples",
    "n_clusters",
    "model",
    *METRICS,
    "error",
]
RUN_STATUS_FILENAME = "run_status.json"


def parse_dataset_dir_name(name):
    for spec in SCENARIO_SPECS:
        match = spec["regex"].match(name)
        if match is None:
            continue

        value = spec["value_parser"](match.group("value"))
        return {
            "dataset": name,
            "dataset_base": match.group("base"),
            "scenario": spec["name"],
            "scenario_value": value,
            "scenario_label": f"{spec['label_prefix']}={match.group('value')}",
            "datasets_subdir": spec["datasets_subdir"],
        }
    return None


def find_dataset_entries(outputs_root):
    entries = []
    for entry in sorted(os.listdir(outputs_root)):
        path = os.path.join(outputs_root, entry)
        if not os.path.isdir(path):
            continue

        info = parse_dataset_dir_name(entry)
        if info is None:
            continue

        info["output_dir"] = path
        entries.append(info)
    return entries


def label_files_for_dir(dataset_dir):
    paths = glob.glob(os.path.join(dataset_dir, "*_labels.npy"))
    return select_latest_per_model(paths)


def read_run_status(dataset_dir):
    path = os.path.join(dataset_dir, RUN_STATUS_FILENAME)
    if not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    results = payload.get("results", [])
    return {item["model"]: item for item in results if "model" in item}


def collect_expected_models(dataset_entries):
    models = set()
    for entry in dataset_entries:
        for path in label_files_for_dir(entry["output_dir"]):
            models.add(infer_model_name(path))
        models.update(read_run_status(entry["output_dir"]).keys())
    return sorted(models)


def data_pt_for_entry(entry, datasets_root):
    return os.path.join(datasets_root, entry["datasets_subdir"], entry["dataset"] + ".pt")


def base_row(entry, model, y_true):
    return {
        "dataset": entry["dataset"],
        "dataset_base": entry["dataset_base"],
        "scenario": entry["scenario"],
        "scenario_value": entry["scenario_value"],
        "scenario_label": entry["scenario_label"],
        "status": "missing",
        "file": "",
        "n_samples": len(y_true),
        "n_clusters": "",
        "model": model,
        "acc": "",
        "nmi": "",
        "ari": "",
        "f1_macro": "",
        "f1_micro": "",
        "error": "",
    }


def build_rows(outputs_root, datasets_root):
    dataset_entries = find_dataset_entries(outputs_root)
    expected_models = collect_expected_models(dataset_entries)
    rows = []

    for entry in dataset_entries:
        data_pt = data_pt_for_entry(entry, datasets_root)
        if not os.path.exists(data_pt):
            raise FileNotFoundError(f"Missing data.pt for {entry['dataset']}: {data_pt}")

        y_true = load_true_labels(data_pt)
        latest_paths = label_files_for_dir(entry["output_dir"])
        model_to_path = {infer_model_name(path): path for path in latest_paths}
        status_by_model = read_run_status(entry["output_dir"])

        for model in expected_models:
            row = base_row(entry, model, y_true)
            path = model_to_path.get(model)
            if path is None:
                status_info = status_by_model.get(model)
                if status_info is not None and not status_info.get("ok", False):
                    row["status"] = "error"
                    row["file"] = status_info.get("log_file", "")
                    row["error"] = status_info.get("error", "")
                rows.append(row)
                continue

            try:
                metrics = evaluate_file(path, y_true)
            except Exception as exc:
                row["status"] = "error"
                row["file"] = path
                row["error"] = str(exc)
                rows.append(row)
                continue

            row.update(metrics)
            row["status"] = "ok"
            row["error"] = ""
            rows.append(row)

    rows.sort(
        key=lambda row: (
            SCENARIO_ORDER.get(row["scenario"], 999),
            row["dataset_base"],
            row["scenario_value"],
            row["model"],
        )
    )
    return rows


def save_csv(rows, output_csv):
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=LONG_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def filter_rows(rows, scenario):
    return [row for row in rows if row["scenario"] == scenario]


def scenario_models(rows, scenario):
    models = {row["model"] for row in rows if row["scenario"] == scenario}
    return sorted(models)


def group_ok_rows(rows, scenario):
    grouped = defaultdict(dict)
    for row in rows:
        if row["scenario"] != scenario or row["status"] != "ok":
            continue

        key = (row["dataset_base"], row["scenario_value"], row["scenario_label"])
        grouped[key][row["model"]] = row
    return grouped


def best_model_for_metric(model_scores, metric):
    best_model = ""
    best_score = ""
    for model, row in model_scores.items():
        score = row.get(metric)
        if score == "":
            continue

        if best_score == "" or score > best_score:
            best_model = model
            best_score = score
    return best_model, best_score


def save_metric_pivots(rows, summary_dir):
    os.makedirs(summary_dir, exist_ok=True)
    generated = []

    for spec in SCENARIO_SPECS:
        scenario = spec["name"]
        models = scenario_models(rows, scenario)
        grouped = group_ok_rows(rows, scenario)
        if not grouped:
            continue

        sorted_keys = sorted(grouped, key=lambda item: (item[0], item[1]))

        for metric in METRICS:
            path = os.path.join(summary_dir, f"{scenario}_{metric}.csv")
            with open(path, "w", newline="") as csvfile:
                fieldnames = [
                    "dataset_base",
                    "scenario",
                    "scenario_value",
                    "scenario_label",
                    "n_models_ok",
                    "best_model",
                    "best_score",
                    *models,
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for dataset_base, scenario_value, scenario_label in sorted_keys:
                    model_scores = grouped[(dataset_base, scenario_value, scenario_label)]
                    best_model, best_score = best_model_for_metric(model_scores, metric)
                    row = {
                        "dataset_base": dataset_base,
                        "scenario": scenario,
                        "scenario_value": scenario_value,
                        "scenario_label": scenario_label,
                        "n_models_ok": len(model_scores),
                        "best_model": best_model,
                        "best_score": best_score,
                    }
                    for model in models:
                        value = model_scores.get(model, {}).get(metric, "")
                        row[model] = value
                    writer.writerow(row)

            generated.append(path)

    return generated


def mean(values):
    return sum(values) / len(values) if values else ""


def save_model_summary(rows, summary_dir):
    os.makedirs(summary_dir, exist_ok=True)
    path = os.path.join(summary_dir, "model_means.csv")
    grouped = defaultdict(lambda: {metric: [] for metric in METRICS})

    for row in rows:
        if row["status"] != "ok":
            continue

        key = (row["scenario"], row["model"])
        for metric in METRICS:
            grouped[key][metric].append(row[metric])

    with open(path, "w", newline="") as csvfile:
        fieldnames = ["scenario", "model", "runs_ok", *[f"{metric}_mean" for metric in METRICS]]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for scenario, model in sorted(grouped, key=lambda item: (SCENARIO_ORDER.get(item[0], 999), item[1])):
            metric_values = grouped[(scenario, model)]
            row = {
                "scenario": scenario,
                "model": model,
                "runs_ok": len(metric_values["acc"]),
            }
            for metric in METRICS:
                row[f"{metric}_mean"] = mean(metric_values[metric])
            writer.writerow(row)

    return path


def parse_args():
    parser = argparse.ArgumentParser(description="Export scatt benchmark results to CSV")
    parser.add_argument("--outputs_root", default="outputs", help="Root directory containing output subdirectories")
    parser.add_argument("--datasets_root", default="datasets/custom", help="Directory containing the corresponding .pt files")
    parser.add_argument("--output_csv", default="outputs/benchmark_results.csv", help="Long-format CSV output path")
    parser.add_argument(
        "--summary_dir",
        default="outputs/summaries",
        help="Directory where pivoted CSV summaries will be written",
    )
    parser.add_argument(
        "--heterophily_csv",
        default="outputs/heterophily_results.csv",
        help="Compatibility CSV containing only heterophily rows",
    )
    parser.add_argument(
        "--edges_csv",
        default="outputs/edge_results.csv",
        help="Convenience CSV containing only edge-sweep rows",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows = build_rows(args.outputs_root, args.datasets_root)

    save_csv(rows, args.output_csv)
    save_csv(filter_rows(rows, "heterophily"), args.heterophily_csv)
    save_csv(filter_rows(rows, "edges"), args.edges_csv)

    pivot_paths = save_metric_pivots(rows, args.summary_dir)
    model_summary_path = save_model_summary(rows, args.summary_dir)

    print(f"Long CSV saved to {args.output_csv} with {len(rows)} rows.")
    print(f"Heterophily CSV saved to {args.heterophily_csv}.")
    print(f"Edges CSV saved to {args.edges_csv}.")
    print(f"Model summary saved to {model_summary_path}.")
    if pivot_paths:
        print("Pivot summaries:")
        for path in pivot_paths:
            print(f"  - {path}")


if __name__ == "__main__":
    main()
