import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

from evaluate_outputs import evaluate_file, infer_model_name, load_true_labels, select_latest_per_model


COMM_COLLAPSE_RE = re.compile(r"^(?P<base>.+)_comm_colapse_(?P<value>\d+)$")
METRICS = ("acc", "nmi", "ari", "f1_macro", "f1_micro")
LONG_FIELDS = [
    "dataset",
    "dataset_base",
    "scenario",
    "collapse_level",
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
    match = COMM_COLLAPSE_RE.match(name)
    if match is None:
        return None

    collapse_level = int(match.group("value"))
    return {
        "dataset": name,
        "dataset_base": match.group("base"),
        "scenario": "comm_collapse",
        "collapse_level": collapse_level,
        "scenario_label": f"collapse={collapse_level}",
    }


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
    return os.path.join(datasets_root, entry["dataset"] + ".pt")


def base_row(entry, model, y_true):
    return {
        "dataset": entry["dataset"],
        "dataset_base": entry["dataset_base"],
        "scenario": entry["scenario"],
        "collapse_level": entry["collapse_level"],
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

    rows.sort(key=lambda row: (row["dataset_base"], row["collapse_level"], row["model"]))
    return rows


def save_csv(rows, output_csv):
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=LONG_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    return sum(values) / len(values) if values else ""


def save_model_summary(rows, summary_dir):
    os.makedirs(summary_dir, exist_ok=True)
    path = os.path.join(summary_dir, "model_means.csv")
    grouped = defaultdict(lambda: {metric: [] for metric in METRICS})

    for row in rows:
        if row["status"] != "ok":
            continue
        key = row["model"]
        for metric in METRICS:
            grouped[key][metric].append(row[metric])

    with open(path, "w", newline="") as csvfile:
        fieldnames = ["model", "runs_ok", *[f"{metric}_mean" for metric in METRICS]]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for model in sorted(grouped):
            metric_values = grouped[model]
            row = {
                "model": model,
                "runs_ok": len(metric_values["acc"]),
            }
            for metric in METRICS:
                row[f"{metric}_mean"] = mean(metric_values[metric])
            writer.writerow(row)

    return path


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
    models = sorted({row["model"] for row in rows if row["status"] == "ok"})
    grouped = defaultdict(dict)

    for row in rows:
        if row["status"] != "ok":
            continue
        key = (row["dataset_base"], row["collapse_level"], row["scenario_label"])
        grouped[key][row["model"]] = row

    sorted_keys = sorted(grouped, key=lambda item: (item[0], item[1]))

    for metric in METRICS:
        path = os.path.join(summary_dir, f"comm_collapse_{metric}.csv")
        with open(path, "w", newline="") as csvfile:
            fieldnames = [
                "dataset_base",
                "scenario",
                "collapse_level",
                "scenario_label",
                "n_models_ok",
                "best_model",
                "best_score",
                *models,
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for dataset_base, collapse_level, scenario_label in sorted_keys:
                model_scores = grouped[(dataset_base, collapse_level, scenario_label)]
                best_model, best_score = best_model_for_metric(model_scores, metric)
                row = {
                    "dataset_base": dataset_base,
                    "scenario": "comm_collapse",
                    "collapse_level": collapse_level,
                    "scenario_label": scenario_label,
                    "n_models_ok": len(model_scores),
                    "best_model": best_model,
                    "best_score": best_score,
                }
                for model in models:
                    row[model] = model_scores.get(model, {}).get(metric, "")
                writer.writerow(row)

        generated.append(path)

    return generated


def parse_args():
    parser = argparse.ArgumentParser(description="Export comm_collapse benchmark results to CSV")
    parser.add_argument(
        "--outputs_root",
        default="outputs/comm_collapse",
        help="Root directory containing comm_collapse output subdirectories",
    )
    parser.add_argument(
        "--datasets_root",
        default="datasets/custom/comm_collapse",
        help="Directory containing the corresponding .pt files",
    )
    parser.add_argument(
        "--output_csv",
        default="outputs/comm_collapse_results.csv",
        help="Long-format CSV output path",
    )
    parser.add_argument(
        "--summary_dir",
        default="outputs/comm_collapse_summaries",
        help="Directory where pivoted CSV summaries will be written",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows = build_rows(args.outputs_root, args.datasets_root)

    save_csv(rows, args.output_csv)
    pivot_paths = save_metric_pivots(rows, args.summary_dir)
    model_summary_path = save_model_summary(rows, args.summary_dir)

    print(f"Long CSV saved to {args.output_csv} with {len(rows)} rows.")
    print(f"Model summary saved to {model_summary_path}.")
    if pivot_paths:
        print("Pivot summaries:")
        for path in pivot_paths:
            print(f"  - {path}")


if __name__ == "__main__":
    main()
