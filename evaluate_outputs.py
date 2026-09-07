import argparse
import csv
import glob
import os
import re

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score


TIMESTAMPED_LABELS_RE = re.compile(r"^(?P<model>.+)_\d{8}_\d{4}_labels\.npy$")


def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def unwrap_data_pt(obj):
    if isinstance(obj, (tuple, list)):
        for item in obj:
            if hasattr(item, "y") or (isinstance(item, dict) and "y" in item):
                return item
        return obj[0]
    if isinstance(obj, dict):
        if "data" in obj:
            return obj["data"]
        if "y" in obj:
            return obj
    return obj


def get_attr(data, name):
    if isinstance(data, dict):
        return data.get(name)
    return getattr(data, name, None)


def as_label_vector(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    labels = np.asarray(value)
    if labels.ndim > 1:
        labels = labels.argmax(axis=-1)
    return labels.reshape(-1).astype(np.int64)


def load_true_labels(data_pt):
    data = unwrap_data_pt(torch_load(data_pt))
    y = get_attr(data, "y")
    if y is None:
        raise ValueError(f"{data_pt} precisa conter data.y para avaliação")
    return as_label_vector(y)


def load_pred_labels(path):
    labels = np.load(path, allow_pickle=True)
    if isinstance(labels, np.ndarray) and labels.shape == () and isinstance(labels.item(), dict):
        labels = extract_labels_from_dict(labels.item(), path)
    return as_label_vector(labels)


def extract_labels_from_dict(value, path):
    for key in ("labels", "pred", "preds", "y_pred", "clusters"):
        if key in value:
            return value[key]
    raise ValueError(f"{path} contem um dicionario, mas nenhuma chave de labels conhecida")


def infer_model_name(path):
    filename = os.path.basename(path)
    match = TIMESTAMPED_LABELS_RE.match(filename)
    if match:
        return match.group("model")
    if filename.endswith("_labels.npy"):
        return filename[: -len("_labels.npy")]
    return os.path.splitext(filename)[0]


def clustering_match(y_true, y_pred):
    true_values = np.unique(y_true)
    pred_values = np.unique(y_pred)
    true_index = {label: idx for idx, label in enumerate(true_values)}
    pred_index = {label: idx for idx, label in enumerate(pred_values)}

    contingency = np.zeros((len(true_values), len(pred_values)), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        contingency[true_index[true_label], pred_index[pred_label]] += 1

    row_ind, col_ind = linear_sum_assignment(-contingency)
    mapping = {pred_values[col]: true_values[row] for row, col in zip(row_ind, col_ind)}
    fallback = true_values[0]
    y_pred_matched = np.array([mapping.get(label, fallback) for label in y_pred], dtype=np.int64)
    acc = (y_pred_matched == y_true).mean()
    return acc, y_pred_matched


def evaluate_file(path, y_true):
    y_pred = load_pred_labels(path)
    if len(y_pred) != len(y_true):
        raise ValueError(f"tamanho incompatível: y_true={len(y_true)} labels={len(y_pred)}")

    acc, y_pred_matched = clustering_match(y_true, y_pred)
    return {
        "model": infer_model_name(path),
        "file": path,
        "n_samples": len(y_true),
        "n_clusters": len(np.unique(y_pred)),
        "acc": acc,
        "nmi": normalized_mutual_info_score(y_true, y_pred),
        "ari": adjusted_rand_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred_matched, average="macro", zero_division=0),
        "f1_micro": f1_score(y_true, y_pred_matched, average="micro", zero_division=0),
    }


def find_label_files(outputs_dir, pattern):
    return sorted(glob.glob(os.path.join(outputs_dir, pattern)))


def select_latest_per_model(paths):
    latest = {}
    for path in paths:
        model = infer_model_name(path)
        current = latest.get(model)
        if current is None or os.path.getmtime(path) > os.path.getmtime(current):
            latest[model] = path
    return sorted(latest.values())


def format_float(value):
    return f"{value:.4f}"


def print_table(rows):
    if not rows:
        print("Nenhum arquivo de labels válido foi avaliado.")
        return

    headers = ["model", "acc", "nmi", "ari", "f1_macro", "f1_micro", "n_clusters", "file"]
    table = []
    for row in rows:
        table.append(
            [
                row["model"],
                format_float(row["acc"]),
                format_float(row["nmi"]),
                format_float(row["ari"]),
                format_float(row["f1_macro"]),
                format_float(row["f1_micro"]),
                str(row["n_clusters"]),
                os.path.basename(row["file"]),
            ]
        )

    widths = [
        max(len(str(value)) for value in [header, *[line[idx] for line in table]])
        for idx, header in enumerate(headers)
    ]
    print("  ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for line in table:
        print("  ".join(value.ljust(widths[idx]) for idx, value in enumerate(line)))


def save_csv(rows, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "model",
                "file",
                "n_samples",
                "n_clusters",
                "acc",
                "nmi",
                "ari",
                "f1_macro",
                "f1_micro",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV salvo em {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate clustering label outputs against data.pt labels")
    parser.add_argument("--data_pt", required=True, help="PyG data.pt path containing data.y")
    parser.add_argument("--outputs_dir", default="outputs", help="Directory containing *_labels.npy files")
    parser.add_argument("--pattern", default="*_labels.npy", help="Glob pattern inside outputs_dir")
    parser.add_argument("--labels", nargs="*", default=None, help="Specific .npy label files to evaluate")
    parser.add_argument("--latest_per_model", action="store_true", help="Evaluate only the newest labels file per model")
    parser.add_argument("--csv", default=None, help="Optional CSV output path")
    parser.add_argument("--skip_invalid", action="store_true", help="Skip files with incompatible lengths or invalid data")
    return parser.parse_args()


def main():
    args = parse_args()
    y_true = load_true_labels(args.data_pt)
    paths = args.labels if args.labels else find_label_files(args.outputs_dir, args.pattern)
    paths = [os.path.abspath(path) for path in paths]

    if args.latest_per_model:
        paths = select_latest_per_model(paths)

    rows = []
    for path in paths:
        try:
            rows.append(evaluate_file(path, y_true))
        except Exception as exc:
            if not args.skip_invalid:
                raise
            print(f"Pulando {path}: {exc}")

    rows.sort(key=lambda row: (row["model"], row["file"]))
    print_table(rows)

    if args.csv:
        save_csv(rows, args.csv)


if __name__ == "__main__":
    main()


