#!/usr/bin/env python3
"""Backfill exact DBCV in existing final-result CSVs without running models."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np

from evaluate import evaluate_clustering, load_pred_labels


REPO_ROOT = Path(__file__).resolve().parents[1]


def atomic_write(path: Path, write) -> None:
    """Replace a complete file on the same filesystem."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", newline="", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def evaluate_saved_run(model_dir: Path) -> dict:
    paths = [model_dir / "labels.npy", model_dir / "embeddings.npy"]
    signature = {
        "files": {
            path.name: {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for path in paths
        },
        "hdbscan_version": version("hdbscan"),
        "evaluator_sha256": hashlib.sha256(
            Path(__file__).with_name("evaluate.py").read_bytes()
        ).hexdigest(),
    }
    record_path = model_dir / "dbcv_evaluation.json"
    if record_path.is_file():
        previous = json.loads(record_path.read_text(encoding="utf-8"))
        if previous.get("signature") == signature:
            return previous

    started = time.perf_counter()
    labels = load_pred_labels(paths[0])
    embeddings = np.load(paths[1], mmap_mode="r", allow_pickle=False)
    # These artifacts were node-count validated by the original benchmark.
    value = evaluate_clustering(
        {"num_nodes": len(embeddings)}, labels, ["dbcv"], embeddings=embeddings
    )["dbcv"]
    reason = None
    if not math.isfinite(value):
        _, counts = np.unique(labels[labels != -1], return_counts=True)
        if len(counts) < 2:
            reason = "fewer_than_two_non_noise_clusters"
        elif np.any(counts < 2):
            reason = "singleton_cluster"
        elif not np.isfinite(embeddings).all():
            reason = "non_finite_embeddings"
        else:
            reason = "undefined_or_non_finite_density_score"
    record = {
        "metric": "dbcv",
        "space": "saved_embeddings",
        "distance": "euclidean",
        "core_distance_computation": "scaled_generalized_mean",
        "noise_label": -1,
        "subsampling": False,
        "signature": signature,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_seconds": time.perf_counter() - started,
        "status": "success" if reason is None else "undefined",
        "reason": reason,
        "dbcv": value if reason is None else None,
    }
    atomic_write(record_path, lambda stream: json.dump(record, stream, indent=2))
    return record


def update_csv(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    if not {"model", "status"}.issubset(columns):
        raise ValueError(f"Not a benchmark CSV: {csv_path}")

    records = []
    for row in rows:
        identifier = {"csv": str(csv_path), "model": row["model"]}
        if row["status"] != "success":
            row["dbcv"] = "NaN"
            record = {"status": "skipped_failed_run", "reason": row.get("error")}
        else:
            try:
                record = evaluate_saved_run(csv_path.parent / row["model"])
            except Exception as exc:
                record = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
            row["dbcv"] = record.get("dbcv") if record.get("dbcv") is not None else "NaN"
        records.append({**identifier, **record})
        print(f"  {row['model']}: {row['dbcv']} ({record['status']})", flush=True)

    if "dbcv" not in columns:
        columns.append("dbcv")
    backup = csv_path.with_name(csv_path.name + ".before_dbcv.bak")
    if not backup.exists():
        shutil.copy2(csv_path, backup)

    def write_csv(stream):
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    atomic_write(csv_path, write_csv)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "results")
    args = parser.parse_args()
    root = args.results_root.resolve()
    paths = sorted(root.glob("*_final/runs/*/metrics.csv"))
    if not paths:
        parser.error(f"No *_final run metrics.csv files below {root}")
    report_path = root / "dbcv_update_report.json"
    records = []
    for index, csv_path in enumerate(paths, 1):
        print(f"[{index}/{len(paths)}] {csv_path.relative_to(root)}", flush=True)
        records.extend(update_csv(csv_path))
        report = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "csv_files_updated": index,
            "csv_files_total": len(paths),
            "counts": dict(Counter(record["status"] for record in records)),
            "records": records,
        }
        atomic_write(report_path, lambda stream: json.dump(report, stream, indent=2))
    print(json.dumps(report["counts"], sort_keys=True), flush=True)
    print(f"Report: {report_path}", flush=True)
    return int(any(record["status"] == "failed" for record in records))


if __name__ == "__main__":
    raise SystemExit(main())
