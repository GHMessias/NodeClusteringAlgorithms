"""Optional, supervised result selection shared by benchmark subprocesses.

The runner supplies a label vector and a private artifact directory through the
environment. Baseline entry points import this module only in selection mode.
No model weights, objectives, or optimizer state are changed by the selector.
"""

from __future__ import annotations

from contextlib import contextmanager
import csv
import json
import os
from pathlib import Path
import random

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score


def as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class BestNMISelection:
    def __init__(self, labels, directory, seed=42):
        self.labels = as_numpy(labels).reshape(-1)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        summary = self.directory / "selection.json"
        self.best = json.loads(summary.read_text(encoding="utf-8")) if summary.is_file() else None

    @contextmanager
    def inference(self, model=None):
        """Keep extra diagnostics from consuming the training RNG or mode."""
        import torch

        numpy_state = np.random.get_state()
        python_state = random.getstate()
        modes = [(module, module.training) for module in model.modules()] if model is not None else []
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
        try:
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                if model is not None:
                    model.eval()
                yield
        finally:
            for module, training in modes:
                module.training = training
            np.random.set_state(numpy_state)
            random.setstate(python_state)

    def consider(self, labels, embeddings, epoch, *, stage="train"):
        labels = as_numpy(labels).reshape(-1)
        embeddings = as_numpy(embeddings)
        if labels.size != self.labels.size:
            raise ValueError("NMI selection requires one predicted label per graph node")
        if embeddings.ndim != 2 or embeddings.shape[0] != labels.size:
            raise ValueError("NMI selection requires matching labels and embedding rows")
        if not np.isfinite(embeddings).all():
            raise ValueError("NMI selection received non-finite embeddings")
        score = float(normalized_mutual_info_score(self.labels, labels))
        record = {"epoch": epoch, "stage": stage, "nmi": score}
        history = self.directory / "history.csv"
        exists = history.is_file()
        with history.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(record))
            if not exists:
                writer.writeheader()
            writer.writerow(record)
        if self.best is None or score > self.best["nmi"]:
            self.best = {"criterion": "best_nmi", "supervised": True,
                         "tie_break": "first_observed", **record}
            np.save(self.directory / "labels.npy", labels)
            np.save(self.directory / "embeddings.npy", embeddings)
            (self.directory / "selection.json").write_text(
                json.dumps(self.best, indent=2),
                encoding="utf-8",
            )
        return score

    def kmeans(self, embeddings, n_clusters, epoch, *, stage="train", **params):
        embeddings = as_numpy(embeddings)
        estimator = KMeans(n_clusters=n_clusters, random_state=self.seed, **params)
        with self.inference():
            labels = estimator.fit_predict(embeddings)
        return self.consider(labels, embeddings, epoch, stage=stage)


selection = None
if os.environ.get("BENCHMARK_SELECT_BEST_NMI") == "1":
    selection = BestNMISelection(
        np.load(os.environ["BENCHMARK_NMI_LABELS"], allow_pickle=False),
        os.environ["BENCHMARK_NMI_DIR"],
        int(os.environ["BENCHMARK_NMI_SEED"]),
    )
