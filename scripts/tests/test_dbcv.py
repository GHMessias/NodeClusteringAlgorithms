"""DBCV correctness and benchmark/CLI integration (no model training required)."""

import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml
from hdbscan.validity import validity_index
from sklearn.datasets import make_blobs

from scripts.evaluate import evaluate_clustering


REPO_ROOT = Path(__file__).resolve().parents[2]


class DBCVTests(unittest.TestCase):
    def setUp(self):
        self.features, self.labels = make_blobs(
            n_samples=40, centers=[[-5, 0], [5, 0]], cluster_std=0.3, random_state=42
        )

    def score(self, labels=None, features=None):
        labels = self.labels if labels is None else labels
        features = self.features if features is None else features
        return evaluate_clustering(
            {"num_nodes": len(labels)}, labels, ["dbcv"], embeddings=features
        )["dbcv"]

    def test_matches_reference_and_prefers_separated_partition(self):
        score = self.score()
        self.assertAlmostEqual(score, validity_index(self.features, self.labels))
        self.assertGreater(score, 0.9)
        self.assertLess(self.score(labels=np.arange(40) % 2), 0)

    def test_float32_embeddings_and_noncontiguous_cluster_ids(self):
        features = torch.tensor(self.features, dtype=torch.float32)
        labels = np.where(self.labels == 0, 14, 100000)
        self.assertAlmostEqual(self.score(labels, features), self.score(), places=6)
        np.testing.assert_array_equal(labels, np.where(self.labels == 0, 14, 100000))

    def test_noise_is_not_a_cluster_and_penalizes_total_score(self):
        labels = np.r_[self.labels, -1, -1]
        features = np.vstack([self.features, [[0, 10], [0, -10]]])
        self.assertAlmostEqual(self.score(labels, features), self.score() * 40 / 42)

    def test_high_dimensional_scores_are_finite_and_scale_invariant(self):
        features = np.tile(self.features, (1, 2048))
        score = self.score(features=features)
        self.assertTrue(math.isfinite(score))
        self.assertGreater(score, 0.9)
        for scale in (1e-6, 1e6):
            with self.subTest(scale=scale):
                self.assertAlmostEqual(self.score(features=features * scale), score, places=8)

    def test_undefined_partitions_return_nan(self):
        for labels in (
            np.zeros(40, dtype=int),
            np.full(40, -1),
            np.r_[np.zeros(20, dtype=int), np.full(20, -1)],
            np.arange(40),
            np.r_[np.zeros(39, dtype=int), 1],
        ):
            with self.subTest(labels=labels):
                self.assertTrue(math.isnan(self.score(labels=labels)))
        self.assertTrue(math.isnan(self.score(features=np.zeros_like(self.features))))
        self.assertTrue(math.isnan(self.score(features=np.full_like(self.features, np.nan))))

    def test_embeddings_are_required_and_validated(self):
        for features, message in (
            (None, "Model embeddings are required"),
            (self.features[:-1], "Embedding row count"),
            (self.features[:, 0], "two-dimensional"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                evaluate_clustering(
                    {"num_nodes": 40}, self.labels, ["dbcv"], embeddings=features
                )

    def test_unselected_dbcv_does_not_load_dependency(self):
        with patch.dict(sys.modules, {"hdbscan.validity": None}):
            self.assertEqual(
                evaluate_clustering({"num_nodes": 40}, self.labels, ["n_clusters"]),
                {"n_clusters": 2},
            )
            with self.assertRaisesRegex(ImportError, "DBCV requires hdbscan"):
                self.score()

    def test_runner_csv_and_manual_evaluator(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_path = root / "data.pt"
            # The runner's existing default policy reads K from data.y.
            torch.save(
                {"x": torch.tensor(self.features, dtype=torch.float32),
                 "y": torch.tensor(self.labels),
                 "edge_index": torch.empty((2, 0), dtype=torch.long)},
                data_path,
            )
            config = {
                "experiment": {"dataset": "dbcv_smoke", "data_pt": str(data_path),
                               "output_root": str(root / "results"), "device": "cpu",
                               "seed": 42, "runs": 1, "continue_on_error": False},
                "output": {"save_embeddings": True, "generate_figures": False},
                "metrics": {"include": ["runtime_seconds", "dbcv"]},
                "models": {"kmeans_attributes": {"enabled": True,
                                                "params": {"n_clusters": 2, "n_init": 1}}},
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(REPO_ROOT / "scripts/main.py"),
                 "--config", str(config_path), "--model", "kmeans_attributes"],
                cwd=root, capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            csv_path, = (root / "results").rglob("metrics.csv")
            with csv_path.open(newline="", encoding="utf-8") as stream:
                row, = csv.DictReader(stream)
            self.assertEqual(row["status"], "success", row)
            self.assertEqual(set(row), {"dataset", "run_id", "model", "seed",
                                        "status", "error", "runtime_seconds", "dbcv"})
            self.assertGreater(float(row["dbcv"]), 0.9)
            model_dir = csv_path.parent / "kmeans_attributes"
            manual = subprocess.run(
                [sys.executable, str(REPO_ROOT / "scripts/evaluate.py"),
                 "--data-pt", str(data_path), "--labels", str(model_dir / "labels.npy"),
                 "--embeddings", str(model_dir / "embeddings.npy"), "--metrics", "dbcv"],
                cwd=root, capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(manual.returncode, 0, manual.stdout + manual.stderr)
            self.assertAlmostEqual(json.loads(manual.stdout)["dbcv"], float(row["dbcv"]))


if __name__ == "__main__":
    unittest.main()
