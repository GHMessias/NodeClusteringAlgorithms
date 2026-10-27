"""Result selection, RNG isolation, and subprocess/CSV integration."""

import csv
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "models" / "_shared"))
import main as benchmark
from benchmark_selection import BestNMISelection


class NMISelectionTests(unittest.TestCase):
    def test_best_pair_ties_resume_and_shapes(self):
        truth = np.array([0, 0, 1, 1])
        features = np.arange(8).reshape(4, 2)
        with tempfile.TemporaryDirectory() as directory:
            selection = BestNMISelection(truth, directory)
            selection.consider(truth, features, 1)
            selection.consider(1 - truth, features + 100, 2)
            selection.consider(np.zeros(4), features + 200, 3)
            resumed = BestNMISelection(truth, directory)
            resumed.consider(np.zeros(4), features + 300, 4)
            self.assertEqual(resumed.best["epoch"], 1)
            np.testing.assert_array_equal(np.load(Path(directory) / "labels.npy"), truth)
            np.testing.assert_array_equal(np.load(Path(directory) / "embeddings.npy"), features)
            with self.assertRaises(ValueError):
                resumed.consider(truth[:-1], features, 5)
            with self.assertRaises(ValueError):
                resumed.consider(truth, features[:-1], 5)

    def test_inference_restores_rng_modes_and_gradients_even_on_error(self):
        with tempfile.TemporaryDirectory() as directory:
            selection = BestNMISelection([0, 1], directory)
            model = torch.nn.Sequential(torch.nn.Dropout(), torch.nn.Linear(2, 2))
            model.train()
            model[1].eval()
            torch.manual_seed(12)
            np.random.seed(12)
            random.seed(12)
            expected = (torch.rand(2), np.random.rand(2), random.random())
            torch.manual_seed(12)
            np.random.seed(12)
            random.seed(12)
            with self.assertRaisesRegex(RuntimeError, "diagnostic"):
                with selection.inference(model):
                    self.assertFalse(model.training)
                    self.assertFalse(torch.is_grad_enabled())
                    torch.rand(20)
                    np.random.rand(20)
                    random.random()
                    raise RuntimeError("diagnostic")
            torch.testing.assert_close(torch.rand(2), expected[0])
            np.testing.assert_array_equal(np.random.rand(2), expected[1])
            self.assertEqual(random.random(), expected[2])
            self.assertTrue(model.training)
            self.assertFalse(model[1].training)
            self.assertTrue(torch.is_grad_enabled())

    def test_runner_selects_pair_before_evaluation_and_records_csv(self):
        self.run_fake_model(enabled=True, labeled=True)

    def test_runner_preserves_native_behavior_when_disabled(self):
        self.run_fake_model(enabled=False, labeled=True)

    def test_runner_falls_back_without_ground_truth(self):
        self.run_fake_model(enabled=True, labeled=False)

    def test_runner_records_failed_selection_without_training_candidates(self):
        self.run_fake_model(enabled=True, labeled=True, report_candidates=False)

    def test_selection_does_not_require_nmi_in_requested_csv_metrics(self):
        self.run_fake_model(enabled=True, labeled=True, metrics=["ari", "runtime_seconds"])

    def test_multistage_pretraining_does_not_enable_selection(self):
        self.run_fake_model(enabled=True, labeled=True, multistage=True)

    def run_fake_model(self, *, enabled, labeled, report_candidates=True, metrics=None, multistage=False):
        metrics = metrics or ["nmi", "runtime_seconds"]
        # This process emits an excellent first epoch and a poor final output.
        source = '''
import os, sys
from pathlib import Path
import numpy as np
out = Path(sys.argv[1])
features = np.arange(8).reshape(4, 2)
if os.environ.get("BENCHMARK_SELECT_BEST_NMI") == "1" and sys.argv[2] == "True":
    from benchmark_selection import selection
    selection.consider([0, 0, 1, 1], features, 1)
    selection.consider([0, 1, 0, 1], features + 50, 2)
np.save(out / "fake_labels.npy", np.zeros(4, dtype=int))
np.save(out / "embeddings.npy", features + 100)
'''
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory) / "run" / "ucode"
            context = benchmark.RunContext("test", None, "cpu", 42, model_dir)
            command = benchmark.Command(
                [sys.executable, "-c", source, str(model_dir), str(report_candidates)], ROOT
            )
            commands = [command]
            if multistage:
                pretrain = benchmark.Command([
                    sys.executable, "-c",
                    'import os; assert os.environ.get("BENCHMARK_SELECT_BEST_NMI") is None',
                ], ROOT)
                commands = [pretrain, pretrain, command]
            data = {"num_nodes": 4, "y": np.array([0, 0, 1, 1]) if labeled else None}
            with patch.dict(benchmark.MODEL_REGISTRY, {"ucode": lambda context, params: commands}):
                row = benchmark.run_model(
                    "ucode", context, {"epochs": 3}, data, metrics,
                    {"save_metadata": True}, "test", select_best_nmi=enabled,
                )
            metadata = yaml.safe_load((model_dir / "metadata.yaml").read_text())
            if enabled and labeled and not report_candidates:
                self.assertEqual(row["status"], "failed")
                self.assertIn("did not report any training candidates", row["error"])
                return
            self.assertEqual(row["status"], "success", row["error"])
            selection = metadata["result_selection"]
            if enabled and labeled:
                self.assertEqual(row.get("nmi", row.get("ari")), 1.0)
                self.assertEqual(selection["epoch"], 1)
                self.assertTrue(selection["supervised"])
                np.testing.assert_array_equal(np.load(model_dir / "embeddings.npy"), np.arange(8).reshape(4, 2))
            else:
                self.assertEqual(selection["criterion"], "native")
                np.testing.assert_array_equal(np.load(model_dir / "labels.npy"), np.zeros(4))
                self.assertFalse((model_dir / "selection").exists())
                if not labeled:
                    self.assertTrue(np.isnan(row["nmi"]))
                    self.assertIn("data.y unavailable", selection["reason"])
            csv_path = Path(directory) / "metrics.csv"
            benchmark.append_metrics_row(csv_path, metrics, row)
            with csv_path.open() as stream:
                saved = next(csv.DictReader(stream))
            if labeled and "nmi" in metrics:
                self.assertEqual(float(saved["nmi"]), row["nmi"])
            if "nmi" not in metrics:
                self.assertNotIn("nmi", saved)


if __name__ == "__main__":
    unittest.main()
