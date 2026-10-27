"""Noise semantics, run weighting, exact LaTeX cohort, and label provenance."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import cluster_size_violin as violin


class ClusterSizeViolinTests(unittest.TestCase):
    def test_noise_is_not_a_cluster_and_remains_in_denominator(self):
        ids, counts, nodes, noise = violin.cluster_sizes(np.array([0, 0, 2, 2, 2, -1]))
        np.testing.assert_array_equal(ids, [0, 2])
        np.testing.assert_array_equal(counts, [2, 3])
        self.assertEqual((nodes, noise), (6, 1))
        self.assertAlmostEqual((100 * counts / nodes).sum(), 100 * 5 / 6)
        ids, counts, nodes, noise = violin.cluster_sizes(np.array([-1, -1]))
        self.assertEqual((len(counts), nodes, noise), (0, 2, 2))

    def test_runs_with_many_clusters_do_not_dominate(self):
        frame = pd.DataFrame({"run_id": ["a"] * 2 + ["b"] * 8, "cluster_percent": [1.] * 10})
        weights = violin.equal_run_weights(frame)
        self.assertAlmostEqual(weights[:2].sum(), .5)
        self.assertAlmostEqual(weights[2:].sum(), .5)
        self.assertAlmostEqual(weights.sum(), 1)

    def test_rejects_invalid_labels(self):
        for labels in [[], [[0, 1]], [0, np.nan], [0, .5], [-2, 1]]:
            with self.assertRaises(ValueError):
                violin.cluster_sizes(np.asarray(labels))

    def fixture(self, root):
        folder = root / "results" / "table"
        folder.mkdir(parents=True)
        latex = folder / "comparison.tex"
        latex.write_text("\\begin{tabular}{llc}\nModelo & Métrica & Cora \\\\\n"
                         "UCoDe & NMI & 1 \\\\\nDensGNN (adaptive) & NMI & 1 \\\\\n\\end{tabular}\n")
        attempts, audit = [], []
        for model in ("ucode", "densgnn", "ns4gc"):
            run = root / "results/cora_benchmark_adaptive/runs" / model
            model_dir = run / model
            model_dir.mkdir(parents=True)
            np.save(model_dir / "labels.npy", [0, 0, 1, 1])
            metadata = {"seed": 1, "embeddings_shape": [4, 2],
                        "model_params": {"point_selection": "adaptive"} if model == "densgnn" else {}}
            (model_dir / "metadata.yaml").write_text(yaml.safe_dump(metadata))
            attempts.append({"dataset": "cora_benchmark_adaptive", "model": model, "source_model": model,
                             "seed": 1, "run_id": model, "status": "success", "error": "",
                             "metrics_file": str(run / "metrics.csv")})
            audit.append({"dataset": "cora_benchmark_adaptive", "model": model, "expected_runs": 2})
        pd.DataFrame(attempts).to_csv(folder / "selected_attempts.csv", index=False)
        pd.DataFrame(audit).to_csv(folder / "run_audit.csv", index=False)
        return latex

    def test_exact_models_in_latex_and_partial_run_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = violin.load_cluster_distributions(self.fixture(root), root)
            self.assertEqual(data.model_order, ["ucode", "densgnn"])
            self.assertNotIn("ns4gc", data.clusters.model.tolist())
            self.assertEqual(len(data.runs), 2)
            self.assertTrue(data.coverage.plotted_runs.eq(1).all())
            self.assertTrue(data.coverage.expected_runs.eq(2).all())

    def test_mismatched_adaptive_variant_is_not_plotted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            latex = self.fixture(root)
            metadata = root / "results/cora_benchmark_adaptive/runs/densgnn/densgnn/metadata.yaml"
            metadata.write_text(yaml.safe_dump({"seed": 1, "model_params": {"point_selection": "core"}}))
            data = violin.load_cluster_distributions(latex, root)
            self.assertNotIn("densgnn", data.clusters.model.tolist())
            self.assertEqual(data.runs.loc[data.runs.model.eq("densgnn"), "status"].iloc[0], "unavailable")


if __name__ == "__main__":
    unittest.main()
