from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import torch
from torch_geometric.data import Data

from graph_benchmark.experiments.node_feature_clustering import run_node_feature_clustering
from graph_benchmark.registry import ALGORITHM_REGISTRY, DATASET_REGISTRY


def check_label_format(tmp_path, label_format):
    assignments = np.array([0, 0, 0, 1, 1, 1])
    features = np.array([[0.0], [0.1], [0.2], [5.0], [5.1], [5.2]])
    if label_format == "multilabel":
        labels = torch.tensor([[1, 0], [1, 1], [1, 0], [0, 1], [1, 1], [0, 1]])
    elif label_format == "single_label":
        labels = torch.tensor(assignments)
    else:
        labels = None
    data = Data(
        x=torch.tensor(features),
        edge_index=torch.tensor([[0, 1, 1, 2, 3, 4, 4, 5], [1, 0, 2, 1, 4, 3, 5, 4]]),
        y=labels,
    )
    bundle = {"data": data, "metadata": {"name": "label_format_test"}}
    output = {"assignments": assignments, "features": features, "metadata": {}}
    config = {
        "dataset": {"loader": "test"},
        "algorithm": {"name": "test"},
        "output": {"dir": str(tmp_path)},
    }

    with (
        patch.object(DATASET_REGISTRY, "get", return_value=lambda config: bundle),
        patch.object(ALGORITHM_REGISTRY, "get", return_value=lambda bundle, config: output),
        warnings.catch_warnings(record=True) as captured,
    ):
        warnings.simplefilter("always")
        result = run_node_feature_clustering(config)
    if label_format == "multilabel":
        assert any("Skipping external clustering metrics" in str(w.message) for w in captured)
    else:
        assert not captured

    np.testing.assert_array_equal(np.load(tmp_path / "assignments.npy"), assignments)
    np.testing.assert_array_equal(np.load(tmp_path / "features.npy"), features)
    saved_rows = np.loadtxt(tmp_path / "assignments.csv", delimiter=",", skiprows=1)
    assert saved_rows.shape == (6, 3)
    np.testing.assert_array_equal(saved_rows[:, 2], assignments)
    assert (tmp_path / "metrics.json").is_file()
    assert np.isfinite(result["metrics"]["modularity"])
    assert result["metrics"]["num_clusters_found"] == 2
    if label_format == "single_label":
        np.testing.assert_allclose(result["metrics"]["nmi"], 1.0)
        np.testing.assert_allclose(result["metrics"]["ari"], 1.0)
        np.testing.assert_array_equal(saved_rows[:, 1], assignments)
    else:
        assert result["metrics"]["nmi"] is None
        assert result["metrics"]["ari"] is None
        np.testing.assert_array_equal(saved_rows[:, 1], np.full(6, -1))
    assert data.y is labels


class ExperimentLabelFormatTests(unittest.TestCase):
    def run_label_format(self, label_format):
        with tempfile.TemporaryDirectory() as directory:
            check_label_format(Path(directory), label_format)

    def test_multilabel(self):
        self.run_label_format("multilabel")

    def test_single_label(self):
        self.run_label_format("single_label")

    def test_missing_labels(self):
        self.run_label_format("missing")


if __name__ == "__main__":
    unittest.main()
