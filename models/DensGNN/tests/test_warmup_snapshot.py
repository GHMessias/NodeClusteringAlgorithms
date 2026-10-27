from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np
import torch
from torch_geometric.data import Data

from graph_benchmark.models.densgnn import DensGNNConfig, train_densgnn
from graph_benchmark.utils.seed import set_random_seed


class WarmupSnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)
        nodes = torch.arange(12)
        destinations = (nodes + 1) % 12
        self.data = Data(
            x=torch.arange(48, dtype=torch.float32).reshape(12, 4) / 48,
            edge_index=torch.stack((torch.cat((nodes, destinations)), torch.cat((destinations, nodes)))),
        )
        self.config = DensGNNConfig(
            input_channels=4, hidden_channels=8, embedding_channels=3,
            epochs=4, warmup_epochs=2, dropout=0.35, device="cpu",
            hdbscan_min_cluster_size=2, hdbscan_min_samples=2, mrd_k=2,
        )

    def train(self, config, callback=None):
        set_random_seed(73)
        return train_densgnn(self.data, None, config, warmup_callback=callback)

    def test_snapshot_matches_encoder_after_exact_warmup_steps(self) -> None:
        snapshots = []
        self.train(self.config, lambda array, epoch: snapshots.append((array, epoch)))
        reference = self.train(replace(self.config, epochs=2))
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0][1], 2)
        np.testing.assert_array_equal(snapshots[0][0], reference["embeddings"])

    def test_snapshot_preserves_training_with_dropout_and_negative_sampling(self) -> None:
        reference = self.train(self.config)
        captured = self.train(self.config, lambda array, epoch: None)
        np.testing.assert_array_equal(captured["embeddings"], reference["embeddings"])
        np.testing.assert_array_equal(captured["assignments"], reference["assignments"])
        self.assertEqual(captured["loss_history"], reference["loss_history"])

    def test_zero_warmup_and_warmup_at_final_epoch(self) -> None:
        for warmup in (0, self.config.epochs):
            with self.subTest(warmup=warmup):
                snapshots = []
                result = self.train(
                    replace(self.config, warmup_epochs=warmup),
                    lambda array, epoch: snapshots.append((array, epoch)),
                )
                self.assertEqual(len(snapshots), 1)
                self.assertEqual(snapshots[0][1], warmup)
                self.assertEqual(snapshots[0][0].shape, (12, 3))
                if warmup == self.config.epochs:
                    np.testing.assert_array_equal(snapshots[0][0], result["embeddings"])

    def test_snapshot_boundary_must_be_reachable(self) -> None:
        with self.assertRaisesRegex(ValueError, "warmup_epochs"):
            self.train(replace(self.config, warmup_epochs=5), lambda array, epoch: None)


if __name__ == "__main__":
    unittest.main()
