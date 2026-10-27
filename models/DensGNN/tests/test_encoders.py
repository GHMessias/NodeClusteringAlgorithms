"""Encoder routing, default-GCN compatibility, gradients and short training."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GAE, GATConv, GCNConv, SAGEConv

MODEL_ROOT = Path(__file__).resolve().parents[1]
ROOT = MODEL_ROOT.parents[1]
sys.path.insert(0, str(MODEL_ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from graph_benchmark.clustering.densgnn import run_densgnn
from graph_benchmark.models.encoders import build_densgnn_encoder
from graph_benchmark.models.gae import TwoLayerGCNEncoder
from graph_benchmark.utils.seed import set_random_seed

SPEC = importlib.util.spec_from_file_location("densgnn_encoder_benchmark", ROOT / "scripts/main.py")
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def small_graph() -> Data:
    generator = torch.Generator().manual_seed(42)
    nodes = torch.arange(16)
    edges = torch.stack([nodes, nodes.roll(1)])
    return Data(
        x=torch.randn(16, 5, generator=generator),
        edge_index=torch.cat([edges, edges.flip(0)], dim=1),
    )


class DensGNNEncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.previous_threads)

    def test_encoder_layers_and_reconstruction_gradients(self) -> None:
        data = small_graph()
        for name, layer in (("gcn", GCNConv), ("gat", GATConv), ("sage", SAGEConv)):
            with self.subTest(encoder=name):
                encoder = build_densgnn_encoder(name, 5, 7, 3, dropout=0.2)
                self.assertIsInstance(encoder.conv1, layer)
                self.assertIsInstance(encoder.conv2, layer)
                model = GAE(encoder)
                embeddings = model.encode(data.x, data.edge_index)
                self.assertEqual(tuple(embeddings.shape), (16, 3))
                loss = model.recon_loss(embeddings, data.edge_index)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                for parameter in model.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_gcn_keeps_original_initialization_and_forward(self) -> None:
        torch.manual_seed(91)
        original = TwoLayerGCNEncoder(5, 7, 3, dropout=0.2)
        torch.manual_seed(91)
        selected = build_densgnn_encoder("gcn", 5, 7, 3, dropout=0.2)
        self.assertIsInstance(selected, TwoLayerGCNEncoder)
        for key, value in original.state_dict().items():
            torch.testing.assert_close(value, selected.state_dict()[key], rtol=0, atol=0)
        data = small_graph()
        torch.manual_seed(19)
        expected = original(data.x, data.edge_index)
        torch.manual_seed(19)
        torch.testing.assert_close(selected(data.x, data.edge_index), expected, rtol=0, atol=0)

    def test_adapter_trains_all_encoders_and_keeps_gcn_default(self) -> None:
        data = small_graph()
        bundle = {"data": data, "dataset": SimpleNamespace(num_features=5)}
        params = {
            "encoder_hidden_channels": 7, "embedding_channels": 3,
            "epochs": 2, "warmup_epochs": 1, "update_p_interval": 1,
            "point_selection": "adaptive", "hdbscan_min_cluster_size": 2,
            "hdbscan_min_samples": 2, "mrd_k": 2, "device": "cpu",
        }
        outputs = {}
        for encoder in (None, "gcn", "gat", "sage"):
            with self.subTest(encoder=encoder):
                set_random_seed(42)
                selected = params if encoder is None else {**params, "encoder_type": encoder}
                output = run_densgnn(bundle, {"params": selected})
                self.assertEqual(output["metadata"]["encoder_type"], encoder or "gcn")
                self.assertEqual(output["features"].shape, (16, 3))
                self.assertEqual(output["assignments"].shape, (16,))
                self.assertTrue(np.isfinite(output["features"]).all())
                self.assertEqual(len(output["metadata"]["loss_history"]), 2)
                outputs[encoder] = output
        np.testing.assert_array_equal(outputs[None]["features"], outputs["gcn"]["features"])
        np.testing.assert_array_equal(outputs[None]["assignments"], outputs["gcn"]["assignments"])

    def test_benchmark_routes_encoder_and_rejects_invalid_values(self) -> None:
        context = benchmark.RunContext("tiny", ROOT / "tiny.pt", "cpu", 42, ROOT / "unused")
        for encoder in ("gcn", "gat", "sage"):
            commands = benchmark.build_densgnn(context, {"encoder_type": encoder})
            self.assertIn(f"algorithm.params.encoder_type={encoder}", commands[0].argv)
            self.assertIn("algorithm.name=densgnn", commands[0].argv)
        for encoder in ("typo", None):
            with self.subTest(encoder=encoder):
                with self.assertRaisesRegex(ValueError, "encoder_type"):
                    build_densgnn_encoder(encoder, 5, 7, 3)
                with self.assertRaisesRegex(benchmark.ConfigError, "encoder_type"):
                    benchmark.build_densgnn(context, {"encoder_type": encoder})


if __name__ == "__main__":
    unittest.main()
