"""Encoder compatibility, gradients, YAML routing and short training checks."""

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

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "DensGNN" / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from clustering import run_deep_densgnn  # noqa: E402
from encoders import build_encoder  # noqa: E402
from graph_benchmark.models.gae import TwoLayerGCNEncoder  # noqa: E402
from graph_benchmark.utils.seed import set_random_seed  # noqa: E402

# Avoid colliding with the model's own main.py when loading the outer runner.
SPEC = importlib.util.spec_from_file_location("benchmark_main", ROOT / "scripts" / "main.py")
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def small_graph() -> Data:
    generator = torch.Generator().manual_seed(42)
    x = torch.randn(16, 5, generator=generator)
    nodes = torch.arange(16)
    edges = torch.stack([nodes, nodes.roll(1)])
    return Data(x=x, edge_index=torch.cat([edges, edges.flip(0)], dim=1))


class EncoderTests(unittest.TestCase):
    def test_reconstruction_gradients_for_each_encoder(self) -> None:
        data = small_graph()
        for name, layer_type in (("gcn", GCNConv), ("gat", GATConv), ("sage", SAGEConv)):
            with self.subTest(encoder=name):
                encoder = build_encoder(name, 5, 7, 3, dropout=0.2)
                self.assertIsInstance(encoder.conv1, layer_type)
                self.assertIsInstance(encoder.conv2, layer_type)
                model = GAE(encoder)
                z = model.encode(data.x, data.edge_index)
                self.assertEqual(tuple(z.shape), (16, 3))
                loss = model.recon_loss(z, data.edge_index)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                for param in model.parameters():
                    self.assertIsNotNone(param.grad)
                    self.assertTrue(torch.isfinite(param.grad).all())

    def test_gcn_preserves_original_initialization_and_output(self) -> None:
        torch.manual_seed(91)
        original = TwoLayerGCNEncoder(5, 7, 3, dropout=0.2)
        torch.manual_seed(91)
        selected = build_encoder("gcn", 5, 7, 3, dropout=0.2)
        self.assertIsInstance(selected, TwoLayerGCNEncoder)
        for key, value in original.state_dict().items():
            torch.testing.assert_close(value, selected.state_dict()[key], rtol=0, atol=0)
        data = small_graph()
        torch.manual_seed(19)
        expected = original(data.x, data.edge_index)
        torch.manual_seed(19)
        torch.testing.assert_close(selected(data.x, data.edge_index), expected, rtol=0, atol=0)

    def test_runner_passes_encoder_from_yaml(self) -> None:
        config = benchmark.load_config(ROOT / "configs" / "config_newman_citeseer.yaml")
        context = benchmark.RunContext("tiny", ROOT / "tiny.pt", "cpu", 42, ROOT / "unused")
        for name in ("gcn", "gat", "sage"):
            with self.subTest(encoder=name):
                params = {**config["models"]["deep_densgnn"]["params"], "encoder_type": name}
                commands = benchmark.build_deep_densgnn(context, params)
                self.assertIn(f"algorithm.params.encoder_type={name}", commands[0].argv)

    def test_invalid_encoder_is_rejected(self) -> None:
        context = benchmark.RunContext("tiny", ROOT / "tiny.pt", "cpu", 42, ROOT / "unused")
        for value in ("typo", None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "encoder_type"):
                    build_encoder(value, 5, 7, 3)
                with self.assertRaisesRegex(benchmark.ConfigError, "encoder_type"):
                    benchmark.build_deep_densgnn(context, {"encoder_type": value})

    def test_adapter_training_and_default_compatibility(self) -> None:
        data = small_graph()
        bundle = {"data": data, "dataset": SimpleNamespace(num_features=5)}
        params = {
            "encoder_hidden_channels": 7, "embedding_channels": 3,
            "epochs": 2, "warmup_epochs": 1, "update_p_interval": 1,
            "hdbscan_min_cluster_size": 2, "hdbscan_min_samples": 2,
            "candidate_k": [1, 2, 3], "device": "cpu",
        }
        outputs = {}
        for name in (None, "gcn", "gat", "sage"):
            with self.subTest(encoder=name):
                set_random_seed(42)
                selected = params if name is None else {**params, "encoder_type": name}
                output = run_deep_densgnn(bundle, {"params": selected})
                self.assertEqual(output["metadata"]["encoder_type"], name or "gcn")
                self.assertEqual(output["features"].shape, (16, 3))
                self.assertEqual(output["assignments"].shape, (16,))
                self.assertTrue(np.isfinite(output["features"]).all())
                self.assertEqual(len(output["metadata"]["loss_history"]), 2)
                outputs[name] = output
        np.testing.assert_array_equal(outputs[None]["features"], outputs["gcn"]["features"])
        np.testing.assert_array_equal(outputs[None]["assignments"], outputs["gcn"]["assignments"])


if __name__ == "__main__":
    unittest.main()
