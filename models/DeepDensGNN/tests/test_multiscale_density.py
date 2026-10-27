"""Sanity checks for the experimental multi-scale MRD variant."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "DensGNN" / "src"))

from cluster_scale import ClusterScaleManager  # noqa: E402
from density import cluster_conditioned_distances, multiscale_core_distances  # noqa: E402
from model import DeepDensGNNConfig, _soft_assignments  # noqa: E402
from graph_benchmark.utils.clustering_loss import kl_clustering_loss, target_distribution  # noqa: E402
from graph_benchmark.utils.densgnn_loss import (  # noqa: E402
    ClusterSupport,
    inverse_distance_assignments,
    support_cluster_distances,
)


def test_shapes_probabilities_gradient_and_k_effective() -> None:
    embeddings = torch.tensor([[0.0], [1.0], [3.0], [8.0], [9.0], [12.0]], requires_grad=True)
    supports = [
        ClusterSupport(0, torch.tensor([0, 1])),
        ClusterSupport(1, torch.tensor([4, 5])),
    ]
    optimizer = torch.optim.Adam([embeddings], lr=0.01)
    manager = ClusterScaleManager((1, 2, 3))
    manager.refresh(np.array([0, 0, 0, 1, 1, 1]), supports, optimizer, device=embeddings.device, learning_rate=0.01)
    core, pairwise = multiscale_core_distances(embeddings, (1, 2, 3))
    pi = manager.pi()
    effective = core @ pi.T
    distances = cluster_conditioned_distances(pairwise, core, supports, pi)
    assert core.shape == (6, 3)
    assert effective.shape == distances.shape == (6, 2)
    assert torch.all(core >= 0)
    torch.testing.assert_close(pi.sum(dim=1), torch.ones(2))
    info = manager.get_density_scale_info()
    assert info["candidate_k"] == [1, 2, 3]
    assert all(1 <= value <= 3 for value in info["k_eff"])
    distances.sum().backward()
    assert all(parameter.grad is not None for parameter in manager.logits)
    assert any(torch.any(parameter.grad != 0) for parameter in manager.logits)


def test_full_density_kl_loss_reaches_scale_logits() -> None:
    embeddings = torch.tensor([[0.0], [1.0], [3.0], [8.0], [9.0], [12.0]])
    supports = [
        ClusterSupport(0, torch.tensor([0, 1])),
        ClusterSupport(1, torch.tensor([4, 5])),
    ]
    optimizer = torch.optim.Adam([torch.nn.Parameter(torch.tensor(1.0))], lr=0.01)
    manager = ClusterScaleManager((1, 2, 3))
    manager.refresh(np.array([0, 0, 0, 1, 1, 1]), supports, optimizer, device=embeddings.device, learning_rate=0.01)
    core, pairwise = multiscale_core_distances(embeddings, (1, 2, 3))
    q = inverse_distance_assignments(cluster_conditioned_distances(pairwise, core, supports, manager.pi()))
    p = target_distribution(q).detach()
    kl_clustering_loss(q, p).backward()
    assert all(parameter.grad is not None for parameter in manager.logits)
    assert any(torch.any(parameter.grad != 0) for parameter in manager.logits)


def test_one_hot_scale_matches_original_single_k_mrd() -> None:
    embeddings = torch.tensor([[0.0], [1.0], [3.0], [8.0], [9.0]], dtype=torch.float64)
    supports = [
        ClusterSupport(0, torch.tensor([0, 1])),
        ClusterSupport(1, torch.tensor([3, 4]), torch.tensor([0.25, 0.75], dtype=torch.float64)),
    ]
    core, pairwise = multiscale_core_distances(embeddings, (1, 2, 3))
    pi = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64)
    actual = cluster_conditioned_distances(pairwise, core, supports, pi)
    expected = support_cluster_distances(embeddings, supports, mrd_k=2)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_euclidean_mode_uses_same_weighted_supports_without_scales() -> None:
    embeddings = torch.tensor([[0.0], [1.0], [3.0], [8.0], [9.0]], requires_grad=True)
    supports = [
        ClusterSupport(0, torch.tensor([0, 1])),
        ClusterSupport(1, torch.tensor([3, 4]), torch.tensor([0.25, 0.75])),
    ]
    config = DeepDensGNNConfig(
        input_channels=1,
        hidden_channels=2,
        embedding_channels=1,
        mrd_k=2,
        candidate_k=(1, 2, 3),
        support_distance="euclidean",
    )
    actual = _soft_assignments(embeddings, SimpleNamespace(supports=supports), None, config)
    expected_distances = support_cluster_distances(
        embeddings, supports, mrd_k=2, support_distance="euclidean"
    )
    torch.testing.assert_close(actual, inverse_distance_assignments(expected_distances))
    actual[:, 0].sum().backward()
    assert embeddings.grad is not None and torch.isfinite(embeddings.grad).all()


def test_cluster_relabel_preserves_parameter_and_new_cluster_starts_uniform() -> None:
    optimizer = torch.optim.Adam([torch.nn.Parameter(torch.tensor(1.0))], lr=0.01)
    manager = ClusterScaleManager((1, 2, 3))
    old = np.array([0, 0, 0, 1, 1, 1, -1, -1])
    supports = [ClusterSupport(0, torch.tensor([0])), ClusterSupport(1, torch.tensor([3]))]
    manager.refresh(old, supports, optimizer, device=torch.device("cpu"), learning_rate=0.01)
    first, vanished = list(manager.logits)
    with torch.no_grad():
        first.copy_(torch.tensor([1.0, 2.0, 3.0]))
    manager.pi()[:, 0].sum().backward()
    optimizer.step()
    retained_state = optimizer.state[first]
    assert vanished in optimizer.state
    labels = np.array([8, 8, 8, -1, -1, -1, 7, 7])
    matches = manager.refresh(
        labels,
        [ClusterSupport(7, torch.tensor([6])), ClusterSupport(8, torch.tensor([0]))],
        optimizer,
        device=torch.device("cpu"),
        learning_rate=0.01,
    )
    assert matches == {8: 0}
    assert manager.logits[1] is first
    assert optimizer.state[first] is retained_state
    assert vanished not in optimizer.state
    assert all(not any(p is vanished for p in group["params"]) for group in optimizer.param_groups)
    torch.testing.assert_close(manager.logits[0], torch.zeros(3))
    torch.testing.assert_close(manager.pi()[0], torch.full((3,), 1 / 3))
