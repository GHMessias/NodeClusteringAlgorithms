from __future__ import annotations

import numpy as np
import torch

from graph_benchmark.utils.densgnn_loss import (
    ClusterSupport,
    adaptive_cluster_support,
    canonical_point_selection,
    canonical_support_distance,
    robust_density_alphas,
    support_cluster_distances,
)


def test_adaptative_alias_is_normalized() -> None:
    assert canonical_point_selection("adaptive") == "adaptive"
    assert canonical_point_selection("adaptative") == "adaptive"


def test_density_alpha_increases_with_relative_density() -> None:
    relative_densities, alphas = robust_density_alphas(
        {0: 2.0, 1: 1.0, 2: 0.5},
        global_radius=1.0,
    )

    assert relative_densities[0] < relative_densities[1] < relative_densities[2]
    assert alphas[0] < alphas[1] < alphas[2]
    assert all(0.0 <= alpha <= 1.0 for alpha in alphas.values())


def test_equal_cluster_densities_use_balanced_alpha() -> None:
    _, alphas = robust_density_alphas(
        {0: 1.0, 1: 1.0, 2: 1.0},
        global_radius=1.0,
    )

    assert alphas == {0: 0.5, 1: 0.5, 2: 0.5}


def test_adaptive_support_matches_weighted_core_border_distance() -> None:
    embeddings = torch.tensor([[0.0], [2.0], [10.0]], dtype=torch.float32)
    core_support = ClusterSupport(label=0, node_ids=torch.tensor([0]))
    border_support = ClusterSupport(label=0, node_ids=torch.tensor([1, 2]))
    adaptive_support = adaptive_cluster_support(
        label=0,
        core_node_ids=np.asarray([0]),
        border_node_ids=np.asarray([1, 2]),
        alpha=0.25,
        embeddings=embeddings,
    )

    core_distance = support_cluster_distances(embeddings, [core_support], mrd_k=1)
    border_distance = support_cluster_distances(embeddings, [border_support], mrd_k=1)
    adaptive_distance = support_cluster_distances(
        embeddings,
        [adaptive_support],
        mrd_k=1,
    )

    expected = 0.75 * core_distance + 0.25 * border_distance
    torch.testing.assert_close(adaptive_distance, expected)
    torch.testing.assert_close(adaptive_support.node_weights.sum(), torch.tensor(1.0))


def test_uniform_weights_preserve_original_support_mean() -> None:
    embeddings = torch.tensor([[0.0], [2.0], [10.0]], dtype=torch.float32)
    original_support = ClusterSupport(label=0, node_ids=torch.tensor([0, 2]))
    weighted_support = ClusterSupport(
        label=0,
        node_ids=torch.tensor([0, 2]),
        node_weights=torch.tensor([0.5, 0.5]),
    )

    original_distance = support_cluster_distances(
        embeddings,
        [original_support],
        mrd_k=1,
    )
    weighted_distance = support_cluster_distances(
        embeddings,
        [weighted_support],
        mrd_k=1,
    )

    torch.testing.assert_close(weighted_distance, original_distance)


def test_euclidean_distance_uses_same_adaptive_support_weights() -> None:
    embeddings = torch.tensor([[0.0], [2.0], [10.0]], dtype=torch.float32)
    support = adaptive_cluster_support(
        label=0,
        core_node_ids=np.asarray([0]),
        border_node_ids=np.asarray([1, 2]),
        alpha=0.25,
        embeddings=embeddings,
    )

    euclidean = support_cluster_distances(
        embeddings, [support], mrd_k=1, support_distance="euclidean"
    )
    mrd = support_cluster_distances(
        embeddings, [support], mrd_k=1, support_distance="mrd"
    )
    default = support_cluster_distances(embeddings, [support], mrd_k=1)

    torch.testing.assert_close(euclidean[:, 0], torch.tensor([1.5, 2.5, 8.5]))
    torch.testing.assert_close(mrd, default)
    assert torch.all(mrd >= euclidean)


def test_euclidean_distance_remains_differentiable() -> None:
    embeddings = torch.tensor([[0.0], [2.0], [10.0]], requires_grad=True)
    support = ClusterSupport(label=0, node_ids=torch.tensor([0, 2]))

    distances = support_cluster_distances(
        embeddings, [support], mrd_k=1, support_distance="euclidean"
    )
    distances.sum().backward()

    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()


def test_support_distance_rejects_unknown_values() -> None:
    assert canonical_support_distance("MRD") == "mrd"
    assert canonical_support_distance(" Euclidean ") == "euclidean"
    try:
        canonical_support_distance("cosine")
    except ValueError as exc:
        assert "support_distance" in str(exc)
    else:
        raise AssertionError("Unknown support distances must be rejected")
