"""Track learnable density-scale weights across changing HDBSCAN clusters."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn

from graph_benchmark.utils.densgnn_loss import ClusterSupport
from density import validate_candidate_k


class ClusterScaleManager(nn.Module):
    """One trainable logit vector per current HDBSCAN cluster.

    Cluster numeric labels are never treated as identities. On refresh, a
    one-to-one maximum-Jaccard assignment matches full member sets. Matched
    clusters retain the exact Parameter (and Adam state); new clusters start
    with zero logits; vanished clusters and optimizer state are discarded.
    """

    def __init__(
        self,
        candidate_k: Sequence[int],
        temperature: float = 1.0,
        min_jaccard: float = 0.25,
    ) -> None:
        super().__init__()
        self.candidate_k = validate_candidate_k(candidate_k)
        if temperature <= 0:
            raise ValueError("density_temperature must be positive")
        if not 0 <= min_jaccard <= 1:
            raise ValueError("min_jaccard must be in [0, 1]")
        self.temperature = float(temperature)
        self.min_jaccard = float(min_jaccard)
        self.logits = nn.ParameterList()
        self.labels: list[int] = []
        self.memberships: list[frozenset[int]] = []

    def refresh(
        self,
        raw_labels: np.ndarray,
        supports: list[ClusterSupport],
        optimizer: torch.optim.Optimizer,
        *,
        device: torch.device,
        learning_rate: float,
    ) -> dict[int, int]:
        """Return {new HDBSCAN label: prior label} for accepted matches."""

        new_memberships = [frozenset(np.flatnonzero(raw_labels == s.label).tolist()) for s in supports]
        old_parameters = list(self.logits)
        similarity = np.zeros((len(old_parameters), len(supports)), dtype=np.float64)
        for i, old in enumerate(self.memberships):
            for j, new in enumerate(new_memberships):
                union = len(old | new)
                similarity[i, j] = len(old & new) / union if union else 0.0

        reused: dict[int, int] = {}
        if similarity.size:
            eligible_scores = np.where(similarity >= self.min_jaccard, similarity, 0.0)
            old_rows, new_cols = linear_sum_assignment(-eligible_scores)
            reused = {
                int(j): int(i)
                for i, j in zip(old_rows, new_cols, strict=True)
                if similarity[i, j] >= self.min_jaccard and similarity[i, j] > 0
            }

        new_parameters: list[nn.Parameter] = []
        for j in range(len(supports)):
            if j in reused:
                new_parameters.append(old_parameters[reused[j]])
            else:
                parameter = nn.Parameter(torch.zeros(len(self.candidate_k), device=device))
                new_parameters.append(parameter)
                optimizer.add_param_group({"params": [parameter], "lr": learning_rate, "weight_decay": 0.0})

        retained_ids = {id(parameter) for parameter in new_parameters}
        for parameter in old_parameters:
            if id(parameter) not in retained_ids:
                optimizer.param_groups[:] = [
                    group for group in optimizer.param_groups
                    if not any(member is parameter for member in group["params"])
                ]
                optimizer.state.pop(parameter, None)

        prior_labels = self.labels
        self.logits = nn.ParameterList(new_parameters)
        self.labels = [int(s.label) for s in supports]
        self.memberships = new_memberships
        return {self.labels[j]: prior_labels[i] for j, i in reused.items()}

    def pi(self) -> torch.Tensor:
        if not self.logits:
            raise ValueError("No current clusters: refresh the manager first")
        return torch.softmax(torch.stack(list(self.logits), dim=0) / self.temperature, dim=1)

    def get_density_scale_info(self) -> dict[str, object]:
        """Return Pi and diagnostic k_eff, both aligned with current labels."""

        with torch.no_grad():
            probabilities = self.pi()
            ranks = torch.as_tensor(self.candidate_k, device=probabilities.device, dtype=probabilities.dtype)
            effective_k = probabilities @ ranks
            return {
                "cluster_labels": list(self.labels),
                "candidate_k": list(self.candidate_k),
                "pi": probabilities.cpu().tolist(),
                "k_eff": effective_k.cpu().tolist(),
            }
