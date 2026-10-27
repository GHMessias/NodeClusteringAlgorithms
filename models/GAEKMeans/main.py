#!/usr/bin/env python3
"""Train a graph autoencoder and cluster its node embeddings with KMeans."""

from __future__ import annotations

# Optional benchmark result selection; standalone/default training is unchanged.
import os as _benchmark_os
if _benchmark_os.environ.get("BENCHMARK_SELECT_BEST_NMI") == "1":
    from benchmark_selection import selection as _nmi_selection
else:
    _nmi_selection = None


from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GAE, GCNConv

MODELS_ROOT = Path(__file__).resolve().parents[1]
if str(MODELS_ROOT) not in sys.path:
    sys.path.insert(0, str(MODELS_ROOT))

from _shared.graph_autoencoder_kmeans import (  # noqa: E402
    cluster_and_save,
    common_parser,
    prepare_inputs,
    resolve_device,
    set_seed,
    validate_common_args,
)


class GCNEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


def main() -> int:
    parser = common_parser(__doc__ or "GAE + KMeans")
    args = parser.parse_args()
    validate_common_args(args)
    set_seed(args.seed)
    device = resolve_device(args.device)
    _, features, edge_index, num_clusters = prepare_inputs(args)
    features = features.to(device)
    edge_index = edge_index.to(device)

    model = GAE(
        GCNEncoder(
            int(features.shape[1]),
            args.hidden_channels,
            args.embedding_dim,
            args.dropout,
        )
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        embeddings = model.encode(features, edge_index)
        loss = model.recon_loss(embeddings, edge_index)
        if not torch.isfinite(loss):
            raise RuntimeError(f"GAE produced a non-finite loss at epoch {epoch}")
        loss.backward()
        optimizer.step()
        if _nmi_selection is not None:
            with _nmi_selection.inference(model):
                _selection_embeddings = model.encode(features, edge_index)
                _nmi_selection.kmeans(
                    _selection_embeddings, num_clusters, epoch,
                    n_init=args.kmeans_n_init, max_iter=args.kmeans_max_iter,
                )
        value = float(loss.detach().cpu())
        history.append({"epoch": float(epoch), "reconstruction_loss": value, "total_loss": value})
        if epoch == 1 or epoch % args.log_interval == 0 or epoch == args.epochs:
            print(f"epoch={epoch:04d}/{args.epochs:04d} loss={value:.6f}", flush=True)

    model.eval()
    with torch.no_grad():
        embeddings = model.encode(features, edge_index)
    cluster_and_save(
        model_name="GAEKMeans",
        embeddings=embeddings,
        num_clusters=num_clusters,
        args=args,
        history=history,
        summary={"final_reconstruction_loss": history[-1]["reconstruction_loss"]},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
