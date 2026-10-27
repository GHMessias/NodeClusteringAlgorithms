#!/usr/bin/env python3
"""Train a variational graph autoencoder and cluster its embeddings with KMeans."""

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
from torch_geometric.nn import GCNConv, VGAE

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


class VariationalGCNEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float):
        super().__init__()
        self.shared = GCNConv(in_channels, hidden_channels)
        self.mu = GCNConv(hidden_channels, out_channels)
        self.logstd = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.shared(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.mu(x, edge_index), self.logstd(x, edge_index)


def main() -> int:
    parser = common_parser(__doc__ or "VGAE + KMeans")
    parser.add_argument("--kl_weight", "--kl-weight", dest="kl_weight", type=float, default=1.0)
    args = parser.parse_args()
    validate_common_args(args)
    if args.kl_weight < 0:
        raise ValueError("kl_weight must be non-negative")
    set_seed(args.seed)
    device = resolve_device(args.device)
    _, features, edge_index, num_clusters = prepare_inputs(args)
    features = features.to(device)
    edge_index = edge_index.to(device)

    model = VGAE(
        VariationalGCNEncoder(
            int(features.shape[1]),
            args.hidden_channels,
            args.embedding_dim,
            args.dropout,
        )
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history: list[dict[str, float]] = []
    num_nodes = int(features.shape[0])

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        embeddings = model.encode(features, edge_index)
        reconstruction_loss = model.recon_loss(embeddings, edge_index)
        kl_loss = model.kl_loss() / num_nodes
        loss = reconstruction_loss + args.kl_weight * kl_loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"VGAE produced a non-finite loss at epoch {epoch}")
        loss.backward()
        optimizer.step()
        if _nmi_selection is not None:
            with _nmi_selection.inference(model):
                _selection_embeddings = model.encode(features, edge_index)
                _nmi_selection.kmeans(
                    _selection_embeddings, num_clusters, epoch,
                    n_init=args.kmeans_n_init, max_iter=args.kmeans_max_iter,
                )
        record = {
            "epoch": float(epoch),
            "reconstruction_loss": float(reconstruction_loss.detach().cpu()),
            "kl_loss": float(kl_loss.detach().cpu()),
            "total_loss": float(loss.detach().cpu()),
        }
        history.append(record)
        if epoch == 1 or epoch % args.log_interval == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:04d}/{args.epochs:04d} loss={record['total_loss']:.6f} "
                f"reconstruction={record['reconstruction_loss']:.6f} kl={record['kl_loss']:.6f}",
                flush=True,
            )

    model.eval()
    with torch.no_grad():
        embeddings = model.encode(features, edge_index)
    cluster_and_save(
        model_name="VGAEKMeans",
        embeddings=embeddings,
        num_clusters=num_clusters,
        args=args,
        history=history,
        summary={
            "kl_weight": args.kl_weight,
            "final_reconstruction_loss": history[-1]["reconstruction_loss"],
            "final_kl_loss": history[-1]["kl_loss"],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
