#!/usr/bin/env python3
"""Train an adversarially regularized graph autoencoder and apply KMeans."""

from __future__ import annotations

# Optional benchmark result selection; standalone/default training is unchanged.
import os as _benchmark_os
if _benchmark_os.environ.get("BENCHMARK_SELECT_BEST_NMI") == "1":
    from benchmark_selection import selection as _nmi_selection
else:
    _nmi_selection = None


from itertools import chain
from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import ARGA, GCNConv

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


class Discriminator(nn.Module):
    def __init__(self, embedding_dim: int, hidden_channels: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(embedding_dim, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, 1)
        self.dropout = dropout

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        embeddings = F.relu(self.linear1(embeddings))
        embeddings = F.dropout(embeddings, p=self.dropout, training=self.training)
        return self.linear2(embeddings)


def main() -> int:
    parser = common_parser(__doc__ or "ARGA + KMeans")
    parser.add_argument(
        "--discriminator_hidden_channels",
        "--discriminator-hidden-channels",
        dest="discriminator_hidden_channels",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--discriminator_lr", "--discriminator-lr", dest="discriminator_lr", type=float, default=0.01
    )
    parser.add_argument(
        "--discriminator_steps", "--discriminator-steps", dest="discriminator_steps", type=int, default=1
    )
    parser.add_argument(
        "--adversarial_weight", "--adversarial-weight", dest="adversarial_weight", type=float, default=1.0
    )
    args = parser.parse_args()
    validate_common_args(args)
    if args.discriminator_hidden_channels < 1 or args.discriminator_steps < 1:
        raise ValueError("discriminator_hidden_channels and discriminator_steps must be positive")
    if args.discriminator_lr <= 0:
        raise ValueError("discriminator_lr must be positive")
    if args.adversarial_weight < 0:
        raise ValueError("adversarial_weight must be non-negative")

    set_seed(args.seed)
    device = resolve_device(args.device)
    _, features, edge_index, num_clusters = prepare_inputs(args)
    features = features.to(device)
    edge_index = edge_index.to(device)

    model = ARGA(
        GCNEncoder(
            int(features.shape[1]),
            args.hidden_channels,
            args.embedding_dim,
            args.dropout,
        ),
        Discriminator(args.embedding_dim, args.discriminator_hidden_channels, args.dropout),
    ).to(device)
    encoder_parameters = chain(model.encoder.parameters(), model.decoder.parameters())
    encoder_optimizer = torch.optim.Adam(
        encoder_parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    discriminator_optimizer = torch.optim.Adam(
        model.discriminator.parameters(),
        lr=args.discriminator_lr,
        weight_decay=args.weight_decay,
    )
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        with torch.no_grad():
            detached_embeddings = model.encode(features, edge_index)
        discriminator_total = 0.0
        for _ in range(args.discriminator_steps):
            discriminator_optimizer.zero_grad(set_to_none=True)
            discriminator_loss = model.discriminator_loss(detached_embeddings)
            if not torch.isfinite(discriminator_loss):
                raise RuntimeError(f"ARGA discriminator produced a non-finite loss at epoch {epoch}")
            discriminator_loss.backward()
            discriminator_optimizer.step()
            discriminator_total += float(discriminator_loss.detach().cpu())

        encoder_optimizer.zero_grad(set_to_none=True)
        embeddings = model.encode(features, edge_index)
        reconstruction_loss = model.recon_loss(embeddings, edge_index)
        regularization_loss = model.reg_loss(embeddings)
        loss = reconstruction_loss + args.adversarial_weight * regularization_loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"ARGA encoder produced a non-finite loss at epoch {epoch}")
        loss.backward()
        encoder_optimizer.step()
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
            "regularization_loss": float(regularization_loss.detach().cpu()),
            "discriminator_loss": discriminator_total / args.discriminator_steps,
            "total_loss": float(loss.detach().cpu()),
        }
        history.append(record)
        if epoch == 1 or epoch % args.log_interval == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:04d}/{args.epochs:04d} loss={record['total_loss']:.6f} "
                f"reconstruction={record['reconstruction_loss']:.6f} "
                f"regularization={record['regularization_loss']:.6f} "
                f"discriminator={record['discriminator_loss']:.6f}",
                flush=True,
            )

    model.eval()
    with torch.no_grad():
        embeddings = model.encode(features, edge_index)
    cluster_and_save(
        model_name="ARGAKMeans",
        embeddings=embeddings,
        num_clusters=num_clusters,
        args=args,
        history=history,
        summary={
            "adversarial_weight": args.adversarial_weight,
            "discriminator_steps": args.discriminator_steps,
            "final_reconstruction_loss": history[-1]["reconstruction_loss"],
            "final_regularization_loss": history[-1]["regularization_loss"],
            "final_discriminator_loss": history[-1]["discriminator_loss"],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
