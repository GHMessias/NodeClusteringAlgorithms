"""Selectable two-layer DensGNN encoders; the default retains the original GCN.

GAT and GraphSAGE follow the encoder definitions already used by DeepDensGNN.
The graph autoencoder decoder and DensGNN training objectives remain unchanged.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv

from graph_benchmark.models.gae import TwoLayerGCNEncoder


class TwoLayerEncoder(nn.Module):
    def __init__(self, conv1: nn.Module, conv2: nn.Module, dropout: float) -> None:
        super().__init__()
        self.conv1 = conv1
        self.conv2 = conv2
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


def build_densgnn_encoder(
    encoder_type: str,
    input_channels: int,
    hidden_channels: int,
    embedding_channels: int,
    dropout: float = 0.0,
) -> nn.Module:
    if encoder_type == "gcn":
        return TwoLayerGCNEncoder(input_channels, hidden_channels, embedding_channels, dropout)
    if encoder_type == "gat":
        return TwoLayerEncoder(
            GATConv(input_channels, hidden_channels, heads=1, concat=False),
            GATConv(hidden_channels, embedding_channels, heads=1, concat=False),
            dropout,
        )
    if encoder_type == "sage":
        return TwoLayerEncoder(
            SAGEConv(input_channels, hidden_channels, aggr="mean"),
            SAGEConv(hidden_channels, embedding_channels, aggr="mean"),
            dropout,
        )
    raise ValueError(f"Invalid encoder_type {encoder_type!r}; expected gcn, gat or sage")
