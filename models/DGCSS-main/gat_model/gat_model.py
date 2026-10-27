import torch
import torch.nn as nn
import logging
from torch_geometric.nn import GATConv
import torch.nn.functional as F


class MultiLayerGAT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        heads: int = 1,
        dropout: float = 0.6,
        concat: bool = True
    ):
        """
        A multilayer Graph Attention Network (GAT).

        Args:
            in_channels (int): Number of features in the input.
            hidden_channels (int): Number of hidden units per head.
            out_channels (int): Number of features in the output.
            num_layers (int): Total number of layers in the network.
            heads (int): Number of attention heads in hidden layers.
            dropout (float): Dropout probability.
            concat (bool): Whether to concatenate multi-head outputs in hidden layers.
                           For the final layer, it is typical to average (concat=False).
        """
        super(MultiLayerGAT, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()

        # First layer: from input to hidden dimension.
        self.convs.append(
            GATConv(
                in_channels,
                hidden_channels,
                heads=heads,
                dropout=dropout,
                add_self_loops=True,
                concat=concat
            )
        )

        # Hidden layers: if num_layers > 2 then add (num_layers-2) hidden layers.
        for _ in range(num_layers - 2):
            # When concatenating, the input dimension is hidden_channels * heads.
            self.convs.append(
                GATConv(
                    hidden_channels * heads if concat else hidden_channels,
                    hidden_channels,
                    heads=heads,
                    dropout=dropout,
                    add_self_loops=True,
                    concat=concat
                )
            )

        # Final layer: typically, we do not concatenate (i.e. average the heads).
        final_in_channels = hidden_channels * heads if (num_layers > 1 and concat) else hidden_channels
        self.convs.append(
            GATConv(
                final_in_channels,
                out_channels,
                heads=1,  # Single head for the final output.
                dropout=dropout,
                add_self_loops=True,
                concat=False
            )
        )

        logging.info(
            f"MultiLayerGAT initialized with structure: {in_channels} -> " +
            " -> ".join([str(hidden_channels)] * (num_layers - 1)) + f" -> {out_channels}"
        )

    def forward(self, x, edge_index, return_all_attn: bool = False):
        # Ensure return_all_attn is a Python bool
        if isinstance(return_all_attn, torch.Tensor):
            return_all_attn = bool(return_all_attn.flatten()[0].item())
        else:
            return_all_attn = bool(return_all_attn)
        
        attn_list = []  # To store attention weights from layers if desired.
        for i, conv in enumerate(self.convs):
            # For the first layer, request attention weights.
            if i == 0:
                x, attn = conv(x, edge_index, return_attention_weights=True)
                attn_list.append(attn)
            else:
                x = conv(x, edge_index)
            # Apply nonlinearity and dropout except for the final layer.
            if i < self.num_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        # Swap the order so that attention info is first, matching the original GATLayer.
        if return_all_attn:
            return (attn_list, x)
        else:
            return (attn_list[0] if attn_list else None, x)

class GATLayer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GATLayer, self).__init__()

        self.gat_conv1 = GATConv(in_channels, hidden_channels, add_self_loops=True)
        self.gat_conv2 = GATConv(hidden_channels, out_channels, add_self_loops=True)

        msg = (
            "Network structure: "
            + str(in_channels)
            + ", "
            + str(hidden_channels)
            + ", "
            + str(out_channels)
        )
        logging.info(msg)

    # IMPORTANT: Just the Att matrix from one layer is being returned from this method.
    def forward(self, x, edge_index, weights=None):
        x, att_tuple = self.gat_conv1(
            x, edge_index, weights, return_attention_weights=True
        )
        x = F.relu(x)
        x = self.gat_conv2(x, edge_index, weights)

        return att_tuple, x