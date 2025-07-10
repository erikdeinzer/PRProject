from torch.nn import Linear
import torch
from torch import nn
from torch_geometric.nn import SAGEConv, global_mean_pool
from GraphFW.build import MODULES

# GraphSAGE block with normalization, activation, and dropout
class GraphSAGEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate, norm_layer, activation):
        super().__init__()
        # Only pass in_channels and out_channels to SAGEConv
        self.sage = SAGEConv(in_channels, out_channels)
        self.norm = norm_layer(out_channels)
        self.act = activation
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index):
        x = self.sage(x, edge_index)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x

# GraphSAGE block with skip connection
class GraphSAGESkipBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout, norm_layer, activation):
        super().__init__()
        self.sage = SAGEConv(in_channels, out_channels)
        self.norm = norm_layer(out_channels)
        self.act = activation
        self.dropout = nn.Dropout(dropout)
        self.residual = (in_channels == out_channels)

    def forward(self, x, edge_index):
        identity = x
        x = self.sage(x, edge_index)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        if self.residual:
            x = x + identity  # Add skip connection if dimensions match
        return x



# Register the GraphSAGE model in the module registry
@MODULES.register_module(type='GraphSAGEv2')
class GraphSAGEv2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels=64,
                 num_layers=4,
                 norm='layer',
                 dropout_rate=0.0,
                 activation='relu',
                 use_skip_conections =False,
                 **kwargs):
        super(GraphSAGEv2, self).__init__()
        # Remove unused kwargs to avoid passing them to SAGEConv
        norm_layer = self._get_normalization(norm)
        activation_fn = self._get_activation(activation)

        BlockType = GraphSAGEBlock if not use_skip_conections else GraphSAGESkipBlock
        self.layers = nn.ModuleList()
        # First layer
        self.layers.append(
            BlockType(in_channels, hidden_channels, dropout_rate, norm_layer, activation_fn)
        )
        # Hidden layers
        for _ in range(num_layers):
            self.layers.append(
                BlockType(hidden_channels, hidden_channels, dropout_rate, norm_layer, activation_fn)
            )
        # Final layer
        self.final_conv = SAGEConv(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout_rate)
        self.pool = global_mean_pool
        self.lin = Linear(out_channels, out_channels)

    def forward(self, x, edge_index, batch):
        for layer in self.layers:
            x = layer(x, edge_index)
        x = self.final_conv(x, edge_index)
        x = self.pool(x, batch)
        x = self.dropout(x)
        x = self.lin(x)
        return x

    def _get_activation(self, act):
        if act == 'relu':
            return nn.ReLU()
        elif act == 'elu':
            return nn.ELU()
        elif act == 'leaky_relu':
            return nn.LeakyReLU(0.2)
        else:
            raise ValueError(f"Unsupported activation function: {act}")

    def _get_normalization(self, norm):
        if norm == 'layer':
            return nn.LayerNorm
        elif norm == 'batch':
            return nn.BatchNorm1d
        elif norm == 'instance':
            return nn.InstanceNorm1d
        else:
            raise ValueError(f"Unsupported normalization type: {norm}")
