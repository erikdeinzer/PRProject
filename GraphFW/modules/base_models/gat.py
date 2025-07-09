from torch.nn import Linear
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool
from GraphFW.build import MODULES

@MODULES.register_module(type='GATv2')
class GATv2(nn.Module):
    def __init__(self, in_channels, hidden_channels, n_heads, out_channels, **kwargs):
        super(GATv2, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, n_heads)
        self.conv2 = GATConv(hidden_channels*n_heads, hidden_channels, n_heads)
        self.conv3 = GATConv(hidden_channels*n_heads, hidden_channels, 1)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x
