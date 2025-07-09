import torch_geometric.nn as nn
import torch
import torch.nn.functional as F

from GraphFW.build import MODULES

@MODULES.register_module(type='GCNv2')
class GCNv2(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=64, num_layers=4):
        super(GCNv2, self).__init__()
        self.num_layers = num_layers
        self.in_conv = nn.GCNConv(in_channels, hidden_channels)

        self.hidden_convs = self.hidden_convs = torch.nn.ModuleList([
            nn.GCNConv(hidden_channels, hidden_channels) for _ in range(num_layers - 1)
        ])
        self.lin = torch.nn.Linear(hidden_channels, out_channels)
        

    def forward(self, x, edge_index, batch):
        x = F.relu(self.in_conv(x, edge_index))
        for conv in self.hidden_convs:
            x = F.relu(conv(x, edge_index))
        x = nn.global_mean_pool(x, batch)  # Graph-level readout
        x = self.lin(x)
        return x