import torch
import torch_geometric as pyg
import torch_geometric.nn as nn
import torch.nn.functional as F
from registry import MODELS

@MODELS.register(type='GINv2')
class GINv2(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=64, num_layers=4):
        """
        Initialize the GIN model.
        Args:
            in_channels (int): Number of input features.
            out_channels (int): Number of output classes.
            hidden_channels (int): Number of hidden channels.
            num_layers (int): Number of GIN layers.
        """
        super().__init__()

        self.modules=torch.nn.ModuleList()
        
        for i in range(num_layers - 1):
            if i == 0:
                lay = nn.Sequential(nn.Linear(in_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
            else:
                lay = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
            self.modules.append(nn.GINConv(lay))

        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        for conv in self.modules:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = nn.global_add_pool(x, batch)
        return self.lin(x)