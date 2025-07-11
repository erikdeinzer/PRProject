import torch
import torch.nn as nn
import torch_geometric as pyg
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
import torch.nn.functional as F
from GraphFW.build import MODULES


from torch_geometric.nn import GINConv
from GraphFW.modules.basic_block import BasicBlock

class GINBlock(BasicBlock):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 hidden_channels=64, 
                 dropout=0.0, 
                 norm=None,
                 act=None):
        """        
        Initialize a GIN block.
        Args:
            in_channels (int): Number of input features.
            out_channels (int): Number of output features.
            hidden_channels (int): Number of hidden channels.
            dropout (float): Dropout rate.
            norm (callable, optional): Normalization layer to apply after the convolution.
        """
        
        super(GINBlock, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )

        self.conv = GINConv(self.mlp)
        
        self.norm = self._get_normalization(norm)(out_channels) if norm is not None else nn.Identity()
        self.act = self._get_activation(act)() if act is not None else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.norm(x)
        x = self.dropout(x)
        x = self.act(x)
        return x


@MODULES.register_module(type='GINv2')
class GINv2(nn.Module):
    def __init__(
            self, 
            in_channels, 
            out_channels, 
            hidden_channels=64, 
            num_layers=4, 
            dropout_rate=0.0,
            act='relu', 
            norm='layer'):
        """
        Initialize the GIN model.
        Args:
            in_channels (int): Number of input features.
            out_channels (int): Number of output classes.
            hidden_channels (int): Number of hidden channels.
            num_layers (int): Number of GIN layers.
        """
        super().__init__()

        self.convs=torch.nn.ModuleList()
        self.act_fn = act
        self.norm_fn = norm

        make_block = lambda in_c, out_c: GINBlock(
            in_channels=in_c, 
            out_channels=out_c, 
            hidden_channels=hidden_channels, 
            dropout=dropout_rate, 
            norm=self.norm_fn, 
            act=self.act_fn
        )   

        self.convs.append(make_block(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(make_block(hidden_channels, hidden_channels))
        
        # Final layer without activation and normalization for better classification
        final_block = lambda in_c, out_c: GINBlock(
            in_channels=in_c, 
            out_channels=out_c, 
            hidden_channels=hidden_channels,  # Need hidden_channels parameter
            dropout=dropout_rate, 
            norm=None,  # No normalization on final layer
            act=None    # No activation on final layer  
        )
        self.convs.append(final_block(hidden_channels, out_channels))

    def forward(self, x, edge_index, batch=None):
        for block in self.convs:
            x = block(x, edge_index)
        return x