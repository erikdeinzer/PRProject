import torch
import torch.nn as nn
import torch_geometric as pyg
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
import torch.nn.functional as F
from GraphFW.build import MODULES

from torch_geometric.nn import GCNConv
from GraphFW.modules.basic_block import BasicBlock

class GCNBlock(BasicBlock):
    def __init__(self, in_channels, out_channels, dropout=0.0, norm=None, act=None):
        super(GCNBlock, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.norm = self._get_normalization(norm)(out_channels) if norm is not None else nn.Identity()
        self.act = self._get_activation(act)() if act is not None else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x
    
@MODULES.register_module(type='GCNv2')
class GCNv2(torch.nn.Module):
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
        Initialize the GCN backbone.
        Args:
            in_channels (int): Number of input features.
            out_channels (int): Number of output classes.
            hidden_channels (int): Number of hidden channels.
            num_layers (int): Number of GCN layers.
            dropout_rate (float): Dropout rate for the layers.
            act (str): Activation function to use ('relu', 'elu', 'leaky_relu').
            norm (str): Normalization type ('layer', 'batch', 'instance').
        """
        super().__init__()

        self.convs=torch.nn.ModuleList()

        self.norm_fn = norm
        self.act_fn = act

        make_block = lambda in_c, out_c: GCNBlock(
            in_channels=in_c, 
            out_channels=out_c, 
            dropout=dropout_rate, 
            norm=self.norm_fn,
            act=self.act_fn 
        )   

        self.convs.append(make_block(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(make_block(hidden_channels, hidden_channels))
        
        self.convs.append(make_block(hidden_channels, out_channels))

    def forward(self, x, edge_index, batch=None):
        for block in self.convs:
            x = block(x, edge_index)
        return x
    
    def _get_normalization(self, norm):
        if norm == 'layer':
            return nn.LayerNorm
        elif norm == 'batch':
            return nn.BatchNorm1d
        elif norm == 'instance':
            return nn.InstanceNorm1d
        else:
            raise ValueError(f"Unsupported normalization type: {norm}")
    def _get_activation(self, act):
        if act == 'relu':
            return nn.ReLU()
        elif act == 'elu':
            return nn.ELU()
        elif act == 'leaky_relu':
            return nn.LeakyReLU(0.2)
        else:
            raise ValueError(f"Unsupported activation function: {act}")