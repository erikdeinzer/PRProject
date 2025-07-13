from torch.nn import Linear
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool
from GraphFW.build import MODULES
from GraphFW.modules.basic_block import BasicBlock


class GATBlock(BasicBlock):
    def __init__(self, in_channels, out_channels, n_heads, dropout_rate=0.0, norm=None, act=None, concat=False):
        super().__init__()
        self.gat = GATConv(in_channels, out_channels, heads=n_heads, concat=concat)
        self.norm = self._get_normalization(norm)(out_channels * n_heads) if norm is not None else nn.Identity()
        self.act = self._get_activation(act)() if act is not None else nn.Identity()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index):
        x = self.gat(x, edge_index)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x
    
class GATSkipBlock(BasicBlock):
    """GAT block with skip connection."""
    def __init__(self, in_channels, out_channels, n_heads, dropout=0.0, norm=None, act=None, concat=False):
        super().__init__()
        self.gat = GATConv(in_channels, out_channels, heads=n_heads, concat=concat)
        self.norm = self._get_normalization(norm)(out_channels * n_heads) if norm is not None else nn.Identity()
        self.act = self._get_activation(act)() if act is not None else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.residual = (in_channels == out_channels * n_heads)

    def forward(self, x, edge_index):
        identity = x
        x = self.gat(x, edge_index)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        if self.residual:
            x = x + identity  # Add skip connection if dimensions match
        return x


@MODULES.register_module(type='GATv2')
class GATv2(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,  
                 n_heads=1, 
                 hidden_channels=64,
                 num_layers=4, 
                 norm='layer', 
                 dropout_rate=0.0, 
                 act='elu', 
                 use_skip_connections =False):
        super(GATv2, self).__init__()

        self.norm_fn = norm
        self.act_fn = act

        BlockType = GATBlock if not use_skip_connections else GATSkipBlock

        make_block = lambda in_ch, out_ch, nh=n_heads, concat=True, norm=self.norm_fn, act=self.act_fn: BlockType(
            in_channels=in_ch, 
            out_channels=out_ch, 
            n_heads=nh, 
            dropout_rate=dropout_rate, 
            norm=norm, 
            act=act,
            concat=concat
        )
        self.layers = nn.ModuleList()
        self.layers.append(
            make_block(in_channels, hidden_channels)
        )
        for _ in range(num_layers - 2):
            self.layers.append(
                make_block(hidden_channels * n_heads, hidden_channels)
            )
        self.final_conv=GATConv(
                in_channels=hidden_channels * n_heads, 
                out_channels=out_channels, 
                heads=1, 
                concat=False
            )
        
                

    def forward(self, x, edge_index, batch=None):
        for layer in self.layers:
            x = layer(x, edge_index)
        x = self.final_conv(x, edge_index)

        return x