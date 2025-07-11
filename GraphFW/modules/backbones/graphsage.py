from torch.nn import Linear
import torch
from torch import nn
from torch_geometric.nn import SAGEConv, global_mean_pool
from GraphFW.build import MODULES
from GraphFW.modules.basic_block import BasicBlock
# GraphSAGE block with normalization, activation, and dropout
class GraphSAGEBlock(BasicBlock):
    def __init__(self, in_channels, out_channels, dropout_rate, norm=None, act=None):
        super().__init__()
        # Only pass in_channels and out_channels to SAGEConv
        self.sage = SAGEConv(in_channels, out_channels)
        self.norm = self._get_normalization(norm)(out_channels) if norm is not None else nn.Identity()
        self.act = self._get_activation(act)() if act is not None else nn.Identity()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index):
        x = self.sage(x, edge_index)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x

# GraphSAGE block with skip connection
class GraphSAGESkipBlock(BasicBlock):
    def __init__(self, in_channels, out_channels, dropout_rate, norm=None, act=None):
        super().__init__()
        self.sage = SAGEConv(in_channels, out_channels)
        self.norm = self._get_normalization(norm)(out_channels) if norm is not None else nn.Identity()
        self.act = self._get_activation(act)() if act is not None else nn.Identity()
        self.dropout = nn.Dropout(dropout_rate)
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
                 act='relu',
                 use_skip_connections =False):
        super(GraphSAGEv2, self).__init__()
        # Remove unused kwargs to avoid passing them to SAGEConv
        
        self.norm_fn = norm
        self.act_fn = act
        BlockType = GraphSAGEBlock if not use_skip_connections else GraphSAGESkipBlock

        make_block = lambda in_c, out_c: BlockType(
            in_channels=in_c, 
            out_channels=out_c, 
            dropout_rate=dropout_rate, 
            norm=self.norm_fn, 
            act=self.act_fn
        )

        self.layers = nn.ModuleList()
        # First layer
        self.layers.append(
            make_block(in_channels, hidden_channels)
        )
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(
                make_block(hidden_channels, hidden_channels)
            )
        # Final layer
        self.layers.append(
            make_block(hidden_channels, out_channels)
        )
        

    def forward(self, x, edge_index, batch = None):
        for layer in self.layers:
            x = layer(x, edge_index)
        return x
