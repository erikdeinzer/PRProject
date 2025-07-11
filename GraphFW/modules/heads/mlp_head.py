from torch.nn import Linear
import torch
from torch import nn
from GraphFW.build import MODULES

from GraphFW.modules.basic_block import BasicBlock

class PerceptronLayer(BasicBlock):
    def __init__(self, in_features, out_features, dropout_rate=0.0, act=None, norm=None):
        super(PerceptronLayer, self).__init__()
        self.linear = Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout_rate)
        self.act = self._get_activation(act)() if act is not None else None
        self.norm = self._get_normalization(norm)(out_features) if norm is not None else None

    def forward(self, x):
        x = self.linear(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        x = self.dropout(x)
        return x

@MODULES.register_module(type='MLPHead')
class MLPHead(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=64, num_layers=2, dropout_rate=0.0, act='relu', norm=None):
        super(MLPHead, self).__init__()
        layers = []
        self.act = act
        self.norm = norm
        # Input layer
        make_block = lambda in_c, out_c: PerceptronLayer(
            in_features=in_c,
            out_features=out_c,
            dropout_rate=dropout_rate,
            act=self.act,
            norm=self.norm
        )
        
        layers.append(make_block(in_channels, hidden_channels))
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(make_block(hidden_channels, hidden_channels))
        # Output layer without activation and normalization
        final_layer = PerceptronLayer(
            in_features=hidden_channels,
            out_features=out_channels,
            dropout_rate=0.0,  # No dropout on final layer
            act=None,          # No activation on final layer
            norm=None          # No normalization on final layer
        )
        layers.append(final_layer)
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)