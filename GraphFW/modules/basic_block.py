import torch
from torch import nn

class BasicBlock(nn.Module):
    """
    Base class for a block of a neural network.
    """
    def __init__(self):
        super(BasicBlock, self).__init__()
        
    def _get_activation(self, act) -> nn.Module:
        if act == 'relu':
            return nn.ReLU
        elif act == 'elu':
            return nn.ELU
        elif act == 'leaky_relu':
            return nn.LeakyReLU
        else:
            raise ValueError(f"Unsupported activation function: {act}")

    def _get_normalization(self, norm) -> nn.Module:
        if norm == 'layer':
            return nn.LayerNorm
        elif norm == 'batch':
            return nn.BatchNorm1d
        elif norm == 'instance':
            return nn.InstanceNorm1d
        else:
            raise ValueError(f"Unsupported normalization type: {norm}")