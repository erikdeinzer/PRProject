from torch.nn import Linear
import torch
from torch import nn
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from GraphFW.build import MODULES, build_module

@MODULES.register_module(type='BaseOneStage')
class BaseOneStage(nn.Module):
    """
    Base class for one-stage models in GraphFW.
    This class serves as a template for models that consist of a backbone and a head,
    with optional pooling operations.
    Args:
        backbone (dict): Configuration for the backbone module.
        head (dict): Configuration for the head module.
        pooling (str): Type of pooling operation to apply ('mean', 'sum', 'max').
    """
    def __init__(self, backbone, head, pooling='mean'):
        super(BaseOneStage, self).__init__()
        self.backbone = build_module(backbone, MODULES)
        self.head = build_module(head, MODULES)
        self.pooling = self._get_pooling(pooling)

    def forward(self, x, edge_index, batch=None):
        x = self.backbone(x, edge_index, batch)
        if batch is not None:
            x = self.pooling(x, batch)
        return self.head(x)

    def _get_pooling(self, pool):
        if pool == 'mean':
            return global_mean_pool
        elif pool == 'sum':
            return global_add_pool
        elif pool == 'max':
            return global_max_pool
        else:
            raise ValueError(f"Unsupported pooling type: {pool}")