import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import BatchNorm, LayerNorm
from registry import MODELS


@MODELS.register(type='GraphSAGESOTA')
class GraphSAGESOTA(nn.Module):
    """
    GraphSAGE model for graph classification.
    
    This is a SOTA model that uses sampling and aggregation for scalable graph neural networks.
    It supports different aggregation methods and pooling strategies.
    """
    
    def __init__(self, in_channels, out_channels, hidden_channels=128, num_layers=4, 
                 dropout=0.5, aggr='mean', pooling='mean', use_batch_norm=True):
        """
        Initialize the GraphSAGE model.
        
        Args:
            in_channels (int): Number of input features
            out_channels (int): Number of output classes
            hidden_channels (int): Number of hidden channels
            num_layers (int): Number of GraphSAGE layers
            dropout (float): Dropout probability
            aggr (str): Aggregation method ('mean', 'max', 'add')
            pooling (str): Graph pooling method ('mean', 'max', 'add')
            use_batch_norm (bool): Whether to use batch normalization
        """
        super(GraphSAGESOTA, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling = pooling
        self.use_batch_norm = use_batch_norm
        
        # GraphSAGE layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList() if use_batch_norm else None
        
        # Input layer
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
        if use_batch_norm:
            self.bns.append(BatchNorm(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
            if use_batch_norm:
                self.bns.append(BatchNorm(hidden_channels))
        
        # Output layer
        self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
        if use_batch_norm:
            self.bns.append(BatchNorm(hidden_channels))
        
        # Final classification layer
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )
        
        # Choose pooling function
        if pooling == 'mean':
            self.pool = global_mean_pool
        elif pooling == 'max':
            self.pool = global_max_pool
        elif pooling == 'add':
            self.pool = global_add_pool
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")
    
    def forward(self, x, edge_index, batch):
        """
        Forward pass of the GraphSAGE model.
        
        Args:
            x: Node features
            edge_index: Edge indices
            batch: Batch indices for graph pooling
            
        Returns:
            Graph-level predictions
        """
        # Apply GraphSAGE layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
            if self.use_batch_norm and self.bns is not None:
                x = self.bns[i](x)
            
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Graph-level pooling
        x = self.pool(x, batch)
        
        # Final classification
        x = self.classifier(x)
        
        return x


@MODELS.register(type='GraphSAGESOTAv2')
class GraphSAGESOTAv2(nn.Module):
    """
    Enhanced GraphSAGE model with residual connections and improved architecture.
    
    This version includes:
    - Residual connections
    - Layer normalization option
    - Multiple pooling strategies combined
    - Improved classifier head
    """
    
    def __init__(self, in_channels, out_channels, hidden_channels=128, num_layers=4,
                 dropout=0.5, aggr='mean', use_residual=True, use_layer_norm=False):
        """
        Initialize the enhanced GraphSAGE model.
        
        Args:
            in_channels (int): Number of input features
            out_channels (int): Number of output classes
            hidden_channels (int): Number of hidden channels
            num_layers (int): Number of GraphSAGE layers
            dropout (float): Dropout probability
            aggr (str): Aggregation method ('mean', 'max', 'add')
            use_residual (bool): Whether to use residual connections
            use_layer_norm (bool): Whether to use layer normalization
        """
        super(GraphSAGESOTAv2, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        
        # GraphSAGE layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList() if use_layer_norm else None
        
        # Input layer
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
        if use_layer_norm:
            self.norms.append(LayerNorm(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
            if use_layer_norm:
                self.norms.append(LayerNorm(hidden_channels))
        
        # Projection layer for residual connections
        if use_residual and in_channels != hidden_channels:
            self.input_proj = nn.Linear(in_channels, hidden_channels)
        else:
            self.input_proj = None
        
        # Multi-scale pooling
        self.pool_mean = global_mean_pool
        self.pool_max = global_max_pool
        
        # Enhanced classifier with multiple pooling features
        pooled_dim = hidden_channels * 2  # mean + max pooling
        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, pooled_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(pooled_dim // 2, pooled_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(pooled_dim // 4, out_channels)
        )
    
    def forward(self, x, edge_index, batch):
        """
        Forward pass of the enhanced GraphSAGE model.
        
        Args:
            x: Node features
            edge_index: Edge indices
            batch: Batch indices for graph pooling
            
        Returns:
            Graph-level predictions
        """
        # Store input for residual connection
        if self.use_residual:
            if self.input_proj is not None:
                residual = self.input_proj(x)
            else:
                residual = x
        
        # Apply GraphSAGE layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
            if self.use_layer_norm and self.norms is not None:
                x = self.norms[i](x)
            
            x = F.relu(x)
            
            # Add residual connection
            if self.use_residual and i == 0:
                x = x + residual
            
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Multi-scale graph pooling
        x_mean = self.pool_mean(x, batch)
        x_max = self.pool_max(x, batch)
        x_pooled = torch.cat([x_mean, x_max], dim=1)
        
        # Final classification
        x = self.classifier(x_pooled)
        
        return x