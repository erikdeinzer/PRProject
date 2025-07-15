# GraphFW.modules

This module contains model definitions for graph neural networks (GNNs) organized in a modular backbone-head architecture for flexible experimentation.

## Contents
- `backbones/`: GNN encoder architectures (GCN, GIN, GAT, GraphSAGE) that transform node features into embeddings.
- `heads/`: Classification heads and output layers that process graph-level embeddings.
- `models/`: Complete model definitions that combine backbones and heads with pooling strategies. More abstract connectors than computational powerhouses.
- `basic_block.py`: Fundamental building blocks and utilities for model construction.
- `__init__.py`: Module initialization.

## Architecture Overview

The framework uses a modular design:

1. **Backbone**: Encodes node features into embeddings using GNN layers
2. **Pooling**: Aggregates node embeddings to graph-level representation  
3. **Head**: Processes graph embeddings for final classification

## Usage

### Configuration Structure
```python
backbone = dict(
    type='GCNv2',           # Choose: GCNv2, GINv2, GATv2, GraphSAGEv2
    in_channels=7,          # Input feature dimension
    out_channels=64,        # Embedding dimension
    hidden_channels=64,     # Hidden layer dimension
    num_layers=3,           # Number of GNN layers
    norm='batch',           # Normalization: 'batch', 'layer', None
    dropout_rate=0.2,       # Dropout probability
    act='relu',             # Activation function
)

head = dict(
    type='MLPHead',         # Classification head
    in_channels=64,         # Must match backbone out_channels
    out_channels=2,         # Number of classes
    hidden_channels=32,     # Hidden layer dimension
    num_layers=2,           # Number of MLP layers
    norm='batch',           # Normalization
    dropout_rate=0.2,       # Dropout probability
)

model = dict(
    type='BaseOneStage',    # Complete model
    backbone=backbone,      # GNN encoder
    head=head,              # Classifier
    pooling='mean',         # Graph pooling: 'mean', 'max', 'add'
)
```

## Module Registration
GraphFW uses a registry system to make models, datasets, runners, and other components easily configurable and extensible. Registration allows you to add new modules (e.g., models) and reference them by name in your config files, without changing the core codebase.

To register a backbone, head, or model, use the appropriate decorator:

```python
from GraphFW.build import MODULES

@MODULES.register_module(type='MyCustomBackbone')
class MyCustomBackbone(nn.Module):
    ...

@MODULES.register_module(type='MyCustomHead')
class MyCustomHead(nn.Module):
    ...

@MODULES.register_module(type='MyCustomModel')
class MyCustomModel(nn.Module):
    ...
```

You can then use `'type': 'MyCustomBackbone'` in your config to select this component.

## Available Registries
- `MODULES`: For all model components (backbones, heads, complete models).
- `DATASETS`: For dataset loaders (in `datasets/`).
- `RUNNERS`: For training/validation/test workflow classes (in `runners/`).
- `OPTIMIZERS`: For optimizer classes.
- `SCHEDULERS`: For learning rate schedulers.

This registry system makes the framework modular and easy to extend for new research or experiments.

---
