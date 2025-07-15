# PRProject

A modular framework for graph classification experiments using PyTorch Geometric. This project is designed for flexible experimentation with graph neural networks (GNNs) on datasets such as TUDataset using a backbone-head architecture.

## Module Overview

- **GraphFW/datasets/**: Dataset loaders and utilities for graph data, including TUDataset support and preprocessing tools.
- **GraphFW/modules/**: Model definitions including backbones (GNN encoders), heads (classifiers), and complete models. Organized in subdirectories for better modularity.
- **GraphFW/runners/**: Training, validation, and testing logic. Includes base runner, K-Fold cross-validation, and split runners with early stopping and model checkpointing.
- **configs/**: Experiment configuration files with backbone-head model definitions, dataset configurations, and runner setups.
- **train.py**: Main entry point for running experiments using a config file.
- **work_dirs/**: Output directory for experiment logs, checkpoints, and results with timestamped runs.

## Quick Start Tutorial

1. **Install requirements**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare your dataset**
   - Place your dataset in the `data/` directory, or use one of the supported TUDatasets (MUTAG, ENZYMES, PROTEINS).

3. **Configure your experiment**
   - Edit or create a config file in `configs/`. Example: `configs/mutag/mutag_gcn_fold.py`.
   - Define your backbone (GNN encoder), head (classifier), model composition, dataset, and runner settings.

4. **Run training**
   ```bash
   python train.py --config configs/mutag/mutag_gcn_fold.py
   ```

5. **Check results**
   - Outputs (logs, checkpoints, metrics) are saved in `work_dirs/<experiment_name>/run_<timestamp>/`.
   - Cross-validation results include individual fold histories and aggregate statistics.

## Configuration Structure

Modern configs use a modular backbone-head architecture:

```python
# Backbone: GNN encoder (GCN, GIN, GAT, GraphSAGE)
backbone = dict(
    type='GCNv2',
    in_channels=7,  # Input feature dimension
    out_channels=64,  # Embedding dimension
    hidden_channels=64,
    num_layers=3,
    norm='batch',
    dropout_rate=0.2,
    act='relu',
)

# Head: Classification head
head = dict(
    type='MLPHead',
    in_channels=64,  # Must match backbone out_channels
    out_channels=2,  # Number of classes
    hidden_channels=32,
    num_layers=2,
    norm='batch',
    dropout_rate=0.2,
)

# Model: Combines backbone + head + pooling
model = dict(
    type='BaseOneStage',
    backbone=backbone,
    head=head,
    pooling='mean',  # Graph-level pooling
)
```

## Module Explanations

- **datasets/**: Handles loading and preprocessing of graph datasets with configurable transforms.
- **modules/backbones/**: GNN encoder architectures (GCN, GIN, GAT, GraphSAGE).
- **modules/heads/**: Classification heads and output layers.
- **modules/models/**: Complete model definitions combining backbones and heads.
- **runners/**: Implements experiment workflows with early stopping, model saving, and cross-validation.

For more details, see the README in each submodule.