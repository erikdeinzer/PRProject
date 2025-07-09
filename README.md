# PRProject

A modular framework for graph classification experiments using PyTorch Geometric. This project is designed for flexible experimentation with graph neural networks (GNNs) on datasets such as TUDataset.

## Module Overview

- **GraphFW/datasets/**: Dataset loaders and utilities for graph data, including TUDataset support and preprocessing tools.
- **GraphFW/modules/**: Model definitions for GNNs (e.g., GCN, GIN). Add your own models here for use in experiments.
- **GraphFW/runners/**: Training, validation, and testing logic. Includes base runner, K-Fold cross-validation, and split runners.
- **configs/**: Experiment configuration files. Define your model, dataset, dataloaders, and runner setup here.
- **train.py**: Main entry point for running experiments using a config file.
- **work_dirs/**: Output directory for experiment logs, checkpoints, and results.

## Quick Start Tutorial

1. **Install requirements**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare your dataset**
   - Place your dataset in the `data/` directory, or use one of the supported TUDatasets.

3. **Configure your experiment**
   - Edit or create a config file in `configs/`. Example: `configs/mutag/mutag_kfold.py`.
   - Specify your model, dataset, dataloader, and runner settings.

4. **Run training**
   ```bash
   python train.py --config configs/mutag/mutag_kfold.py
   ```

5. **Check results**
   - Outputs (logs, checkpoints, metrics) are saved in `work_dirs/<experiment_name>/`.

## Block Explanations

- **datasets/**: Handles loading and preprocessing of graph datasets.
- **modules/**: Contains GNN model architectures. Register new models here for use in configs.
- **runners/**: Implements experiment workflows (training, validation, testing, cross-validation).

For more details, see the README in each submodule.