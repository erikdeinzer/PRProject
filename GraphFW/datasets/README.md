# GraphFW.datasets

This module contains dataset loaders and utilities for graph classification tasks. It provides interfaces to load, preprocess, and split datasets, especially those from the TUDataset collection with configurable preprocessing transforms.

## Contents
- `tudataset.py`: Loader for TUDataset with preprocessing pipeline support.
- `__init__.py`: Module initialization.

## Supported Datasets

### TUDataset Collection
- **MUTAG**: 188 molecular graphs, 2 classes, 7 node features
- **ENZYMES**: 600 protein graphs, 6 classes, 3 node features  
- **PROTEINS**: 1113 protein graphs, 2 classes, 4 node features
- All other TUDataset (although not tried yet)

## Configuration

### Dataset Configuration
```python
dataset = dict(
    type='TUDatasetLoader',
    name='MUTAG',              # Dataset name
    root='./data',             # Data directory
    transforms=[               # Preprocessing pipeline
        dict(type='ToUndirected'),      # Convert to undirected
        dict(type='NormalizeFeatures'), # Normalize node features
        dict(type='RemoveIsolatedNodes'), # Remove isolated nodes
    ],
    use_node_attr=True         # Use node attributes as features
)
```

### Available Transforms
- **ToUndirected**: Converts directed graphs to undirected
- **NormalizeFeatures**: Normalizes node features to unit norm
- **RemoveIsolatedNodes**: Removes nodes with no edges
- **AddSelfLoops**: Adds self-loops to all nodes
- All other Transforms in pytorch-geometric

### Dataset Properties
Each dataset provides:
- **Node features**: Numerical attributes per node
- **Edge connectivity**: Graph structure information  
- **Graph labels**: Classification targets
- **Metadata**: Number of classes, feature dimensions

## Usage
Configure datasets in your experiment config files. The loader handles automatic downloading, caching, and preprocessing of TUDataset collections.

### Data Directory Structure
data is automatically loaded if not present. It will be loaded to the set path.
```
data/
├── MUTAG/
│   ├── raw/          # Original dataset files
│   └── processed/    # Preprocessed PyTorch tensors
├── ENZYMES/
└── PROTEINS/
```

---
