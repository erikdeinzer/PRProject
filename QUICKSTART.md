# SOTA Models for ENZYMES Dataset - Quick Start Guide

This guide will help you quickly get started with the SOTA GraphSAGE models for the ENZYMES dataset.

## Prerequisites

Make sure you have the required dependencies installed:

```bash
pip install torch torchvision torch_geometric scikit-learn beautifulsoup4 requests
```

## Quick Usage Examples

### 1. Basic GraphSAGE Model

```python
from configs.enzymes.graphsage_split import *
from dataset import DatasetLoader
from registry import RUNNERS, build_module
import modules
import runners

# Load dataset
loader = DatasetLoader(**dataset_config)
print(f"Dataset: {loader.name}, Graphs: {len(loader)}")

# Build and run model
runner = build_module(
    registry=RUNNERS,
    cfg=split_cfg,
    dataset_loader=loader,
    model_config=model_config,
    train_config=train_config,
)

# Train the model
runner.run()
```

### 2. Enhanced GraphSAGE Model

```python
from configs.enzymes.graphsage_v2_split import *
from dataset import DatasetLoader
from registry import RUNNERS, build_module
import modules
import runners

# Load dataset
loader = DatasetLoader(**dataset_config)

# Build and run enhanced model
runner = build_module(
    registry=RUNNERS,
    cfg=split_cfg,
    dataset_loader=loader,
    model_config=model_config,
    train_config=train_config,
)

# Train the model
runner.run()
```

### 3. K-Fold Cross-Validation

```python
from configs.enzymes.graphsage_kfold import *
from dataset import DatasetLoader
from registry import RUNNERS, build_module
import modules
import runners

# Load dataset
loader = DatasetLoader(**dataset_config)

# Build and run with cross-validation
runner = build_module(
    registry=RUNNERS,
    cfg=split_cfg,  # Uses KFoldRunner
    dataset_loader=loader,
    model_config=model_config,
    train_config=train_config,
)

# Train with 5-fold cross-validation
runner.run()
```

## Available Configurations

| Configuration | Model | Validation | Description |
|---------------|--------|------------|-------------|
| `graphsage_split` | GraphSAGESOTA | 80/20 split | Standard GraphSAGE with optimized parameters |
| `graphsage_kfold` | GraphSAGESOTA | 5-fold CV | Same model with robust evaluation |
| `graphsage_v2_split` | GraphSAGESOTAv2 | 80/20 split | Enhanced model with residual connections |
| `gin_optimized` | GINv2 | 80/20 split | Optimized GIN for comparison |

## Model Parameters

### GraphSAGESOTA
- **Hidden channels**: 128
- **Layers**: 4
- **Dropout**: 0.5
- **Aggregation**: Mean
- **Pooling**: Mean
- **Batch norm**: Enabled

### GraphSAGESOTAv2
- **Hidden channels**: 128
- **Layers**: 4
- **Dropout**: 0.4
- **Aggregation**: Mean
- **Residual connections**: Enabled
- **Layer normalization**: Enabled
- **Multi-scale pooling**: Mean + Max

## Testing

Run the test suite to verify everything works:

```bash
python test_sota_models.py
```

Run the integration tests:

```bash
python test_integration.py
```

Run the demo:

```bash
python demo_sota_models.py
```

## Expected Results

With the ENZYMES dataset (600 graphs, 6 classes):
- **Baseline GCN**: ~60-70% accuracy
- **GIN**: ~70-80% accuracy
- **GraphSAGE SOTA**: ~75-85% accuracy (expected)

## Custom Configuration

To create your own configuration:

```python
dataset_config = {
    "name": "ENZYMES",
    "root": "./data",
    "transforms": [
        {"type": "ToUndirected"},
        {"type": "NormalizeFeatures"}
    ],
}

model_config = {
    'type': 'GraphSAGESOTA',
    'in_channels': 'auto',
    'hidden_channels': 256,  # Custom hidden size
    'out_channels': 6,
    'num_layers': 5,  # Deeper network
    'dropout': 0.3,
    'aggr': 'max',  # Max aggregation
    'pooling': 'add',  # Add pooling
    'use_batch_norm': True,
}

train_config = {
    "batch_size": 32,
    "train_epochs": 500,
    "optimizer": {
        "type": "Adam",
        "lr": 0.0001,
        "weight_decay": 1e-5,
    }
}
```

## Troubleshooting

### Dataset Download Issues
If you encounter network issues downloading the dataset, the data should already be available in the `data/` directory.

### Memory Issues
If you run out of memory, try:
- Reducing batch size
- Reducing hidden channels
- Reducing number of layers

### Import Errors
Make sure to import the modules to register the models:
```python
import modules  # This registers the models
import runners  # This registers the runners
```

## Next Steps

1. Try different configurations
2. Experiment with hyperparameters
3. Compare model performances
4. Add your own models to the registry
5. Create custom configurations for other datasets