# PRProject - SOTA Graph Neural Networks for Enzymes Dataset

This repository provides a comprehensive framework for graph neural network experiments on the ENZYMES dataset, featuring state-of-the-art (SOTA) models and a flexible registry-based configuration system.

## 📋 Features

- **SOTA Models**: Implementation of GraphSAGE-based models optimized for graph classification
- **Registry System**: Flexible model, optimizer, and runner registration system
- **Multiple Configurations**: Various setup options for different experimental needs
- **Cross-validation Support**: Both train/test split and K-fold cross-validation
- **Modular Design**: Easy to extend with new models and configurations

## 🚀 Quick Start

### Installation

```bash
pip install torch torchvision torch_geometric scikit-learn beautifulsoup4 requests
```

### Basic Usage

```python
from dataset import DatasetLoader
from registry import MODELS, RUNNERS, build_module
from configs.enzymes.graphsage_split import *
import modules
import runners

# Load dataset
loader = DatasetLoader(**dataset_config)
print(f"Dataset: {loader.name}, Graphs: {len(loader)}, Classes: {loader.get_metadata()['num_classes']}")

# Build and run model
runner = build_module(
    registry=RUNNERS,
    cfg=split_cfg,
    dataset_loader=loader,
    model_config=model_config,
    train_config=train_config,
)
runner.run()
```

## 🏗️ Architecture

### Registry System

The registry system allows for flexible model and configuration management:

```python
from registry import MODELS, RUNNERS, OPTIMIZERS, TRANSFORMS

# Available models
print("Models:", list(MODELS.all().keys()))
print("Runners:", list(RUNNERS.all().keys()))
```

### SOTA Models

#### GraphSAGESOTA
- **Type**: `GraphSAGESOTA`
- **Features**: Sampling and aggregation for scalable GNNs
- **Options**: Multiple aggregation methods (mean, max, add)
- **Pooling**: Global pooling strategies (mean, max, add)
- **Normalization**: Batch normalization support

#### GraphSAGESOTAv2 (Enhanced)
- **Type**: `GraphSAGESOTAv2`
- **Features**: Enhanced version with residual connections
- **Multi-scale Pooling**: Combines mean and max pooling
- **Normalization**: Layer normalization option
- **Architecture**: Deeper classifier with dropout

## 📊 Available Configurations

### ENZYMES Dataset Configurations

| Configuration | Model | Validation | Features |
|---------------|--------|------------|----------|
| `graphsage_split.py` | GraphSAGESOTA | 80/20 split | Basic GraphSAGE with optimized hyperparameters |
| `graphsage_kfold.py` | GraphSAGESOTA | 5-fold CV | Robust evaluation with cross-validation |
| `graphsage_v2_split.py` | GraphSAGESOTAv2 | 80/20 split | Enhanced model with residual connections |
| `gin_optimized.py` | GINv2 | 80/20 split | Optimized GIN for comparison |
| `base.py` | GCNv2 | 80/20 split | Basic GCN baseline |

### Configuration Examples

#### GraphSAGE with Train/Test Split
```python
# configs/enzymes/graphsage_split.py
model_config = {
    'type': 'GraphSAGESOTA',
    'in_channels': 'auto',
    'hidden_channels': 128,
    'out_channels': 6,
    'num_layers': 4,
    'dropout': 0.5,
    'aggr': 'mean',
    'pooling': 'mean',
    'use_batch_norm': True,
}
```

#### Enhanced GraphSAGE with Residual Connections
```python
# configs/enzymes/graphsage_v2_split.py
model_config = {
    'type': 'GraphSAGESOTAv2',
    'in_channels': 'auto',
    'hidden_channels': 128,
    'out_channels': 6,
    'num_layers': 4,
    'dropout': 0.4,
    'aggr': 'mean',
    'use_residual': True,
    'use_layer_norm': True,
}
```

## 🧪 Running Experiments

### Using Different Configurations

```python
# GraphSAGE with split validation
from configs.enzymes.graphsage_split import *

# GraphSAGE with K-fold cross-validation
from configs.enzymes.graphsage_kfold import *

# Enhanced GraphSAGE with residual connections
from configs.enzymes.graphsage_v2_split import *

# Optimized GIN for comparison
from configs.enzymes.gin_optimized import *
```

### Custom Configuration

```python
# Custom model configuration
model_config = {
    'type': 'GraphSAGESOTA',
    'in_channels': 3,  # ENZYMES has 3 node features
    'hidden_channels': 256,  # Larger hidden dimension
    'out_channels': 6,  # 6 enzyme classes
    'num_layers': 5,  # Deeper network
    'dropout': 0.3,  # Lower dropout
    'aggr': 'max',  # Max aggregation
    'pooling': 'add',  # Add pooling
    'use_batch_norm': True,
}

# Custom training configuration
train_config = {
    "batch_size": 128,
    "train_epochs": 300,
    "val_interval": 5,
    "log_interval": 10,
    "optimizer": {
        "type": "Adam",
        "lr": 0.0001,
        "weight_decay": 1e-5,
    }
}
```

## 🔧 Model Details

### GraphSAGESOTA Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `in_channels` | int | - | Number of input node features |
| `out_channels` | int | - | Number of output classes |
| `hidden_channels` | int | 128 | Hidden layer dimensions |
| `num_layers` | int | 4 | Number of GraphSAGE layers |
| `dropout` | float | 0.5 | Dropout probability |
| `aggr` | str | 'mean' | Aggregation method ('mean', 'max', 'add') |
| `pooling` | str | 'mean' | Graph pooling method ('mean', 'max', 'add') |
| `use_batch_norm` | bool | True | Whether to use batch normalization |

### GraphSAGESOTAv2 Additional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_residual` | bool | True | Enable residual connections |
| `use_layer_norm` | bool | False | Use layer normalization instead of batch norm |

## 📈 Expected Performance

Based on the ENZYMES dataset characteristics:
- **Dataset Size**: 600 graphs
- **Classes**: 6 enzyme types
- **Node Features**: 3 per node
- **Avg Nodes**: ~33 per graph

### Baseline Results
- **GCN**: ~60-70% accuracy
- **GIN**: ~70-80% accuracy  
- **GraphSAGE**: ~75-85% accuracy (expected with SOTA implementation)

## 🔄 Extending the Framework

### Adding New Models

1. Create a new model file in `modules/base_models/`
2. Register the model using `@MODELS.register(type='YourModel')`
3. Import in `modules/base_models/__init__.py`

```python
from registry import MODELS

@MODELS.register(type='YourModel')
class YourModel(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        # Your model implementation
    
    def forward(self, x, edge_index, batch):
        # Your forward pass
        return x
```

### Adding New Configurations

Create a new configuration file in `configs/enzymes/`:

```python
# configs/enzymes/your_config.py
dataset_config = {
    "name": "ENZYMES",
    "root": "./data",
    "transforms": [{"type": "ToUndirected"}],
}

model_config = {
    'type': 'YourModel',
    'in_channels': 'auto',
    'out_channels': 6,
    # Your model parameters
}

# ... other configurations
```

## 📁 Project Structure

```
PRProject/
├── configs/
│   ├── enzymes/
│   │   ├── base.py                  # Basic GCN configuration
│   │   ├── graphsage_split.py       # GraphSAGE with train/test split
│   │   ├── graphsage_kfold.py       # GraphSAGE with K-fold CV
│   │   ├── graphsage_v2_split.py    # Enhanced GraphSAGE
│   │   └── gin_optimized.py         # Optimized GIN
│   └── mutag/                       # MUTAG dataset configurations
├── modules/
│   └── base_models/
│       ├── gcn.py                   # GCN implementation
│       ├── gin.py                   # GIN implementation
│       └── graphsage.py             # GraphSAGE SOTA implementations
├── runners/
│   ├── split_runner.py              # Train/test split runner
│   └── kfold_runner.py              # K-fold cross-validation runner
├── dataset.py                       # Dataset loading utilities
├── registry.py                      # Model registration system
└── README.md                        # This file
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your model or configuration
4. Test thoroughly
5. Submit a pull request

## 📜 License

This project is open source. Please ensure proper attribution when using the code.

## 🙏 Acknowledgments

- PyTorch Geometric team for the excellent graph neural network library
- TU Dortmund for the ENZYMES dataset
- GraphSAGE paper authors for the original algorithm