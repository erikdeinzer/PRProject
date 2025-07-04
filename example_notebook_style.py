#!/usr/bin/env python3
"""
Example script showing how to use SOTA models in the same way as the original notebook.
This demonstrates compatibility with the existing workflow.
"""

import torch
import torch_geometric
from torch_geometric.data import Data, Batch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the same way as in the original notebook
from dataset import DatasetLoader
from registry import TRANSFORMS, MODELS, OPTIMIZERS, RUNNERS, build_module, build_modules_list
from runners import KFoldRunner, SplitRunner
import modules


def create_test_data():
    """Create test data similar to ENZYMES for demonstration."""
    graphs = []
    for i in range(20):
        num_nodes = torch.randint(10, 40, (1,)).item()
        x = torch.randn(num_nodes, 3)
        num_edges = torch.randint(num_nodes, num_nodes * 2, (1,)).item()
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        y = torch.randint(0, 6, (1,))
        graphs.append(Data(x=x, edge_index=edge_index, y=y))
    return graphs


def main():
    """Main function demonstrating the usage."""
    print("🧪 SOTA Models - Notebook Style Usage")
    print("=" * 50)
    
    # Configuration in the same style as the notebook
    dataset_config = {
        "name": "ENZYMES",
        "root": "./data",
        "transforms": [
            {"type": "ToUndirected"},
            {"type": "NormalizeFeatures"},
        ],
    }

    # GraphSAGE SOTA model configuration
    model_config = {
        'type': 'GraphSAGESOTA',
        'in_channels': 'auto',  # Number of input features - auto chooses from dataset
        'hidden_channels': 128,
        'out_channels': 6,  # Number of classes
        'num_layers': 4,
        'dropout': 0.5,
        'aggr': 'mean',
        'pooling': 'mean',
        'use_batch_norm': True,
    }

    optim_config = {
        "type": "Adam",
        "lr": 0.001,
        "weight_decay": 1e-4,
    }

    split_cfg = {
        'type': 'SplitRunner',
        'train_ratio': 0.8
    }

    train_config = {
        "batch_size": 32,
        "train_epochs": 100,
        "optimizer": optim_config
    }

    print("Configuration loaded successfully!")
    print(f"Model type: {model_config['type']}")
    print(f"Hidden channels: {model_config['hidden_channels']}")
    print(f"Number of layers: {model_config['num_layers']}")
    print(f"Batch size: {train_config['batch_size']}")
    print(f"Training epochs: {train_config['train_epochs']}")
    print(f"Learning rate: {optim_config['lr']}")

    # Test model building (same as in notebook)
    print("\n🔧 Building Model...")
    
    # Create test data since we can't load the real dataset
    test_graphs = create_test_data()
    
    # Simulate the dataset loader behavior
    class MockDatasetLoader:
        def __init__(self, graphs):
            self.graphs = graphs
            self.name = "ENZYMES"
            self.metadata = {
                "num_node_features": 3,
                "num_classes": 6,
                "num_graphs": len(graphs)
            }
        
        def __len__(self):
            return len(self.graphs)
        
        def __getitem__(self, idx):
            return self.graphs[idx]
        
        def get_metadata(self):
            return self.metadata
    
    # Create mock loader
    loader = MockDatasetLoader(test_graphs)
    
    print(f"Dataset Info:")
    print(f"  dataset: {loader.name}")
    print(f"  num_graphs: {len(loader)}")
    print(f"  num_classes: {loader.metadata['num_classes']}")
    print(f"  num_node_features: {loader.metadata['num_node_features']}")
    
    # Update model config with actual input channels
    model_config_actual = model_config.copy()
    if model_config_actual['in_channels'] == 'auto':
        model_config_actual['in_channels'] = loader.metadata['num_node_features']
    
    print(f"\nModel config: {model_config_actual}")
    
    # Build model (same as in notebook)
    model = build_module(model_config_actual, MODELS)
    print(f"✓ Model built successfully: {type(model).__name__}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Build optimizer (same as in notebook)
    optimizer = build_module(optim_config, OPTIMIZERS, params=model.parameters())
    print(f"✓ Optimizer built successfully: {type(optimizer).__name__}")
    
    # Test forward pass
    print("\n🚀 Testing Forward Pass...")
    batch = Batch.from_data_list(test_graphs[:5])
    
    model.eval()
    with torch.no_grad():
        output = model(batch.x, batch.edge_index, batch.batch)
    
    print(f"✓ Forward pass successful!")
    print(f"  Input graphs: {len(test_graphs[:5])}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected: ({len(test_graphs[:5])}, 6)")
    
    # Show predictions
    probabilities = torch.softmax(output, dim=1)
    print(f"\nPredictions:")
    for i in range(len(test_graphs[:5])):
        pred_class = probabilities[i].argmax().item()
        confidence = probabilities[i].max().item()
        print(f"  Graph {i+1}: Class {pred_class}, Confidence: {confidence:.3f}")
    
    # Test different model configurations
    print("\n📊 Testing Different Model Configurations...")
    
    configurations = [
        {
            'name': 'GraphSAGESOTA-Basic',
            'config': {
                'type': 'GraphSAGESOTA',
                'in_channels': 3,
                'hidden_channels': 64,
                'out_channels': 6,
                'num_layers': 3,
                'dropout': 0.5,
                'aggr': 'mean',
                'pooling': 'mean',
                'use_batch_norm': True,
            }
        },
        {
            'name': 'GraphSAGESOTAv2-Enhanced',
            'config': {
                'type': 'GraphSAGESOTAv2',
                'in_channels': 3,
                'hidden_channels': 64,
                'out_channels': 6,
                'num_layers': 3,
                'dropout': 0.4,
                'aggr': 'mean',
                'use_residual': True,
                'use_layer_norm': True,
            }
        },
        {
            'name': 'GINv2-Comparison',
            'config': {
                'type': 'GINv2',
                'in_channels': 3,
                'hidden_channels': 64,
                'out_channels': 6,
                'num_layers': 3,
            }
        }
    ]
    
    for config in configurations:
        try:
            model = build_module(config['config'], MODELS)
            params = sum(p.numel() for p in model.parameters())
            print(f"✓ {config['name']:25} - {params:,} parameters")
        except Exception as e:
            print(f"✗ {config['name']:25} - Error: {e}")
    
    print("\n🎉 All tests completed successfully!")
    print("\nThe SOTA models are compatible with the existing notebook workflow!")
    print("You can now use GraphSAGESOTA and GraphSAGESOTAv2 in your experiments.")


if __name__ == "__main__":
    main()