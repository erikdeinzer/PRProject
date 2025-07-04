#!/usr/bin/env python3
"""
Example usage of SOTA models for ENZYMES dataset.
This script demonstrates how to use the GraphSAGE SOTA models.
"""

import os
import sys
import torch
import torch_geometric
from torch_geometric.data import Data, Batch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import modules
import runners
from registry import MODELS, RUNNERS, OPTIMIZERS, build_module


def create_demo_data():
    """Create demo data similar to ENZYMES dataset."""
    print("📊 Creating Demo Data")
    print("=" * 40)
    
    # Create 10 small graphs similar to ENZYMES
    graphs = []
    for i in range(10):
        # Random number of nodes (10-50, similar to ENZYMES)
        num_nodes = torch.randint(10, 51, (1,)).item()
        
        # 3 node features (like ENZYMES)
        x = torch.randn(num_nodes, 3)
        
        # Create random edges
        num_edges = torch.randint(num_nodes, num_nodes * 3, (1,)).item()
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        
        # Random enzyme class (0-5, 6 classes total)
        y = torch.randint(0, 6, (1,))
        
        graphs.append(Data(x=x, edge_index=edge_index, y=y))
    
    print(f"Created {len(graphs)} demo graphs")
    print(f"Average nodes per graph: {sum(g.num_nodes for g in graphs) / len(graphs):.1f}")
    print(f"Node feature dimensions: {graphs[0].x.shape[1]}")
    print(f"Number of classes: {6}")
    
    return graphs


def demo_graphsage_sota():
    """Demonstrate GraphSAGE SOTA model."""
    print("\n🚀 GraphSAGE SOTA Model Demo")
    print("=" * 40)
    
    # Create demo data
    graphs = create_demo_data()
    batch = Batch.from_data_list(graphs)
    
    # Configure GraphSAGE SOTA model
    model_config = {
        'type': 'GraphSAGESOTA',
        'in_channels': 3,  # ENZYMES has 3 node features
        'hidden_channels': 64,
        'out_channels': 6,  # 6 enzyme classes
        'num_layers': 3,
        'dropout': 0.5,
        'aggr': 'mean',
        'pooling': 'mean',
        'use_batch_norm': True,
    }
    
    # Build model
    model = build_module(model_config, MODELS)
    print(f"Model: {type(model).__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(batch.x, batch.edge_index, batch.batch)
        probabilities = torch.softmax(output, dim=1)
    
    print(f"Input graphs: {len(graphs)}")
    print(f"Output shape: {output.shape}")
    print(f"Predictions shape: {probabilities.shape}")
    
    # Show predictions for first few graphs
    print("\nPredictions for first 5 graphs:")
    for i in range(min(5, len(graphs))):
        predicted_class = probabilities[i].argmax().item()
        confidence = probabilities[i].max().item()
        actual_class = graphs[i].y.item()
        print(f"  Graph {i+1}: Predicted={predicted_class}, Actual={actual_class}, Confidence={confidence:.3f}")


def demo_graphsage_sota_v2():
    """Demonstrate Enhanced GraphSAGE SOTA model."""
    print("\n🔥 GraphSAGE SOTA v2 Model Demo")
    print("=" * 40)
    
    # Create demo data
    graphs = create_demo_data()
    batch = Batch.from_data_list(graphs)
    
    # Configure enhanced GraphSAGE SOTA model
    model_config = {
        'type': 'GraphSAGESOTAv2',
        'in_channels': 3,  # ENZYMES has 3 node features
        'hidden_channels': 64,
        'out_channels': 6,  # 6 enzyme classes
        'num_layers': 3,
        'dropout': 0.4,
        'aggr': 'mean',
        'use_residual': True,
        'use_layer_norm': True,
    }
    
    # Build model
    model = build_module(model_config, MODELS)
    print(f"Model: {type(model).__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Features: Residual connections, Layer normalization, Multi-scale pooling")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(batch.x, batch.edge_index, batch.batch)
        probabilities = torch.softmax(output, dim=1)
    
    print(f"Input graphs: {len(graphs)}")
    print(f"Output shape: {output.shape}")
    print(f"Predictions shape: {probabilities.shape}")
    
    # Show predictions for first few graphs
    print("\nPredictions for first 5 graphs:")
    for i in range(min(5, len(graphs))):
        predicted_class = probabilities[i].argmax().item()
        confidence = probabilities[i].max().item()
        actual_class = graphs[i].y.item()
        print(f"  Graph {i+1}: Predicted={predicted_class}, Actual={actual_class}, Confidence={confidence:.3f}")


def show_available_configurations():
    """Show available configurations."""
    print("\n📋 Available Configurations")
    print("=" * 40)
    
    configurations = [
        {
            'name': 'GraphSAGE Split',
            'file': 'configs.enzymes.graphsage_split',
            'description': 'GraphSAGE with 80/20 train/test split'
        },
        {
            'name': 'GraphSAGE K-Fold',
            'file': 'configs.enzymes.graphsage_kfold',
            'description': 'GraphSAGE with 5-fold cross-validation'
        },
        {
            'name': 'GraphSAGE v2 Split',
            'file': 'configs.enzymes.graphsage_v2_split',
            'description': 'Enhanced GraphSAGE with residual connections'
        },
        {
            'name': 'GIN Optimized',
            'file': 'configs.enzymes.gin_optimized',
            'description': 'Optimized GIN for comparison'
        },
        {
            'name': 'GCN Baseline',
            'file': 'configs.enzymes.base',
            'description': 'Basic GCN baseline'
        }
    ]
    
    for config in configurations:
        print(f"• {config['name']:20} - {config['description']}")
        print(f"  File: {config['file']}")
        print()


def main():
    """Run the demo."""
    print("🧪 SOTA Models for ENZYMES Dataset - Demo")
    print("=" * 60)
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch Geometric: {torch_geometric.__version__}")
    print("=" * 60)
    
    # Show available configurations
    show_available_configurations()
    
    # Demo GraphSAGE SOTA
    demo_graphsage_sota()
    
    # Demo GraphSAGE SOTA v2
    demo_graphsage_sota_v2()
    
    print("\n🎉 Demo completed successfully!")
    print("\nTo run with real ENZYMES dataset, use:")
    print("  python -c \"from configs.enzymes.graphsage_split import *; # your training code\"")


if __name__ == "__main__":
    main()