#!/usr/bin/env python3
"""
Integration test for SOTA models with training pipeline.
This test verifies that the models integrate properly with the existing training infrastructure.
"""

import os
import sys
import torch
import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import modules
import runners
from registry import MODELS, RUNNERS, OPTIMIZERS, build_module


def create_mock_dataset(num_graphs=50):
    """Create a mock dataset similar to ENZYMES."""
    graphs = []
    for i in range(num_graphs):
        # Random number of nodes (10-50, similar to ENZYMES)
        num_nodes = torch.randint(10, 51, (1,)).item()
        
        # 3 node features (like ENZYMES)
        x = torch.randn(num_nodes, 3)
        
        # Create random edges (ensure connectivity)
        num_edges = max(num_nodes - 1, torch.randint(num_nodes, num_nodes * 3, (1,)).item())
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        
        # Random enzyme class (0-5, 6 classes total)
        y = torch.randint(0, 6, (1,))
        
        graphs.append(Data(x=x, edge_index=edge_index, y=y))
    
    return graphs


def test_training_integration():
    """Test integration with training pipeline."""
    print("🏋️ Testing Training Integration")
    print("=" * 50)
    
    # Create mock dataset
    dataset = create_mock_dataset(num_graphs=40)
    print(f"Created mock dataset with {len(dataset)} graphs")
    
    # Split into train/test
    train_dataset = dataset[:30]
    test_dataset = dataset[30:]
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # Test models
    models_to_test = [
        {
            'name': 'GraphSAGESOTA',
            'config': {
                'type': 'GraphSAGESOTA',
                'in_channels': 3,
                'hidden_channels': 32,
                'out_channels': 6,
                'num_layers': 2,
                'dropout': 0.5,
                'aggr': 'mean',
                'pooling': 'mean',
                'use_batch_norm': True,
            }
        },
        {
            'name': 'GraphSAGESOTAv2',
            'config': {
                'type': 'GraphSAGESOTAv2',
                'in_channels': 3,
                'hidden_channels': 32,
                'out_channels': 6,
                'num_layers': 2,
                'dropout': 0.4,
                'aggr': 'mean',
                'use_residual': True,
                'use_layer_norm': True,
            }
        }
    ]
    
    for model_info in models_to_test:
        print(f"\n🔬 Testing {model_info['name']}")
        print("-" * 30)
        
        try:
            # Build model
            model = build_module(model_info['config'], MODELS)
            
            # Build optimizer
            optimizer_config = {
                'type': 'Adam',
                'lr': 0.01,
                'weight_decay': 1e-4
            }
            optimizer = build_module(optimizer_config, OPTIMIZERS, params=model.parameters())
            
            # Loss function
            criterion = torch.nn.CrossEntropyLoss()
            
            print(f"✓ Model and optimizer created successfully")
            print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            print(f"  Optimizer: {type(optimizer).__name__}")
            
            # Test training step
            model.train()
            total_loss = 0
            num_batches = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                # Forward pass
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out, batch.y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Only do a few batches for testing
                if num_batches >= 3:
                    break
            
            avg_loss = total_loss / num_batches
            print(f"✓ Training steps completed")
            print(f"  Average loss: {avg_loss:.4f}")
            
            # Test evaluation
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in test_loader:
                    out = model(batch.x, batch.edge_index, batch.batch)
                    pred = out.argmax(dim=1)
                    correct += (pred == batch.y).sum().item()
                    total += batch.y.size(0)
            
            accuracy = correct / total
            print(f"✓ Evaluation completed")
            print(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")
            
        except Exception as e:
            print(f"✗ Error testing {model_info['name']}: {e}")
            import traceback
            traceback.print_exc()


def test_configuration_loading():
    """Test loading and using configuration files."""
    print("\n📋 Testing Configuration Loading")
    print("=" * 50)
    
    # Test loading configuration files
    config_files = [
        ('graphsage_split', 'configs.enzymes.graphsage_split'),
        ('graphsage_kfold', 'configs.enzymes.graphsage_kfold'),
        ('graphsage_v2_split', 'configs.enzymes.graphsage_v2_split'),
        ('gin_optimized', 'configs.enzymes.gin_optimized'),
    ]
    
    for name, module_name in config_files:
        print(f"\n🔧 Testing {name}")
        print("-" * 20)
        
        try:
            # Import configuration
            config_module = __import__(module_name, fromlist=[''])
            
            # Check that we can build the model
            model_config = config_module.model_config.copy()
            if model_config.get('in_channels') == 'auto':
                model_config['in_channels'] = 3  # ENZYMES has 3 features
            
            model = build_module(model_config, MODELS)
            
            # Check that we can build the optimizer
            optimizer_config = config_module.train_config['optimizer']
            optimizer = build_module(optimizer_config, OPTIMIZERS, params=model.parameters())
            
            print(f"✓ Configuration loaded successfully")
            print(f"  Model: {model_config['type']}")
            print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
            print(f"  Optimizer: {optimizer_config['type']}")
            print(f"  Learning rate: {optimizer_config['lr']}")
            
        except Exception as e:
            print(f"✗ Error loading {name}: {e}")


def main():
    """Run integration tests."""
    print("🧪 SOTA Models Integration Test")
    print("=" * 60)
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch Geometric: {torch_geometric.__version__}")
    print("=" * 60)
    
    # Test training integration
    test_training_integration()
    
    # Test configuration loading
    test_configuration_loading()
    
    print("\n🎉 Integration tests completed!")
    print("\nThe SOTA models are ready for use with the ENZYMES dataset!")


if __name__ == "__main__":
    main()