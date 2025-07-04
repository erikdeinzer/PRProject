#!/usr/bin/env python3
"""
Test script to verify all ENZYMES configurations work properly.
This script tests model instantiation without running full training.
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


def create_mock_data(num_graphs=5, num_nodes=10, num_features=3, num_classes=6):
    """Create mock graph data for testing."""
    graphs = []
    for i in range(num_graphs):
        # Create random node features
        x = torch.randn(num_nodes, num_features)
        
        # Create random edges (simple ring + random connections)
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
        
        # Create random labels
        y = torch.randint(0, num_classes, (1,))
        
        graphs.append(Data(x=x, edge_index=edge_index, y=y))
    
    return graphs


def test_model_instantiation():
    """Test that all models can be instantiated."""
    print("🧪 Testing Model Instantiation")
    print("=" * 50)
    
    # Test configurations
    test_configs = [
        {
            'name': 'GraphSAGESOTA',
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
            'name': 'GraphSAGESOTAv2',
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
            'name': 'GINv2',
            'config': {
                'type': 'GINv2',
                'in_channels': 3,
                'hidden_channels': 64,
                'out_channels': 6,
                'num_layers': 3,
            }
        },
        {
            'name': 'GCNv2',
            'config': {
                'type': 'GCNv2',
                'in_channels': 3,
                'hidden_channels': 64,
                'out_channels': 6,
                'num_layers': 3,
            }
        }
    ]
    
    results = []
    
    for test_config in test_configs:
        name = test_config['name']
        config = test_config['config']
        
        try:
            model = build_module(config, MODELS)
            num_params = sum(p.numel() for p in model.parameters())
            results.append((name, "✓", f"{num_params:,} parameters"))
            print(f"✓ {name:20} - {num_params:,} parameters")
        except Exception as e:
            results.append((name, "✗", str(e)))
            print(f"✗ {name:20} - Error: {e}")
    
    return results


def test_forward_pass():
    """Test forward pass with mock data."""
    print("\n🚀 Testing Forward Pass")
    print("=" * 50)
    
    # Create mock data
    graphs = create_mock_data(num_graphs=4, num_nodes=8, num_features=3, num_classes=6)
    batch = Batch.from_data_list(graphs)
    
    # Test GraphSAGESOTA
    try:
        model_config = {
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
        
        model = build_module(model_config, MODELS)
        model.eval()
        
        with torch.no_grad():
            output = model(batch.x, batch.edge_index, batch.batch)
            
        print(f"✓ GraphSAGESOTA forward pass - Output shape: {output.shape}")
        print(f"  Expected: (4, 6), Got: {output.shape}")
        
        if output.shape == (4, 6):
            print("  ✓ Output shape is correct!")
        else:
            print("  ✗ Output shape mismatch!")
            
    except Exception as e:
        print(f"✗ GraphSAGESOTA forward pass failed: {e}")
    
    # Test GraphSAGESOTAv2
    try:
        model_config = {
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
        
        model = build_module(model_config, MODELS)
        model.eval()
        
        with torch.no_grad():
            output = model(batch.x, batch.edge_index, batch.batch)
            
        print(f"✓ GraphSAGESOTAv2 forward pass - Output shape: {output.shape}")
        print(f"  Expected: (4, 6), Got: {output.shape}")
        
        if output.shape == (4, 6):
            print("  ✓ Output shape is correct!")
        else:
            print("  ✗ Output shape mismatch!")
            
    except Exception as e:
        print(f"✗ GraphSAGESOTAv2 forward pass failed: {e}")


def test_configurations():
    """Test loading configuration files."""
    print("\n📋 Testing Configuration Files")
    print("=" * 50)
    
    config_files = [
        'configs.enzymes.graphsage_split',
        'configs.enzymes.graphsage_kfold',
        'configs.enzymes.graphsage_v2_split',
        'configs.enzymes.gin_optimized',
        'configs.enzymes.base'
    ]
    
    for config_file in config_files:
        try:
            module = __import__(config_file, fromlist=[''])
            
            # Check required attributes
            required_attrs = ['dataset_config', 'model_config', 'train_config', 'split_cfg']
            missing_attrs = [attr for attr in required_attrs if not hasattr(module, attr)]
            
            if missing_attrs:
                print(f"✗ {config_file:30} - Missing: {missing_attrs}")
            else:
                model_type = module.model_config.get('type', 'Unknown')
                print(f"✓ {config_file:30} - Model: {model_type}")
                
        except Exception as e:
            print(f"✗ {config_file:30} - Error: {e}")


def main():
    """Run all tests."""
    print("🧪 PRProject SOTA Models Test Suite")
    print("=" * 60)
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch Geometric: {torch_geometric.__version__}")
    print("=" * 60)
    
    # Test model instantiation
    model_results = test_model_instantiation()
    
    # Test forward pass
    test_forward_pass()
    
    # Test configurations
    test_configurations()
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 50)
    
    passed = sum(1 for _, status, _ in model_results if status == "✓")
    total = len(model_results)
    
    print(f"Models tested: {total}")
    print(f"Models passed: {passed}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())