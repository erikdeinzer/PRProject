#!/usr/bin/env python3
"""
Test script to verify all model architectures work correctly with the fixes.
"""

import torch
from GraphFW.modules.models.base_one_stage import BaseOneStage
from GraphFW.build import build_module, MODULES
from create_synthetic_data import SyntheticDataset

def test_model_architecture(backbone_config, head_config):
    """Test a specific model architecture"""
    print(f"Testing {backbone_config['type']}...")
    
    # Create model
    model_config = dict(
        type='BaseOneStage',
        backbone=backbone_config,
        head=head_config,
        pooling='mean',
    )
    
    model = build_module(model_config, MODULES)
    model.eval()
    
    # Create dummy data
    x = torch.randn(10, backbone_config['in_channels'])
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    batch = torch.zeros(10, dtype=torch.long)
    
    # Forward pass
    with torch.no_grad():
        output = model(x, edge_index, batch)
    
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    # Test that output is not all zeros or NaN
    if torch.isnan(output).any():
        print("  ❌ ERROR: Model output contains NaN!")
        return False
    elif output.abs().max().item() < 1e-6:
        print("  ⚠️  WARNING: Model output is very small")
        return False
    else:
        print("  ✅ Model working correctly")
        return True

def main():
    print("Testing Fixed Model Architectures")
    print("=" * 50)
    
    # Common head config
    head_config = dict(
        type='MLPHead',
        in_channels=64,
        out_channels=2,
        hidden_channels=64,
        num_layers=2,
        norm='layer',
        dropout_rate=0.2,
    )
    
    # Test configurations
    test_configs = [
        {
            'type': 'GCNv2',
            'in_channels': 4,
            'out_channels': 64,
            'hidden_channels': 64,
            'num_layers': 3,
            'norm': 'layer',
            'dropout_rate': 0.2,
            'act': 'relu',
        },
        {
            'type': 'GraphSAGEv2',
            'in_channels': 4,
            'out_channels': 64,
            'hidden_channels': 64,
            'num_layers': 3,
            'norm': 'layer',
            'dropout_rate': 0.2,
            'act': 'relu',
        },
        {
            'type': 'GINv2',
            'in_channels': 4,
            'out_channels': 64,
            'hidden_channels': 64,
            'num_layers': 3,
            'norm': 'layer',
            'dropout_rate': 0.2,
            'act': 'relu',
        },
        {
            'type': 'GATv2',
            'in_channels': 4,
            'out_channels': 64,
            'hidden_channels': 64,
            'num_layers': 3,
            'n_heads': 2,
            'norm': 'layer',
            'dropout_rate': 0.2,
            'act': 'relu',
        },
    ]
    
    results = []
    for config in test_configs:
        try:
            success = test_model_architecture(config, head_config)
            results.append((config['type'], success))
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            results.append((config['type'], False))
        print()
    
    print("Test Results:")
    print("-" * 30)
    for model_type, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{model_type:15s}: {status}")
    
    total_passed = sum(1 for _, success in results if success)
    print(f"\nPassed: {total_passed}/{len(results)} models")

if __name__ == "__main__":
    main()