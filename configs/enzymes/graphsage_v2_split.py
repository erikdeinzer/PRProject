"""
Enhanced GraphSAGE (v2) SOTA configuration for ENZYMES dataset.
This configuration uses the enhanced GraphSAGE model with residual connections and multi-scale pooling.
"""

dataset_config = {
    "name": "ENZYMES",
    "root": "./data",
    "transforms": [
        {"type": "ToUndirected"},
        {"type": "NormalizeFeatures"}
    ],
}

model_config = {
    'type': 'GraphSAGESOTAv2',
    'in_channels': 'auto',  # Number of input features - auto chooses from dataset
    'hidden_channels': 128,
    'out_channels': 6,  # Number of enzyme classes
    'num_layers': 4,
    'dropout': 0.4,  # Slightly lower dropout for enhanced model
    'aggr': 'mean',
    'use_residual': True,
    'use_layer_norm': True,
}

optim_config = {
    "type": "Adam",
    "lr": 0.0005,  # Lower learning rate for enhanced model
    "weight_decay": 1e-4,  # L2 regularization
}

split_cfg = {
    'type': 'SplitRunner',
    'train_ratio': 0.8,
    'shuffle': True,
    'random_state': 42,
}

train_config = {
    "batch_size": 32,  # Smaller batch size for enhanced model
    "train_epochs": 250,  # More epochs for enhanced model
    "val_interval": 10,  # Validate every 10 epochs
    "log_interval": 25,  # Log every 25 epochs
    "optimizer": optim_config
}