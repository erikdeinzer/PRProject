"""
GraphSAGE SOTA configuration for ENZYMES dataset.
This configuration uses GraphSAGE with optimized hyperparameters for the ENZYMES dataset.
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
    'type': 'GraphSAGESOTA',
    'in_channels': 'auto',  # Number of input features - auto chooses from dataset
    'hidden_channels': 128,
    'out_channels': 6,  # Number of enzyme classes
    'num_layers': 4,
    'dropout': 0.5,
    'aggr': 'mean',
    'pooling': 'mean',
    'use_batch_norm': True,
}

optim_config = {
    "type": "Adam",
    "lr": 0.001,  # Lower learning rate for better convergence
    "weight_decay": 1e-4,  # L2 regularization
}

split_cfg = {
    'type': 'SplitRunner',
    'train_ratio': 0.8,
    'shuffle': True,
    'random_state': 42,
}

train_config = {
    "batch_size": 64,  # Larger batch size for stable training
    "train_epochs": 200,  # More epochs for better convergence
    "val_interval": 10,  # Validate every 10 epochs
    "log_interval": 20,  # Log every 20 epochs
    "optimizer": optim_config
}