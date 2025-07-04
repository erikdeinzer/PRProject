"""
GraphSAGE SOTA configuration for ENZYMES dataset with K-Fold Cross-Validation.
This configuration provides robust evaluation using 5-fold cross-validation.
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
    "type": "KFoldRunner",
    "num_folds": 5,
    "shuffle": True,
    "random_state": 42,
}

train_config = {
    "batch_size": 64,  # Larger batch size for stable training
    "train_epochs": 150,  # Fewer epochs per fold
    "val_interval": 10,  # Validate every 10 epochs
    "log_interval": 20,  # Log every 20 epochs
    "optimizer": optim_config
}