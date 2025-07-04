"""
GIN SOTA configuration for ENZYMES dataset for comparison.
This configuration uses the existing GIN model with optimized hyperparameters.
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
    'type': 'GINv2',
    'in_channels': 'auto',  # Number of input features - auto chooses from dataset
    'hidden_channels': 128,
    'out_channels': 6,  # Number of enzyme classes
    'num_layers': 5,  # More layers for GIN
}

optim_config = {
    "type": "Adam",
    "lr": 0.01,  # Higher learning rate for GIN
    "weight_decay": 1e-4,
}

split_cfg = {
    'type': 'SplitRunner',
    'train_ratio': 0.8,
    'shuffle': True,
    'random_state': 42,
}

train_config = {
    "batch_size": 64,
    "train_epochs": 200,
    "val_interval": 10,
    "log_interval": 20,
    "optimizer": optim_config
}