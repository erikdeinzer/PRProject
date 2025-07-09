model = {
    'type': 'GCNv2',
    'in_channels': 'auto',
    'hidden_channels': 64,
    'out_channels': 6,
}

dataset = {
    "type": "TUDatasetLoader",
    "name": "ENZYMES",
    "root": "./data",
    "transforms": [
        {"type": "ToUndirected"},
    ],
}

optim = {
    "type": "Adam",
    "lr": 0.01,
    "weight_decay": 5e-4,
}

dataloader_cfg = {
    "batch_size": 32
}

runner_cfg = {
    'type': 'SplitRunner',
    'train_ratio': 0.8,
    'model': model,
    'dataset': dataset,
    'optim': optim,
    'dataloader_cfg': dataloader_cfg,
    'train_epochs': 50,
    'val_interval': 1,
    'logging_config': {'log_interval': 10},
    'shuffle': True,
}