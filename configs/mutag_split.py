dataset_config = {
    "name": "MUTAG",
    "root": "./data",
    "transforms": [{"type": "ToUndirected"}],
}

model_config = {
    'type': 'GCNv2',
    'in_channels': 7,  # Number of input features - auto chooses the input features from the dataset
    'hidden_channels': 64,
    'out_channels': 2,  # Number of classes
}

optim_config = {
    "type": "Adam",
    "lr": 0.01,
    "weight_decay": 5e-4,
}

"""split_cfg = {
    "type": "KFoldRunner",
    "num_folds": 5,
    "shuffle": True,
    "random_state": 42,
}"""

split_cfg = {
    'type': 'SplitRunner',
    'train_ratio': 0.8
}

train_config = {
    "batch_size": 32,
    "num_epochs": 100,
    "optimizer": optim_config
}