# Config for GraphSAGE on MUTAG dataset
# Adjust parameters as needed for your experiments
dataset_name = 'MUTAG'
model = dict(
    type='GraphSAGEv2',
    in_channels=7,  # Set according to your dataset
    out_channels=2, # Set according to your dataset
    hidden_channels=64,
    num_layers=4,
    norm='layer',
    dropout_rate=0.2,
    activation='relu',
)

dataset = dict(
    type='TUDatasetLoader',
    name=dataset_name,
    root='./data',
    transforms=[
        dict(type='ToUndirected'),
        dict(type='NormalizeFeatures'),
        dict(type='RandomNodeSplit'),
    ],
)

optimizer = dict(
    type='Adam',
    lr=0.0001,
    weight_decay=5e-4,
)

train_dataloader = dict(
    batch_size=32,
    shuffle=True,
)

val_dataloader = dict(
    batch_size=1,
    shuffle=False,
)

test_dataloader = dict(
    batch_size=1,
    shuffle=False,
)

runner = dict(
    type='SplitRunner',
    train_ratio=0.8,
    val_interval=1,
    epochs='inf',
    log_interval = 1,
    patience = 20,
)
