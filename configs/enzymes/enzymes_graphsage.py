# Config for GraphSAGE on ENZYMES dataset
# Adjust parameters as needed for your experiments
dataset_name = 'ENZYMES'

backbone = dict(
    type='GraphSAGEv2',
    in_channels=21,  # Set according to your dataset
    out_channels=64, # Set according to your dataset
    hidden_channels=64,
    num_layers=2,
    norm='layer',
    dropout_rate=0.5,
    act='relu',
)
head = dict(
    type='MLPHead',
    in_channels=64,  # Output channels of the backbone
    out_channels=6,  # Number of classes in ENZYMES dataset
    hidden_channels=64,
    num_layers=4,
    norm='layer',
    dropout_rate=0.5,
)

model = dict(
    type='BaseOneStage',
    backbone=backbone,
    head=head,
    pooling = 'mean',
)

dataset = dict(
    type='TUDatasetLoader',
    name=dataset_name,
    root='./data',
    transforms=[
        dict(type='ToUndirected'),
        dict(type='NormalizeFeatures'),
    ],
    use_node_attr=True
)

optimizer = dict(
    type='Adam',
    lr=0.001,  # Reduced from 0.01
    weight_decay=1e-5,  # Reduced weight decay
)

lr_scheduler = dict(
    type='StepLR',  # Any torch.optim.lr_scheduler.* class
    step_size=3,
    gamma=0.9,
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
    train_ratio=0.6,  # Better split ratio
    val_interval=1,
    epochs='inf',
    log_interval = 1,
    patience = 20,  # Reduced patience
    lr_scheduler=lr_scheduler,  # Pass scheduler config to runner
)