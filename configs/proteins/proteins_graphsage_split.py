# Config for GraphSAGE on ENZYMES dataset
# Adjust parameters as needed for your experiments
dataset_name = 'PROTEINS'

backbone = dict(
    type='GraphSAGEv2',
    in_channels=4,  # Set according to your dataset
    out_channels=64, # Set according to your dataset
    hidden_channels=64,
    num_layers=5,
    norm='layer',
    dropout_rate=0.2,
    act='relu',
)
head = dict(
    type='MLPHead',
    in_channels=64,  # Output channels of the backbone
    out_channels=2,  # Number of classes in ENZYMES dataset
    hidden_channels=64,
    num_layers=4,
    norm='layer',
    dropout_rate=0.2,
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
    lr=0.001,
    weight_decay=1e-5,
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
    train_ratio=0.8,
    val_interval=1,
    epochs='inf',
    log_interval = 1,
    metric='val_acc',
    direction='max',
    patience = 50,  # Increased patience for better training  
    abort_condidtion=0.01,
    lr_scheduler=lr_scheduler,  # Pass scheduler config to runner
    seed = 1967803736
)