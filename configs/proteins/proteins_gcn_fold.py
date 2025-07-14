# Config for GraphSAGE on ENZYMES dataset
# Adjust parameters as needed for your experiments
dataset_name = 'PROTEINS'

backbone = dict(
    type='GCNv2',
    in_channels=4,  # Set according to your dataset
    out_channels=128, # Set according to your dataset
    hidden_channels=128,
    num_layers=2,
    norm='batch',
    dropout_rate=0.2,
    act='relu',
)
head = dict(
    type='MLPHead',
    in_channels=64,  # Output channels of the backbone
    out_channels=2,  # Number of classes in PROTEINS dataset
    hidden_channels=64,
    num_layers=2,
    norm='batch',
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
        dict(type='RemoveIsolatedNodes'),
    ],
    use_node_attr=True
)

optimizer = dict(
    type='Adam',
    lr=5e-3,
    weight_decay=5e-4,
)

lr_scheduler = dict(
    type='ExponentialLR',  # Any torch.optim.lr_scheduler.* class
    gamma=0.95
)

train_dataloader = dict(
    batch_size=16,
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
    type='KFoldRunner',
    n_splits=10,  # Number of folds for K-Fold Cross Validation
    val_interval=1,
    epochs='inf',
    log_interval = 1,
    metric='val_acc',
    direction='max',
    patience = 50,  # Increased patience for better training  
    abort_condition=0.01,
    lr_scheduler=lr_scheduler,  # Pass scheduler config to runner
)