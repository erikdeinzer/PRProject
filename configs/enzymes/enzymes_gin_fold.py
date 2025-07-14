# Config for GraphSAGE on ENZYMES dataset
# Adjust parameters as needed for your experiments
dataset_name = 'ENZYMES'

backbone = dict(
    type='GINv2',
    in_channels=21,  # Set according to your dataset
    out_channels=128, # Set according to your dataset
    hidden_channels=128,
    num_layers=4,
    norm='batch',
    dropout_rate=0.3,
    act='relu',
)
head = dict(
    type='MLPHead',
    in_channels=128,  # Output channels of the backbone
    out_channels=6,  # Number of classes in PROTEINS dataset
    hidden_channels=64,
    num_layers=3,
    norm='batch',
    dropout_rate=0.3,
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
        dict(type='AddSelfLoops'),  # Adding self-loops to the graph
        dict(type='ToUndirected'),
        dict(type='NormalizeFeatures'),
        dict(type='RemoveIsolatedNodes'),  # Remove isolated nodes
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