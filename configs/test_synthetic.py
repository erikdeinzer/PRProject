# Config for testing GCN with synthetic dataset
# This config uses synthetic data to test model training without internet

dataset_name = 'SYNTHETIC'

backbone = dict(
    type='GCNv2',
    in_channels=4,  # Set according to synthetic dataset
    out_channels=64,
    hidden_channels=64,
    num_layers=3,
    norm='layer',
    dropout_rate=0.2,
    act='relu',
)

head = dict(
    type='MLPHead',
    in_channels=64,  # Output channels of the backbone
    out_channels=2,  # Number of classes in synthetic dataset
    hidden_channels=64,
    num_layers=2,
    norm='layer',
    dropout_rate=0.2,
)

model = dict(
    type='BaseOneStage',
    backbone=backbone,
    head=head,
    pooling='mean',
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
    lr=0.001,  # Much lower learning rate
    weight_decay=1e-5,  # Reduced weight decay
)

lr_scheduler = dict(
    type='StepLR',
    step_size=10,
    gamma=0.9,
)

train_dataloader = dict(
    batch_size=32,
    shuffle=True,
)

val_dataloader = dict(
    batch_size=32,
    shuffle=False,
)

test_dataloader = dict(
    batch_size=32,
    shuffle=False,
)

runner = dict(
    type='SplitRunner',
    train_ratio=0.6,  # 60% train, 20% val, 20% test
    val_interval=1,
    epochs=50,  # Fewer epochs for testing
    log_interval=1,
    patience=10,
    abort_condidtion=0.01,
    lr_scheduler=lr_scheduler,
)