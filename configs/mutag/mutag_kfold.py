dataset_name = 'MUTAG'

model = dict(
    type='GCNv2',
    in_channels=7,
    hidden_channels=64,
    out_channels=2,
    num_layers=8
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
    lr=0.01,
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
    type='KFoldRunner',
    s_splits=5,
    epochs='inf',
    train_ratio=0.8,
    val_interval=1,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    test_dataloader=test_dataloader,
    log_interval = 1,
    patience=20,
    abort_condition=0.01,
)
